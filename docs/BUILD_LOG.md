# Build log

Append-only. Each phase gets one entry: what got built, what the exit criterion
actually showed, what the next phase needs to know. Read this + `PROJECT_CONSTANTS.md` first
in any new session instead of re-reading the whole repo.

---

## Phase 1 — Ingest path (done)

**Built:** repo scaffold, `.gitignore`/`.env.example` (dummy values) before any key
touched disk, `app/config.py` (pydantic-settings, keys optional so `--dry-run` and
tests work without them), `app/chunking.py` (heading-aware split + 350/60-token
windows, 380-token hard ceiling asserted), `app/llm.py` (`ChatTogether` +
`E5Embeddings` wrapper enforcing the E5 query/passage prefixes), `app/vectorstore.py`
(Pinecone `ensure_index`/`upsert`/`prune_stale`/`reset_namespace`/`query`),
`app/ingest.py` + `scripts/ingest.py` CLI (`--reset`/`--no-prune`/`--dry-run`),
minimal `app/main.py` with `GET /healthz` and `POST /admin/ingest`.

**Exit criterion — met:**
- `pytest tests/` — 6/6 green, fully offline.
- `python -m scripts.ingest --reset` → 15 chunks upserted across the 6 corpus files
  (01:4, 02:4, 03:2, 04:1, 05:1, 06:3).
- `GET /healthz` → `vector_count: 15`.
- Re-ran `python -m scripts.ingest` **without** `--reset` → `chunks upserted: 15`,
  `stale pruned: 0`, `/healthz` still `vector_count: 15`. Idempotency confirmed:
  deterministic `{source_path}#{chunk_index}` ids overwrite in place.

**Real API quirks found (only mattered once live keys were used — not visible in
--dry-run or offline tests):**
- `pinecone` 9.1.0 `IndexModel` has no `.get()` — use attribute access
  (`desc.dimension`), not `.get("dimension")`. Bracket access (`idx["name"]`) does
  work via a custom `__getitem__`.
- `Index.list()` yields `ListResponse` pages with a `.vectors` list of `ListItem`
  objects (each has `.id`) — not a flat iterable of id strings. Fixed in
  `VectorStore.prune_stale`.

**Security note:** the user initially pasted real `TOGETHER_API_KEY` and
`PINECONE_API_KEY` into `.env.example` (the file meant for git, dummy values only).
Caught before any commit — nothing had been committed yet. Moved both to `.env`
(gitignored, confirmed absent from `git status`), restored `.env.example` to
placeholders. No secret ever entered git history.

**State for Phase 2:** Pinecone index `legixo-qa` is live with 15 vectors in
namespace `legixo-demo`, dimension 1024, cosine. `.env` has both real keys. Chunk
metadata schema is `chunk_id, source_path, source_file, heading_path, chunk_index,
char_start, char_end, content_hash, text` — Phase 2's `retrieve` node reads these
directly off Pinecone match metadata, no second lookup needed.

Committed as `127d49e`.

---

## Phase 2 — LangGraph + /ask (done)

**Built:** `app/prompts.py` (grade/rewrite/answer), `app/graph/state.py` (AskState with
`operator.add` reducers on `trace` and `queries`), `app/graph/nodes.py` (8 nodes + 2
routers, deps injected via `GraphDeps` so tests can stub the retriever),
`app/graph/build.py` (StateGraph wiring + `build_production_deps`), `POST /ask` in
`app/main.py` with `GraphRecursionError` → HTTP 500.

**Exit criterion — met and exceeded.** Three known questions returned correct answers
citing the correct source file; also verified 4 more good answers and 4 out-of-corpus
refusals (all `not_found`, 0 citations), including the Harbor Bean Roasters trap.
Trace confirms the loop: retrieve→grade→rewrite→retrieve→grade→rewrite→retrieve→grade→
no_answer→finalize, i.e. exactly `MAX_ATTEMPTS=2` rewrites then abstain.

**Model behaviour verified before building** (cheap de-risking): `gpt-oss-20b` handles
`with_structured_output` correctly in both directions and emits `[S#]` markers without
citing irrelevant chunks.

**Three real defects found and fixed during this phase:**
1. *Grader retry was missing.* Plan called for "tolerant parser, one retry, then fail
   safe"; only the fail-safe existed. A transient API error surfaced it live. Added the
   retry and full exception capture (previously the error text was self-truncated and
   undiagnosable).
2. *Stripping a fabricated marker left the hallucinated claim behind.* Removing just
   `[S9]` left "Arbitration seat is Mumbai ." sitting in the answer as an **uncited**
   assertion — worse than a visible fake citation. Now drops the whole sentence carrying
   the bad marker; if nothing survives, routes to `no_answer`.
3. *False refusals from missing markers (the big one).* The answer model intermittently
   produced a correct answer with **no** `[S#]` marker; verification then rightly
   discarded it, turning a good answer into a refusal. Measured 4/6 pass on "Can the
   lessee sublet Unit 4B?". Fixed with an explicit good/bad example in `ANSWER_SYSTEM`
   plus one corrective retry in `generate_answer` (never fabricates a marker). Now 8/8.

**Calibration data for Phase 3 — important.** E5 score compression is exactly as the plan
predicted: observed `top_score` sits in **0.828–0.857** for *both* answerable and
out-of-corpus questions, with `margin` only **0.015–0.033**. The `RELEVANCE_FLOOR=0.75`
short-circuit therefore **never fires** — every branch decision is currently made by the
LLM grader alone. Phase 3 must either raise the floor into the ~0.83 band (risky, the
bands overlap) or accept that the floor is only a coarse backstop and document it with the
measured distribution. Do not tune the floor without recording the score spread.

**State for Phase 3:** graph is live and stable; both paths verified by hand. Still to do:
`eval/test_cases.json` (15 cases), `scripts/run_eval.py`, `eval/results.md`, and the
offline `tests/test_graph_branch.py` (stub retriever proving good path / loop-cap /
fake-citation strip).

Committed as `7d209df`.

---

## Phase 3 — Calibration, eval, offline branch tests (done)

**Built:** `eval/test_cases.json` (15 cases), `scripts/calibrate.py` → `eval/calibration.md`,
`scripts/run_eval.py` → `eval/results.md` (with `--repeat` for stability),
`tests/test_graph_branch.py` (11 offline tests, stub retriever + stub chat models).

**Exit criterion — met.** `pytest` 17/17 green and fully offline. Final eval run:
**15/15 passed, out-of-corpus refusals 4/4**, at `--repeat 3` (45 live calls).

### Calibration result — the floor cannot do the job, and that's now proven

Measured over all 15 cases (embeddings only, no LLM):

| set | min | mean | max |
|---|---|---|---|
| answerable `top_score` | 0.8383 | 0.8795 | 0.9170 |
| out-of-corpus `top_score` | 0.8391 | 0.8563 | 0.8719 |

The out-of-corpus band sits **inside** the answerable band. No absolute floor separates
them; anything above 0.8383 causes false refusals. Margins overlap too (answerable
0.0372–0.0992 vs out-of-corpus 0.0152–0.0722), so the relative test doesn't rescue it.

**Decision: `RELEVANCE_FLOOR` stays 0.75** — deliberately far below the answerable
minimum so it can never cause a false refusal. It is a coarse backstop for degenerate
retrieval only; the LLM grader makes the real call. Retrieval recall was **15/15**, which
proves failures here are grading failures, not retrieval failures. Full write-up in
`eval/calibration.md`.

### Defects found and fixed this phase

1. **`recursion_limit` was hardcoded to 12 — a real latent production bug.** The
   parametrised loop-cap test caught it: `MAX_ATTEMPTS=3` needs 14 steps, so raising the
   (env-tunable!) `MAX_ATTEMPTS` turned graceful refusals into HTTP 500s. Replaced with
   `recursion_limit_for(max_attempts) = 3*max_attempts + 8`, derived from the worst-path
   walk, plus a test asserting it covers attempts 1–5. Default `MAX_ATTEMPTS=2` needed 11
   and fit under 12 by a single step — it was one config change from breaking.
2. **`eval/results.md` was gitignored** (added in Phase 1 by reflex). The brief names the
   self-test file with pass/fail notes as a deliverable, so it must ship. Un-ignored.
3. **Eval harness reported the wrong run.** With `--repeat`, the transcript written for a
   failing case came from the *last* run, which was often a passing one — making
   intermittent failures undiagnosable. Now records a failing run's transcript when one
   exists.
4. **Two unfair eval assertions**, fixed as eval bugs rather than by weakening the check:
   case 3 expected `billing head` but the model writes `billing-head` (hyphens now
   normalised to spaces on both sides); case 7 asked "when is mediation mandatory" while
   asserting the 30-day duration — the question didn't ask for it, so the question was
   reworded to request both facts.

### Honest note on run-to-run variance

The system is **not** perfectly deterministic. Across three full `--repeat 3` sweeps
(~135 case-runs) two individual runs failed: case 4 once (false refusal) and case 6 once.
That is roughly a 1–2% per-case-run failure rate, all in the same direction — refusing a
question it could have answered, never fabricating an answer or a citation. The final
recorded run is a clean 15/15, but a reviewer re-running it may see 14/15. Reducing this
further would mean self-consistency on the grader (2-of-3 voting), which triples grader
cost for a marginal gain — noted as a deliberate trade-off, not an oversight.

**State for Phase 4:** all code paths done and verified. Remaining: `README.md`,
`docs/langgraph.md`, the video script, LangSmith wiring behind
`LANGCHAIN_TRACING_V2`, then the clean-clone rehearsal.

Committed as `c12402a`.

---

## Phase 4 — Docs + LangSmith (done)

**Built:** `README.md` (install → keys → Pinecone setup → ingest → serve → curl, with the
idempotency answer the brief asks for and a stated limitations section),
`docs/langgraph.md` (mermaid diagram, 8-node table, branch logic, limit derivation),
the video script (timed shot list covering every item the brief requires on video),
LangSmith wiring behind `LANGCHAIN_TRACING_V2`.

**Defect found and fixed: LangSmith tracing would have silently done nothing.**
`app/main.py` never called `load_dotenv()`, and pydantic-settings' `env_file` only
populates the `Settings` object — it does **not** write to `os.environ`. LangChain's tracer
reads `LANGCHAIN_*` straight from `os.environ`, so setting `LANGCHAIN_TRACING_V2=true` in
`.env` and starting via uvicorn would have produced no traces at all, with no error. Added
`_apply_langsmith_env()` in `build.py`, called from `build_production_deps`. Verified both
directions: disabled leaves the env var absent, enabled sets tracing + project. Also pinned
`langsmith` explicitly in `requirements.txt` rather than relying on it as a transitive dep.

**Verification — every documented command was actually executed, not assumed:**
- `pytest tests/ -q` → 17/17.
- `--dry-run` with **both API keys unset** in a temp dir → 6 files / 15 chunks, confirming
  the README's "no API keys needed" claim for that flag.
- `GET /healthz` → 15 vectors.
- Good-path curl → answered, cited `corpus/02_employment_agreement_excerpt.md#1`.
- Refusal curl (Harbor Bean Roasters trap) → `not_found`, 0 citations, 3 attempts.
- `POST /admin/ingest` with valid token → 15 chunks; with a wrong token → **401**.

**State for Phase 5:** all deliverables written. Remaining: clean-clone rehearsal in a
fresh directory and fresh venv following only the README, final secret sweep of git
history, and the video recording (user action).
