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
