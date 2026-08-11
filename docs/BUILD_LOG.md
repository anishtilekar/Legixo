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

> **SUPERSEDED IN PHASE 5 — this diagnosis was wrong.** The variance below was not
> inherent LLM flakiness; it was caused by never setting `temperature`, so Together
> applied its own sampling default (~0.7) to what are pure classification and extraction
> calls. Setting `temperature=0` eliminated it. See Phase 5. Leaving the original text
> here because the *measurements* were real — only the attribution was wrong.

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

Committed as `7733eae`.

---

## Phase 5 — Clean-clone rehearsal (done)

Cloned the repo to a fresh directory with a fresh 3.11 venv and followed **only** the
README — no memory, no shortcuts. Three real problems surfaced that no amount of
re-reading would have caught.

### 1. `temperature` was never set — the big one

**This is the most important finding in the whole project, and it invalidates the Phase 3
variance diagnosis.**

`ChatTogether`'s `temperature` field defaults to `None`, which means the parameter is
never sent to the API, so Together applies its own server-side default (~0.7). Every LLM
call in this app is a *decision* or a *grounded extraction* — grade relevance, rewrite a
query, answer from supplied text. None of them want creative sampling.

The symptom was the thing I'd already documented and misattributed: identical questions
flipping between `answered` and `not_found` across runs. During Phase 5 an eval run
degraded to **11/15** (four separate cases false-refusing), which was far outside the
"1–2% inherent variance" I had written down — that gap is what prompted looking at
temperature at all.

Ruled out first, so the diagnosis isn't a guess: not rate limiting (8/8 in a rapid burst,
zero grader exceptions, no 429s in the server log), and not retrieval (recall was already
measured at 15/15 in Phase 3).

Measured effect of `temperature=0.0`:

| | result |
|---|---|
| Together default sampling | 11/15 → 15/15, varying run to run |
| `temperature=0.0`, sweep 1 (`--repeat 3`) | **15/15, every case 3/3** |
| `temperature=0.0`, sweep 2 (`--repeat 3`) | **15/15, every case 3/3** |

90 consecutive case-runs with zero failures. The "known flakiness" I'd chosen to document
rather than fix in Phase 3 was a one-line config defect, not a property of the model. Fixed
in `app/llm.py` with the reasoning recorded at the call site.

### 2. Placeholder keys produced an unusable error

Copying `.env.example` → `.env` and forgetting to swap the keys (exactly what a reviewer in
a hurry does) gave a 14-line SDK traceback ending in `[401] Invalid API key`. Accurate,
useless. Added `require_live_keys()` — fails before any network call, names the specific
offending variable, links only the provider(s) actually needed, and points at the
`--dry-run` path that needs no keys. Covered by `tests/test_config_preflight.py`.

### 3. Eval matcher missed a third number format

The clean clone produced `₹1 35 000` (space-separated digit grouping) where earlier runs
gave `₹1,35,000` — a factually correct answer scored as a failure. That's the third
formatting variant the matcher has had to absorb (commas, hyphens, now spaces). Collapse
separators only *between digits*, so `45 days` stays untouched.

### Rehearsal verification

- Fresh clone contains **no `.env`** — confirmed.
- `pip install -r requirements.txt` from scratch — clean.
- `pytest tests/ -q` with **no `.env` at all** → 21/21.
- `python -m scripts.ingest --dry-run` with **no keys** → 6 files / 15 chunks.
- `python -m scripts.ingest --reset` with real keys → 15 chunks, output matches the README
  byte for byte.
- `GET /healthz` → 15 vectors; good-path curl → answered with citation; trap question →
  `not_found`, 0 citations.
- Full git history scanned blob-by-blob for `tgp_v1_*` / `pcsk_*` / `ls__*` — **clean**.
  `.env` was never tracked in any commit.

**Remaining (user action only):** record the demo video from the video script, create
the GitHub remote, and push.

---

## Phase 6 — Corpus + eval expansion (PAUSED MID-PHASE)

> **Status: corpus and eval done, one real defect fixed, 6 failures still open.**
> Paused at user request. Resume instructions at the bottom of this entry.

**Built:** corpus grown 6 → **30 files**, 15 → **93 chunks**. Eval grown 15 → **33 cases**
(24 single-source, 3 multi-source, 6 out-of-corpus). `expected_sources_any` semantics added
to `run_eval.py` + `calibrate.py` for facts legitimately stated in more than one document.

**Why the corpus grew:** at 15 chunks, `TOP_K=5` returned a third of the corpus on every
query, so retrieval recall was a meaningless 15/15 and the brief's remaining extras
(reranker, hybrid) had nothing to improve. At 93 chunks a query sees ~5%, so retrieval is
finally a real problem. **Deliberate distractors** were built in: three employment
agreements with different notice periods (Bluecrest 60d / Vantage 30d / Meridian 90d) and
three leases with different units and deposits, so answering requires picking the right
*document*, not just the right topic.

### Defect found and fixed: grader input was truncated

`_GRADER_SNIPPET_CHARS = 300` (a Phase 2 token optimisation) fed the grader only the first
300 characters of each chunk. At the original scale every fact sat inside that window. At
30 files it does not — e.g. the court-fee cap lives in "Item 4", past the Item 1 preamble.
The grader saw a genuinely incomplete excerpt, correctly reported it could not see the
fact, and the question was refused **even though the right chunk was retrieved at rank 1**.

That optimisation was saving well under a cent per run and costing ~30% of the eval.
Grader now gets full chunk text (bounded anyway by the 380-token chunk ceiling).

**Effect: 23/33 → 27/33.** Out-of-corpus refusals held at **6/6** throughout — grounding
discipline never regressed, which is the failure direction that matters.

### Calibration at scale — a much stronger result than at n=15

| set | min | mean | max |
|---|---|---|---|
| answerable `top_score` | 0.8385 | 0.8766 | 0.9170 |
| out-of-corpus `top_score` | 0.8419 | 0.8634 | **0.9085** |

The overlap is now *worse*, and far more convincingly so: case 32 ("Rohit Desai's salary")
scores **0.9085** — higher than almost every answerable question — because it retrieves his
employment agreement, which is topically perfect and simply never states pay. This is a
much better demonstration than the n=15 version that **no similarity threshold can decide
groundedness**. `RELEVANCE_FLOOR` stays 0.75 as a coarse backstop.

File-level retrieval recall: **32/33**. Note this metric is *generous* — it asks whether
the right file appeared in top-k, not whether the chunk containing the answer did.

### The 6 open failures — all retrieval-precision, diagnosed not guessed

Target-document rank in top-5 for the failing cases:

| case | target rank | what outranked it |
|---|---|---|
| 1 (Bluecrest notice) | 1 | correct chunk is rank 1 — grader/answer issue, not retrieval |
| 3 (next hearing) | **not in top-5** | witness statement, counsel notes, invoice summary |
| 16 (Copperline judgment) | 2 | engagement letter |
| 23 (Copperline retainer) | 5 | three chunks of the IP assignment |
| 27 (declared value + award) | 5 | IP assignment, engagement letter |

The pattern is clear and consistent: **dense E5 locks onto the entity name ("Copperline")
and misses the discriminating term ("retainer", "declared value")**. That is precisely the
weakness lexical/sparse retrieval fixes, so Phase 7 is now motivated by measured evidence
rather than by the brief listing it as an extra.

### Phase 6/7 outcome — 28/33 → 33/33 (superseding the "resume here" notes below)

Four fixes, every one found by tracing a single failing question end to end rather than by
tuning parameters. Measured after each.

| # | Defect | Effect |
|---|---|---|
| 1 | **Chunks embedded without document context.** A judgment's "Relief granted" section is only `Damages: ₹8,50,000 / Interest: 9% / Costs: ₹65,000` — no party name, no case number. "What was awarded in Copperline v. Vantage" could not retrieve the chunk holding the answer; it sat outside the top 12 while the title chunk ranked 2nd. Heading path alone was insufficient too: that file is titled by case *number*, while party *names* live in the `**Matter:**` line. Embedding input is now document context + heading path + body. | 28→29, recall 26/27→**27/27** |
| 2 | **Citation markers in full-width CJK brackets were unparseable.** The model emits `【S1】` (U+3010/3011) often enough to matter; the ASCII-only regex found zero citations and converted a *correct, properly cited* answer into a refusal. Case 24 had perfect retrieval (rank 1, score 0.8999) and the grader had explicitly confirmed the fact — only the bracket glyph differed. | 29→**32** |
| 3 | **Query rewrites restated the question.** Context accumulates across attempts, so the useful second query hunts only the *missing* fact — which the grader already names ("gives the declared value but not the award"). The reason was passed to the prompt but never used to narrow the search. | last case reachable at rank 8 |
| 4 | **`TOP_K=5` too narrow for 93 chunks.** Chosen when the corpus was 15 chunks and k=5 returned a third of it. | **33/33** |

**The `TOP_K` result inverts the usual intuition: a wider window was both more accurate and
faster** (mean 8.9s → 7.4s). With k=5 the answering chunk is often absent, so the question
burns the whole rewrite loop — three retrievals, two rewrites, a regeneration — which costs
far more than one pass over a wider window. Narrow retrieval was buying loops, not saving
tokens.

### Ablation verdict (`eval/ablation.md`)

| config | eval | OOC | recall | latency |
|---|---|---|---|---|
| `dense` (k=5) | 32/33 | 6/6 | 27/27 | 8.9s |
| **`dense k=8`** ← default | **33/33** | 6/6 | 27/27 | **7.4s** |
| `dense k=12` | 33/33 | 6/6 | 27/27 | 7.8s |
| `dense k=12 +rerank` | 31/33 | 6/6 | 27/27 | 9.5s |
| `hybrid k=12` | 33/33 | 6/6 | 27/27 | 11.2s |

k=8 and k=12 both hit 33/33 in **two independent runs**, which is why the default moved.
**Reranking measurably hurt** (31/33) in both ablations — it is implemented, tested, and
left off. **Hybrid matched dense** but costs 50% more latency plus a second index for zero
gain. Out-of-corpus refusals held **6/6 in every configuration**, throughout every fix.

An earlier ablation concluded "dense beats hybrid and rerank" while the embedding bug was
still present; that run was discarded and re-measured rather than reported.

---

## Phase 8 — CI, lint, Docker (PARTIAL — paused)

**Done and verified:**
- `.github/workflows/ci.yml` — Python 3.11, `ruff`, `pytest`, a keyless `--dry-run` ingest,
  and a step that **fails the build if a live-looking API key is committed**. No secrets are
  configured for the workflow on purpose: the suite stubs the retriever and both chat
  models, so if a test ever needs a key, that test is wrong.
- `ruff.toml` (`E,F,W,I,UP,B`) — **lint now clean**, 38/38 tests still pass.
  Two of the fixes were real, not cosmetic:
  - `zip()` calls in `ingest.py` now use `strict=True`. A mismatch between chunks and
    returned embeddings would previously have silently dropped chunks from the index.
  - `build_production_deps` no longer redefines `reranker` conditionally (F811).
- `.dockerignore` excludes `.env` so secrets can't be baked into an image layer.

**NOT verified — `Dockerfile` was never built.** The Docker daemon would not start on this
machine (`dockerDesktopLinuxEngine` pipe missing; Docker Desktop was launched and did not
come up within 5 minutes). The file is written from the same pinned requirements as the
tested path, but it has not been built or run once. This is flagged in the README with a
NOTE block rather than presented as working. **Next session: build it, run it, confirm
`/healthz`, then remove the caveat — or delete the Dockerfile if it does not work.**

**Also outstanding from this phase:** retry/backoff on transient provider errors was
planned and not implemented.

### Resume checklist

1. Verify or drop the Dockerfile (above).
2. Re-run `python -m scripts.ablation` with the **relabelled** configs — every config now
   pins `top_k` explicitly, because the `dense` row silently stopped meaning k=5 once the
   default moved to 8. The committed `eval/ablation.md` still carries the older labels; its
   numbers are real but the `dense` row means **k=5**.
3. Phase 9: web UI at `GET /`, final clean-clone rehearsal, README/demo-script refresh.
4. README CI badge has a placeholder `OWNER/REPO` to fill in once the remote exists.

### Superseded resume notes (kept for the record)

1. Case 1 is the odd one out — target at rank 1 yet still refused. Diagnose separately
   (likely the answer node or grader strictness, *not* retrieval).
2. Then Phase 7: sparse index + RRF fusion + `bge-reranker-v2-m3` node, config-gated, and
   `scripts/ablation.py` across the 4 configs. Cases 3/16/23/27 are the natural before/after
   evidence.
3. the video script, `README.md` and `PROJECT_CONSTANTS.md` still quote the **old 6-file / 15-chunk
   / 15-case** numbers — all must be updated before the video.
4. Phases 8–9 (CI, Docker, robustness, web UI) not started.
