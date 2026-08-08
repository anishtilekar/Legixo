# Legixo Q&A API — grounded question answering over a document corpus

A small HTTP API that answers questions **only** from a document set, shows which chunk
each answer came from, and says it cannot find something rather than guessing.

Built with **Python 3.11**, **LangGraph** (8-node `StateGraph` with a branch and a capped
loop), **Pinecone** (real serverless index), and **Together.ai** for the LLM and embeddings.

> The corpus in `corpus/` is fiction — made-up parties, courts, and facts, supplied with the
> take-home. No real client data is used anywhere in this project.

---

## Quick start

Five commands from clone to answered question. Full detail in the sections below.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m scripts.ingest --reset
```

Then start the server and ask something:

```bash
python -m uvicorn app.main:app --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is the notice period in the Bluecrest employment agreement?\"}"
```

---

## 1. Prerequisites

**Python 3.11 is required** — not 3.12+, and definitely not 3.14. `pinecone`,
`langchain-together`, and `langgraph` all cap support at 3.13, and 3.14 hits wheel build
failures. Check with:

```bash
py -3.11 --version
```

You also need two free API keys:

| Key | Where to get it | Used for |
|---|---|---|
| `TOGETHER_API_KEY` | [api.together.xyz](https://api.together.xyz) → Settings → API Keys | LLM + embeddings |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → API Keys | Vector index |

Pinecone's free tier allows **one index** — if you already have one, either delete it or
set `PINECONE_INDEX_NAME` to reuse it (the dimension must be 1024).

## 2. Install

```bash
git clone <your-repo-url>
cd Legixo
py -3.11 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure

```bash
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Open `.env` and replace the two dummy keys with real ones. **`.env` is gitignored — never
commit it.** `.env.example` contains dummy values only and is the file that ships.

| Variable | Default | Notes |
|---|---|---|
| `TOGETHER_API_KEY` | — | **Required.** |
| `TOGETHER_MODEL` | `openai/gpt-oss-20b` | Answering model. $0.05/$0.20 per 1M tokens. |
| `TOGETHER_UTILITY_MODEL` | `openai/gpt-oss-20b` | Grading + rewriting. Can be set to a cheaper model independently. |
| `TOGETHER_EMBED_MODEL` | `intfloat/multilingual-e5-large-instruct` | The only embedding model on Together's serverless catalog. 1024-dim. |
| `PINECONE_API_KEY` | — | **Required.** |
| `PINECONE_INDEX_NAME` | `legixo-qa` | Created automatically if missing. |
| `PINECONE_NAMESPACE` | `legixo-demo` | Isolates this corpus within the index. |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Free-tier serverless region. |
| `TOP_K` | `5` | Chunks retrieved per query. |
| `MAX_ATTEMPTS` | `2` | Retrieval retries before giving up. See [docs/langgraph.md](docs/langgraph.md). |
| `RELEVANCE_FLOOR` | `0.75` | Coarse backstop only — see [Why the floor is low](#why-the-relevance-floor-is-deliberately-low). |
| `INGEST_TOKEN` | `dummy-ingest-token` | Shared secret for `POST /admin/ingest`. |
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` + add `LANGCHAIN_API_KEY` for LangSmith tracing. |

## 4. Pinecone index setup

**You do not need to create the index by hand.** `ensure_index()` creates it on first
ingest with the correct settings:

- **dimension `1024`** (fixed by the embedding model — mismatches fail loudly rather than
  upserting garbage)
- metric **`cosine`**
- **serverless**, `PINECONE_CLOUD` / `PINECONE_REGION` (default `aws` / `us-east-1`)

If an index of that name already exists with a different dimension, ingest aborts with an
explicit error instead of corrupting it.

## 5. Ingest the corpus

```bash
python -m scripts.ingest --reset
```

Expected output — 6 files, 15 chunks:

```
files processed:  6
  corpus/01_matter_memo_arvind_v_northfield.md: 4 chunks
  corpus/02_employment_agreement_excerpt.md: 4 chunks
  corpus/03_hearing_notice_template.md: 2 chunks
  corpus/04_statute_style_excerpt_fictional.md: 1 chunks
  corpus/05_counsel_notes_settlement.md: 1 chunks
  corpus/06_property_lease_clause.md: 3 chunks
chunks upserted:  15
stale pruned:     0
```

### What happens if you run ingest twice?

**Nothing bad — it overwrites in place and never duplicates.** Chunk IDs are deterministic:

```
chunk_id = f"{source_path}#{chunk_index}"     e.g. corpus/02_employment_agreement_excerpt.md#1
```

Re-running upserts the *same* IDs, so the vector count stays at 15. Verified: two
consecutive ingests both report `chunks upserted: 15` and `/healthz` reports
`vector_count: 15` both times.

| Flag | Behaviour |
|---|---|
| *(none)* | Upsert all chunks by deterministic ID, then prune stale IDs. Safe to repeat. |
| `--reset` | Delete the entire namespace first, then load. Use for a guaranteed-clean state. |
| `--no-prune` | Skip stale-ID cleanup. |
| `--dry-run` | Chunk only — no embedding, no Pinecone, **no API keys needed**. Useful to inspect chunking. |
| `--path DIR` | Ingest a different folder. |

`--prune` (on by default) handles the one gap deterministic IDs leave: if a file is *edited
to produce fewer chunks*, the leftover trailing vectors would otherwise linger. Pruning
deletes IDs previously seen for that file that the new run no longer produces.

### Ingest over HTTP instead

If you'd rather not use the CLI:

```bash
curl -X POST "http://127.0.0.1:8000/admin/ingest?reset=true" -H "X-Ingest-Token: dummy-ingest-token"
```

## 6. Run the server

```bash
python -m uvicorn app.main:app --port 8000
```

Interactive Swagger UI: **http://127.0.0.1:8000/docs**

Confirm vectors are really in Pinecone:

```bash
curl http://127.0.0.1:8000/healthz
```

```json
{"status":"ok","index":"legixo-qa","namespace":"legixo-demo","vector_count":15,
 "total_vector_count":15,"llm_model":"openai/gpt-oss-20b",
 "embed_model":"intfloat/multilingual-e5-large-instruct"}
```

## 7. Ask questions

### A question the documents can answer

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is the notice period in the Bluecrest employment agreement?\"}"
```

```json
{
  "answer": "The notice period is **60 days** written notice [S2].",
  "status": "answered",
  "citations": [
    {
      "marker": "[S2]",
      "chunk_id": "corpus/02_employment_agreement_excerpt.md#1",
      "source_path": "corpus/02_employment_agreement_excerpt.md",
      "heading_path": "Employment agreement excerpt — Bluecrest Analytics (fiction) > Notice period",
      "score": 0.8387,
      "snippet": "Either party may end this agreement by giving **60 days** written notice..."
    }
  ],
  "attempts": 1
}
```

### A question the documents *cannot* answer

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is the notice period at Harbor Bean Roasters?\"}"
```

```json
{
  "answer": "I could not find this in the provided documents. The corpus does not appear to contain the specific information needed to answer this question.",
  "status": "not_found",
  "citations": [],
  "attempts": 3
}
```

That question is a deliberate trap: it splices the notice-period concept (from the
Bluecrest employment agreement) onto the tenant in the lease. Retrieval returns
confident-looking chunks from both files — only a genuine grounding check refuses it.

### Watch the graph work

Add `"include_trace": true` to see every node, the branch decision, and the loop:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is the notice period at Harbor Bean Roasters?\",\"include_trace\":true}"
```

```
normalize_question   question normalized (50 chars)
retrieve             query='What is the notice period at Harbor Bean Roasters?' hits=5
grade_context        LLM grader                          grade=insufficient
rewrite_query        attempt 1 -> 'Harbor Bean Roasters lease notice period clause'
retrieve             query='Harbor Bean Roasters lease notice period clause' hits=5
grade_context        LLM grader                          grade=insufficient
rewrite_query        attempt 2 -> 'Harbor Bean Roasters commercial lease termination notice'
retrieve             query='Harbor Bean Roasters commercial lease termination notice' hits=5
grade_context        LLM grader                          grade=insufficient
no_answer            abstained after 3 retrieval(s)
finalize             status=not_found
```

### API reference

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{question, top_k?, include_trace?}` → `{answer, status, citations[], attempts, trace?}` |
| `GET /healthz` | Config + live Pinecone vector count |
| `POST /admin/ingest` | Ingest over HTTP. Requires `X-Ingest-Token` header. |
| `GET /docs` | Swagger UI |

`status` is `"answered"` or `"not_found"`. There is **no CLI for asking questions** — the
HTTP API is the only Q&A interface, per the brief.

---

## Architecture

```
POST /ask
   │
   ▼
normalize_question → retrieve → grade_context ──sufficient──→ generate_answer
                        ▲            │                              │
                        │            ├─insufficient & attempt<max──→ rewrite_query
                        └────loop────┘                              │
                                     └─insufficient & attempt≥max─→ no_answer
                                                                    ▲
                        generate_answer → verify_citations ─0 real citations─┘
                                                │
                                          ≥1 real citation → finalize → END
```

Full node table, branch logic, and limit derivation: **[docs/langgraph.md](docs/langgraph.md)**.

Key design points:

- **Citation verification is its own node.** Every `[S#]` is resolved back to a real
  retrieved chunk. Unresolvable markers cause the whole sentence to be dropped — stripping
  just the marker would leave a hallucinated claim sitting there *uncited*, which is worse.
  Zero surviving citations routes to `no_answer`.
- **Two independent limits.** `MAX_ATTEMPTS` is the semantic cap; `recursion_limit` is a
  structural backstop, *derived* as `3*max_attempts + 8` rather than hardcoded.
- **Chunking is heading-aware**, then token-windowed at 350 tokens with 60 overlap, capped
  hard at 380 — because the embedding model's context is only 514 tokens and silently
  truncates beyond it.
- **E5 prefixes are enforced in one place** (`app/llm.py`). E5 models are trained with
  asymmetric query/passage prefixes; skipping them measurably degrades retrieval.
- **All LLM calls run at `temperature=0`.** Grading, rewriting, and answering are decisions
  and extractions, not creative writing. Leaving temperature unset sends nothing to the API
  and lets the provider default (~0.7) apply, which made identical questions flip between
  `answered` and `not_found`; pinning it to 0 took the eval from a fluctuating 11–15/15 to
  15/15 across two consecutive `--repeat 3` sweeps (90 case-runs, zero failures).

---

## Self-test / eval

15 cases: 10 single-source, 1 multi-source, 4 out-of-corpus. Start the server, then:

```bash
python -m scripts.run_eval --repeat 3
```

Latest recorded run: **15/15 passed, out-of-corpus refusals 4/4.**

- Cases and expected citations: [`eval/test_cases.json`](eval/test_cases.json)
- Per-case results and notes: [`eval/results.md`](eval/results.md)

Each case asserts three things: the status matches, every expected source file appears in
the citations, and every expected fact appears in the answer. Out-of-corpus cases
additionally require **zero** citations.

### Why the relevance floor is deliberately low

`RELEVANCE_FLOOR=0.75` looks low. It was measured, not guessed:

```bash
python -m scripts.calibrate      # writes eval/calibration.md
```

| set | min | mean | max |
|---|---|---|---|
| answerable `top_score` | 0.8383 | 0.8795 | 0.9170 |
| out-of-corpus `top_score` | 0.8391 | 0.8563 | 0.8719 |

The out-of-corpus band sits **inside** the answerable band. No absolute threshold can
separate them — anything above 0.8383 starts causing false refusals. This is E5 score
compression on a small corpus: with only 15 chunks, `top_k=5` returns a third of the
corpus on every query, and everything looks somewhat relevant.

So the floor is a coarse backstop for degenerate retrieval only, and the **LLM grader makes
the real relevance decision**. Retrieval recall is 15/15, which confirms that any failure
here is a grading failure, not a retrieval one. Full write-up:
[`eval/calibration.md`](eval/calibration.md).

### Offline tests

```bash
pytest tests/ -q      # 17 tests, no network, no API keys
```

`tests/test_graph_branch.py` drives the whole graph with a stub retriever and stub chat
models, proving: the good path answers, the bad path loops **exactly** `MAX_ATTEMPTS`
times then abstains, and a fabricated `[S9]` never survives.

---

## Optional: LangSmith tracing

Set in `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=legixo-qa-takehome
```

Every graph run then appears in LangSmith with per-node timings. Tracing is off by default
and the app runs identically without it.

---

## Cost

Roughly **$0.0003 per question** (~4,000 input + ~310 output tokens across two LLM calls).
A full 15-case eval at `--repeat 3` costs about **$0.012**. Ingest is a one-time ~1,200
embedding tokens. All development and testing for this project came to well under $0.50.

Every node caps `max_tokens` (answer 1024, utility 512) so a malformed-output retry storm
costs cents rather than a balance.

---

## Known limitations

Stated plainly rather than left for you to find:

- **Some run-to-run variance remains.** All LLM calls run at `temperature=0`, which removed
  the large majority of it (see below), but Together's serverless inference is not
  bit-for-bit deterministic, so an occasional borderline grade can still flip. When it
  does, it fails in the safe direction — *refusing a question it could have answered*,
  never fabricating an answer or a citation.
- **Small-corpus retrieval.** With 15 chunks and `top_k=5`, retrieval alone cannot
  discriminate — see the calibration section above. On a larger corpus the relevance floor
  would start doing real work.
- **No hybrid search or reranking.** Both were considered and skipped: on a corpus this
  small the gain is marginal (see calibration — recall is already 15/15, so the bottleneck
  is grading, not retrieval).
- **Markdown-only ingest.** `scripts/ingest.py` globs `*.md`. PDF/DOCX would need a loader.
- **Single-turn only.** No conversation memory; each `/ask` is independent.

## Project layout

```
app/
  config.py        settings + fixed embedding constants
  chunking.py      heading-aware markdown splitter with token ceiling
  llm.py           Together chat + E5 embeddings (prefixes enforced here)
  vectorstore.py   Pinecone: ensure_index / upsert / prune / reset / query
  ingest.py        load → chunk → embed → upsert
  prompts.py       grade / rewrite / answer prompts
  main.py          FastAPI: /ask, /healthz, /admin/ingest
  graph/
    state.py       AskState TypedDict with additive reducers
    nodes.py       the 8 nodes + 2 routing functions
    build.py       StateGraph wiring + recursion_limit_for()
scripts/
  ingest.py        ingest CLI
  calibrate.py     retrieval score distribution → eval/calibration.md
  run_eval.py      eval runner → eval/results.md
eval/              test cases, results, calibration write-up
tests/             offline tests (no network, no keys)
docs/              langgraph.md, BUILD_LOG.md
corpus/            the 6 fiction documents
```
