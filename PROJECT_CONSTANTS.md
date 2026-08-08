# Legixo Gen AI Take-Home — locked constants

Read `docs/BUILD_LOG.md` first for phase-by-phase progress. This file is the constants
that must never drift between phases/sessions.

## Provider
- LLM: `openai/gpt-oss-20b` via Together (`TOGETHER_MODEL`). Utility model (grader/rewrite)
  is the same by default but configurable separately (`TOGETHER_UTILITY_MODEL`).
- Embeddings: `intfloat/multilingual-e5-large-instruct` via Together. **1024 dims**,
  **514-token context**. This is the only embedding model on Together's serverless catalog.
- E5 prefixing is mandatory: query side gets
  `Instruct: Given a legal/business question, retrieve the document passage that answers it.\nQuery: {q}`;
  document side gets `passage: {text}`. Constants live in `app/config.py`.

## Pinecone
- Index name: `legixo-qa` (env `PINECONE_INDEX_NAME`). Serverless, `aws` / `us-east-1`,
  metric `cosine`, dimension `1024`.
- Namespace: `legixo-demo` (env `PINECONE_NAMESPACE`).
- Chunk/vector id is deterministic: `f"{source_path}#{chunk_index}"` — re-ingest overwrites
  in place, never duplicates.

## Chunking
- Split markdown on `##` headings first, then token-window at **350 tokens / 60 overlap**,
  hard ceiling **380 tokens** (asserted) because of the 514-token embedding limit.
- Every chunk metadata: `chunk_id, source_path, source_file, heading_path, chunk_index,
  char_start, char_end, content_hash, text`.

## Runtime
- Python **3.11** only (`py -3.11`). 3.14 is the machine default but unsupported by
  `pinecone`/`langchain-together`/`langgraph` deps — do not use it.
- Venv at `C:\Users\Admin\Legixo\.venv`.

## Graph (8 nodes, see docs/langgraph.md once written)
normalize_question → retrieve → grade_context →(branch)→ generate_answer | rewrite_query(loop back to retrieve, capped by MAX_ATTEMPTS) → verify_citations →(branch)→ finalize | no_answer → finalize
- `MAX_ATTEMPTS=2` in state (env-tunable) + `recursion_limit=12` on `.invoke()`.

## API
- `POST /ask`, `POST /admin/ingest` (header `X-Ingest-Token`), `GET /healthz`.
- No CLI for asking questions — ingest CLI only (`python -m scripts.ingest`).

## Full plan
See `C:\Users\Admin\.claude\plans\quiet-jumping-pinwheel.md` for the complete rationale,
schedule, and eval set.
