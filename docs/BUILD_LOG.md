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

**Not yet committed to git** — first commit happens at the end of this phase, next
message.
