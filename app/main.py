"""FastAPI app — the only way to ask questions (no CLI for Q&A, per the brief)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from langgraph.errors import GraphRecursionError

from app.config import MissingKeysError, get_settings, require_live_keys
from app.graph.build import build_graph, build_production_deps, recursion_limit_for
from app.ingest import ingest_corpus
from app.schemas import AskRequest, AskResponse, Citation, IngestResponse
from app.vectorstore import VectorStore

app = FastAPI(
    title="Legixo Q&A API",
    version="1.0.0",
    description="Grounded Q&A over a fictional legal corpus. LangGraph + Pinecone + Together.ai.",
)

# Structural backstop against a mis-wired edge causing an infinite loop. Derived
# from MAX_ATTEMPTS (the semantic cap) so raising one can't silently break the
# other — see recursion_limit_for().


@lru_cache(maxsize=1)
def _get_graph():
    """Built once and reused — index handshake and client setup are not per-request work."""
    settings = get_settings()
    return build_graph(build_production_deps(settings))


_INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    """A thin browser client over POST /ask.

    The brief requires Q&A to go through the HTTP API, and it still does — this
    page calls the same endpoint a reviewer would curl. It exists because a
    citation list and a graph trace are far easier to read rendered than as raw
    JSON. Self-contained: no CDN, no build step.
    """
    return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/healthz")
def healthz():
    settings = get_settings()
    try:
        require_live_keys(settings)
    except MissingKeysError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        store = VectorStore(settings)
        stats = store.stats()
    except Exception as exc:  # surfaced as-is so config problems are obvious
        raise HTTPException(status_code=503, detail=f"Pinecone not reachable: {exc}") from exc

    ns_stats = stats.get("namespaces", {}).get(settings.pinecone_namespace, {})
    return {
        "status": "ok",
        "index": settings.pinecone_index_name,
        "namespace": settings.pinecone_namespace,
        "vector_count": ns_stats.get("vector_count", 0),
        "total_vector_count": stats.get("total_vector_count", 0),
        "llm_model": settings.together_model,
        "embed_model": settings.together_embed_model,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        require_live_keys(get_settings())
    except MissingKeysError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    graph = _get_graph()
    initial = {"question": req.question}
    if req.top_k:
        initial["top_k"] = req.top_k

    limit = recursion_limit_for(get_settings().max_attempts)
    try:
        final = graph.invoke(initial, config={"recursion_limit": limit})
    except GraphRecursionError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"graph exceeded recursion limit of {limit} — likely an edge-wiring bug",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AskResponse(
        answer=final.get("answer", ""),
        status=final.get("status", "not_found"),
        citations=[Citation(**c) for c in final.get("citations", [])],
        attempts=final.get("attempt", 0) + 1,
        trace=final.get("trace") if req.include_trace else None,
    )


@app.post("/admin/ingest", response_model=IngestResponse)
def admin_ingest(
    reset: bool = False,
    prune: bool = True,
    x_ingest_token: str = Header(default=""),
):
    settings = get_settings()
    if x_ingest_token != settings.ingest_token:
        raise HTTPException(status_code=401, detail="invalid or missing X-Ingest-Token header")

    result = ingest_corpus(settings, reset=reset, prune=prune)
    return IngestResponse(
        files_processed=result.files_processed,
        chunks_upserted=result.chunks_upserted,
        chunks_pruned=result.chunks_pruned,
        per_file=result.per_file,
    )
