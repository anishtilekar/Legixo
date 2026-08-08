"""FastAPI app. Phase 1: /healthz + /admin/ingest only. /ask lands in Phase 2."""
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException

from app.config import get_settings
from app.ingest import ingest_corpus
from app.schemas import IngestResponse
from app.vectorstore import VectorStore

app = FastAPI(title="Legixo Q&A API", version="0.1.0")


@app.get("/healthz")
def healthz():
    settings = get_settings()
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
    }


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
