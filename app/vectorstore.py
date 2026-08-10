"""Pinecone wrapper: index lifecycle, upsert, query, prune, reset.

Idempotency contract (see PROJECT_CONSTANTS.md / README "Idempotency"):
  - chunk_id = f"{source_path}#{chunk_index}" is deterministic, so re-running
    ingest on unchanged files upserts the same IDs in place (no duplicates).
  - prune_stale() deletes any previously-seen ID for a source file that the
    latest chunking run no longer produced (file shrank / was edited).
  - reset_namespace() wipes the whole namespace for a guaranteed-clean load.
"""
from __future__ import annotations

import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec, VectorType

from app.config import EMBED_DIMENSION, Settings


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index = None
        self._sparse_index = None

    # ---- index lifecycle -------------------------------------------------

    def ensure_index(self, *, wait_seconds: int = 60) -> None:
        """Create the index if missing. Fails loudly on a dimension mismatch
        rather than silently upserting vectors that won't match the schema."""
        name = self.settings.pinecone_index_name
        existing = {idx["name"]: idx for idx in self._pc.list_indexes()}

        if name not in existing:
            self._pc.create_index(
                name=name,
                dimension=EMBED_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                desc = self._pc.describe_index(name)
                if desc.status.ready:
                    break
                time.sleep(2)
        else:
            desc = existing[name]
            actual_dim = desc.dimension
            if actual_dim != EMBED_DIMENSION:
                raise RuntimeError(
                    f"Pinecone index '{name}' exists with dimension {actual_dim}, "
                    f"but this project requires {EMBED_DIMENSION} "
                    f"(intfloat/multilingual-e5-large-instruct). Use a different "
                    f"PINECONE_INDEX_NAME or delete the existing index."
                )

        self._index = self._pc.Index(name)

    def _get_index(self):
        if self._index is None:
            self.ensure_index()
        return self._index

    def stats(self) -> dict[str, Any]:
        return self._get_index().describe_index_stats().to_dict()

    # ---- writes -------------------------------------------------------

    def upsert(self, vectors: list[dict[str, Any]]) -> int:
        """vectors: [{id, values, metadata}, ...]. Returns count upserted."""
        if not vectors:
            return 0
        index = self._get_index()
        total = 0
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            index.upsert(vectors=batch, namespace=self.settings.pinecone_namespace)
            total += len(batch)
        return total

    def prune_stale(self, source_path: str, keep_ids: set[str]) -> int:
        """Delete previously-upserted vectors for source_path whose id is not
        in keep_ids (i.e. the file now produces fewer chunks than before)."""
        index = self._get_index()
        existing_ids: set[str] = set()
        for page in index.list(
            prefix=f"{source_path}#", namespace=self.settings.pinecone_namespace
        ):
            existing_ids.update(item.id for item in page.vectors)
        stale = existing_ids - keep_ids
        if stale:
            index.delete(ids=list(stale), namespace=self.settings.pinecone_namespace)
        return len(stale)

    def reset_namespace(self) -> None:
        index = self._get_index()
        try:
            index.delete(delete_all=True, namespace=self.settings.pinecone_namespace)
        except Exception:
            # namespace doesn't exist yet on a fresh index — nothing to reset
            pass

    # ---- reads ----------------------------------------------------------

    def query(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        index = self._get_index()
        result = index.query(
            vector=vector,
            top_k=top_k,
            namespace=self.settings.pinecone_namespace,
            include_metadata=True,
        )
        return [
            {
                "id": m["id"],
                "score": m["score"],
                "dense_score": m["score"],
                "metadata": m.get("metadata", {}),
            }
            for m in result.get("matches", [])
        ]

    # ---- sparse / lexical side (hybrid mode) ----------------------------

    def ensure_sparse_index(self, *, wait_seconds: int = 60) -> None:
        """Create the sparse companion index if missing.

        Separate index because Pinecone keeps dense and sparse vectors apart; the
        Starter plan allows 5 indexes, so two is comfortably inside the free tier.
        """
        name = self.settings.pinecone_sparse_index_name
        existing = {idx["name"]: idx for idx in self._pc.list_indexes()}
        if name not in existing:
            self._pc.create_index(
                name=name,
                metric="dotproduct",  # required for sparse
                vector_type=VectorType.SPARSE,
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                if self._pc.describe_index(name).status.ready:
                    break
                time.sleep(2)
        self._sparse_index = self._pc.Index(name)

    def _get_sparse_index(self):
        if self._sparse_index is None:
            self.ensure_sparse_index()
        return self._sparse_index

    def embed_sparse(self, texts: list[str], *, input_type: str) -> list[dict[str, Any]]:
        """Sparse embeddings from Pinecone's hosted model.

        input_type is "passage" for documents and "query" for questions — the
        model treats them differently, same asymmetry as the E5 dense side.
        """
        out: list[dict[str, Any]] = []
        batch = 96  # hosted embed endpoint caps inputs per request
        for i in range(0, len(texts), batch):
            resp = self._pc.inference.embed(
                model=self.settings.pinecone_sparse_model,
                inputs=texts[i : i + batch],
                parameters={"input_type": input_type, "truncate": "END"},
            )
            for e in resp.data:
                out.append({"indices": e["sparse_indices"], "values": e["sparse_values"]})
        return out

    def upsert_sparse(self, vectors: list[dict[str, Any]]) -> int:
        if not vectors:
            return 0
        index = self._get_sparse_index()
        total = 0
        for i in range(0, len(vectors), 100):
            batch = vectors[i : i + 100]
            index.upsert(vectors=batch, namespace=self.settings.pinecone_namespace)
            total += len(batch)
        return total

    def query_sparse(self, sparse_vector: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        index = self._get_sparse_index()
        result = index.query(
            sparse_vector=sparse_vector,
            top_k=top_k,
            namespace=self.settings.pinecone_namespace,
            include_metadata=True,
        )
        return [
            {
                "id": m["id"],
                "score": m["score"],
                "sparse_score": m["score"],
                "metadata": m.get("metadata", {}),
            }
            for m in result.get("matches", [])
        ]

    def reset_sparse_namespace(self) -> None:
        try:
            self._get_sparse_index().delete(
                delete_all=True, namespace=self.settings.pinecone_namespace
            )
        except Exception:
            pass

    # ---- reranking -------------------------------------------------------

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        """Rerank candidates with Pinecone's hosted cross-encoder.

        Unlike bi-encoder retrieval, the reranker sees query and passage together,
        which is what lets it catch the discriminating term (e.g. "retainer") that
        dense similarity drowns under entity-name overlap.
        """
        if not candidates:
            return []
        docs = [
            {"id": c["metadata"]["chunk_id"], "text": c["metadata"].get("text", "")}
            for c in candidates
        ]
        result = self._pc.inference.rerank(
            model=self.settings.rerank_model,
            query=query,
            documents=docs,
            rank_fields=["text"],
            top_n=min(top_n, len(docs)),
            return_documents=False,
            parameters={"truncate": "END"},
        )
        by_id = {c["metadata"]["chunk_id"]: c for c in candidates}
        out = []
        for row in result.data:
            cand = dict(by_id[docs[row["index"]]["id"]])
            cand["rerank_score"] = row["score"]
            cand["score"] = row["score"]
            out.append(cand)
        return out
