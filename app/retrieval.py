"""Retrieval strategies: RRF fusion for hybrid search, and reranking.

Kept out of the graph nodes so the fusion maths is a pure function that tests can
drive without any network.

Why RRF rather than score blending: dense cosine (~0.83–0.92, tightly compressed)
and sparse dot-product (unbounded) live in different spaces, so a weighted sum
needs normalisation constants that are themselves a tuning problem. Reciprocal
Rank Fusion uses only *rank*, so it sidesteps that entirely.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Standard RRF constant from Cormack et al. Large k flattens the contribution
# curve so no single ranker dominates on its top hit alone.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[dict[str, Any]]],
    *,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse several ranked candidate lists into one.

    Each item must carry `metadata.chunk_id`. Per-retriever scores are preserved
    on the fused item (`dense_score` / `sparse_score`) because the relevance floor
    and the citations both need a real similarity number, not an RRF score — an
    RRF score is ~0.016 and would trip any absolute threshold.
    """
    fused: dict[str, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            chunk_id = item["metadata"]["chunk_id"]
            entry = fused.get(chunk_id)
            if entry is None:
                entry = dict(item)
                entry["rrf_score"] = 0.0
                fused[chunk_id] = entry

            entry["rrf_score"] += 1.0 / (k + rank)

            # keep whichever per-retriever scores this item carried
            for field in ("dense_score", "sparse_score"):
                if item.get(field) is not None:
                    entry[field] = item[field]

    out = sorted(fused.values(), key=lambda c: c["rrf_score"], reverse=True)
    for item in out:
        # `score` is what citations display and what ordering means here.
        # Prefer a real similarity when we have one.
        if item.get("dense_score") is not None:
            item["score"] = item["dense_score"]
        elif item.get("sparse_score") is not None:
            item["score"] = item["sparse_score"]
    return out


def max_dense_score(candidates: list[dict[str, Any]]) -> float:
    """Highest dense cosine among candidates, ignoring sparse-only hits.

    The relevance floor was calibrated against dense cosine specifically, so it
    must keep reading dense cosine in every retrieval mode — otherwise switching
    to hybrid would silently change what the floor means.
    """
    scores = [c["dense_score"] for c in candidates if c.get("dense_score") is not None]
    if scores:
        return max(scores)
    return max((c.get("score", 0.0) for c in candidates), default=0.0)
