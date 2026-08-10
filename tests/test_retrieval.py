"""RRF fusion and floor semantics — pure functions, no network."""
from __future__ import annotations

from app.retrieval import RRF_K, max_dense_score, reciprocal_rank_fusion


def dense(chunk_id: str, score: float) -> dict:
    return {"id": chunk_id, "score": score, "dense_score": score,
            "metadata": {"chunk_id": chunk_id, "text": "t"}}


def sparse(chunk_id: str, score: float) -> dict:
    return {"id": chunk_id, "score": score, "sparse_score": score,
            "metadata": {"chunk_id": chunk_id, "text": "t"}}


def test_agreement_between_retrievers_outranks_either_alone():
    """A chunk both retrievers like should beat one that only tops a single list."""
    fused = reciprocal_rank_fusion(
        [[dense("a", 0.9), dense("b", 0.88)], [sparse("b", 12.0), sparse("c", 9.0)]]
    )
    assert fused[0]["metadata"]["chunk_id"] == "b", "b is ranked by both, so it wins"


def test_fusion_dedupes_and_merges_both_scores():
    fused = reciprocal_rank_fusion([[dense("a", 0.9)], [sparse("a", 7.0)]])
    assert len(fused) == 1
    assert fused[0]["dense_score"] == 0.9
    assert fused[0]["sparse_score"] == 7.0
    # two first-place finishes
    assert fused[0]["rrf_score"] == 2 * (1.0 / (RRF_K + 1))


def test_sparse_only_hit_is_retained():
    fused = reciprocal_rank_fusion([[dense("a", 0.9)], [sparse("z", 5.0)]])
    ids = {c["metadata"]["chunk_id"] for c in fused}
    assert ids == {"a", "z"}, "lexical-only matches are the point of hybrid"


def test_displayed_score_is_a_real_similarity_not_the_rrf_score():
    """RRF scores are ~0.016 and would trip any absolute relevance floor, so the
    visible score must stay a genuine similarity."""
    fused = reciprocal_rank_fusion([[dense("a", 0.87)], [sparse("a", 5.0)]])
    assert fused[0]["score"] == 0.87
    assert fused[0]["rrf_score"] < 0.05


def test_floor_reads_dense_cosine_even_when_sparse_scores_are_larger():
    """The floor was calibrated on dense cosine; a big dot-product must not be
    mistaken for a high-confidence dense match."""
    candidates = [sparse("z", 42.0), dense("a", 0.84)]
    assert max_dense_score(candidates) == 0.84


def test_floor_falls_back_when_no_dense_scores_present():
    assert max_dense_score([sparse("z", 3.0)]) == 3.0
    assert max_dense_score([]) == 0.0


def test_empty_lists_fuse_to_empty():
    assert reciprocal_rank_fusion([[], []]) == []
