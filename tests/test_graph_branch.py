"""Offline proof of the graph's three load-bearing behaviours.

No network, no keys: the retriever and both chat models are stubs, so these run
in CI and pin the branch/loop/citation semantics that the scoring rubric asks
about. Live behaviour is covered separately by eval/ against the real API.
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.graph.build import build_graph, recursion_limit_for
from app.graph.nodes import GradeVerdict, GraphDeps


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _StubStructured:
    """Returned by with_structured_output(); pops from the shared verdict list."""

    def __init__(self, verdicts: list, calls: list):
        self._verdicts = verdicts
        self._calls = calls

    def invoke(self, messages):
        self._calls.append(messages)
        sufficient = self._verdicts.pop(0) if self._verdicts else False
        return GradeVerdict(sufficient=sufficient, reason="stub verdict")


class StubUtilityChat:
    """Serves grading (via with_structured_output) and rewriting (via invoke)."""

    def __init__(self, verdicts: list[bool]):
        self.verdicts = list(verdicts)
        self.grade_calls: list = []
        self.rewrite_calls: list = []

    def with_structured_output(self, schema):
        return _StubStructured(self.verdicts, self.grade_calls)

    def invoke(self, messages):
        self.rewrite_calls.append(messages)
        return _Resp(f"rewritten query {len(self.rewrite_calls)}")


class StubAnswerChat:
    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Resp(self.answers.pop(0) if self.answers else "INSUFFICIENT_CONTEXT")


def make_chunk(chunk_id: str, text: str, score: float = 0.9) -> dict:
    return {
        "id": chunk_id,
        "score": score,
        "metadata": {
            "chunk_id": chunk_id,
            "source_path": f"corpus/{chunk_id.split('#')[0]}",
            "source_file": chunk_id.split("#")[0],
            "heading_path": "Doc > Section",
            "text": text,
        },
    }


def build(verdicts, answers, *, hits=None, max_attempts=2, reranker=None):
    calls: list[str] = []
    chunks = hits if hits is not None else [make_chunk("a.md#0", "Notice is 60 days.")]

    def retriever(query: str, top_k: int):
        calls.append(query)
        return chunks

    utility = StubUtilityChat(verdicts)
    answer_chat = StubAnswerChat(answers)
    deps = GraphDeps(
        settings=Settings(relevance_floor=0.5, top_k=5, max_attempts=max_attempts),
        retriever=retriever,
        chat=answer_chat,
        utility_chat=utility,
        reranker=reranker,
    )
    graph = build_graph(deps)
    return graph, calls, utility, answer_chat


def run(graph, question="What is the notice period?", max_attempts=2, **extra):
    return graph.invoke(
        {"question": question, "max_attempts": max_attempts, **extra},
        config={"recursion_limit": recursion_limit_for(max_attempts)},
    )


def test_recursion_limit_covers_the_worst_path():
    """The derived limit must exceed the longest possible walk, or a legitimate
    refusal surfaces as a crash. A hardcoded 12 broke at max_attempts=3."""
    for max_attempts in range(1, 6):
        # normalize + finalize + no_answer, plus retrieve/rerank/grade per round,
        # plus one rewrite per attempt
        worst_path_steps = 6 + 4 * max_attempts
        assert recursion_limit_for(max_attempts) > worst_path_steps


# --- 1. good path ----------------------------------------------------------


def test_good_path_answers_with_real_citation():
    graph, calls, utility, _ = build([True], ["Notice is 60 days [S1]."])
    out = run(graph)

    assert out["status"] == "answered"
    assert len(out["citations"]) == 1
    assert out["citations"][0]["source_path"] == "corpus/a.md"
    assert out["citations"][0]["chunk_id"] == "a.md#0"
    assert len(calls) == 1, "good path must not retry"
    assert utility.rewrite_calls == [], "good path must not rewrite"


# --- 2. bad path: loops exactly max_attempts, then refuses ------------------


@pytest.mark.parametrize("max_attempts", [1, 2, 3])
def test_bad_path_loops_to_cap_then_abstains(max_attempts):
    graph, calls, utility, answer_chat = build(
        [False] * (max_attempts + 1), [], max_attempts=max_attempts
    )
    out = run(graph, max_attempts=max_attempts)

    assert out["status"] == "not_found"
    assert out["citations"] == []
    # one initial retrieval + one per rewrite
    assert len(calls) == max_attempts + 1
    assert len(utility.rewrite_calls) == max_attempts
    assert answer_chat.calls == [], "must never generate when context is insufficient"


def test_loop_is_bounded_even_if_grader_never_succeeds():
    """The cap is what stops this from spinning forever."""
    graph, calls, _, _ = build([False] * 50, [], max_attempts=2)
    out = run(graph, max_attempts=2)
    assert out["status"] == "not_found"
    assert len(calls) == 3


# --- 3. fabricated citations cannot survive --------------------------------


def test_fabricated_marker_sentence_is_dropped():
    graph, _, _, _ = build(
        [True], ["Notice is 60 days [S1]. The arbitration seat is Mumbai [S9]."]
    )
    out = run(graph)

    assert out["status"] == "answered"
    assert "[S9]" not in out["answer"]
    assert "Mumbai" not in out["answer"], "hallucinated claim must go with its fake marker"
    assert [c["marker"] for c in out["citations"]] == ["[S1]"]


def test_answer_with_only_fabricated_markers_becomes_not_found():
    graph, _, _, _ = build([True], ["The arbitration seat is Mumbai [S9]."])
    out = run(graph)

    assert out["status"] == "not_found"
    assert out["citations"] == []


def test_answer_with_no_markers_at_all_becomes_not_found():
    """Retry also fails to add a marker -> refuse rather than assert ungrounded."""
    graph, _, _, answer_chat = build(
        [True], ["Notice is 60 days.", "Notice is 60 days."]
    )
    out = run(graph)

    assert out["status"] == "not_found"
    assert len(answer_chat.calls) == 2, "should have retried once for the missing marker"


def test_model_declaring_insufficient_context_becomes_not_found():
    graph, _, _, _ = build([True], ["INSUFFICIENT_CONTEXT"])
    out = run(graph)
    assert out["status"] == "not_found"
    assert out["citations"] == []


# --- trace ------------------------------------------------------------------


def test_answerer_sees_every_chunk_the_grader_approved():
    """The grader judges the full accumulated context; the answerer must receive
    exactly that. Slicing it to top_k let the grader approve a chunk that was then
    withheld, so the model replied INSUFFICIENT_CONTEXT and a good answer became a
    refusal."""
    chunks = [make_chunk(f"d{i}.md#0", f"fact {i}", score=0.9 - i * 0.01) for i in range(8)]
    # first grade insufficient -> rewrite -> second grade sufficient, so context
    # accumulates past top_k before generate_answer runs
    graph, _, _, answer_chat = build(
        [False, True], ["The answer is fact 7 [S8]."], hits=chunks, max_attempts=2
    )
    out = run(graph, top_k=5)

    assert out["status"] == "answered"
    rendered = str(answer_chat.calls[-1])
    assert "[S8]" in rendered, "8th chunk must reach the answerer, not be sliced off"
    assert len(out["context_used"]) == len(chunks)


def test_trace_records_every_node_visited():
    graph, _, _, _ = build([True], ["Notice is 60 days [S1]."])
    out = run(graph)
    nodes = [t["node"] for t in out["trace"]]
    assert nodes == [
        "normalize_question",
        "retrieve",
        "rerank",
        "grade_context",
        "generate_answer",
        "verify_citations",
        "finalize",
    ]


# --- rerank node -----------------------------------------------------------


def test_rerank_disabled_is_a_transparent_pass_through():
    graph, _, _, _ = build([True], ["Notice is 60 days [S1]."], reranker=None)
    out = run(graph)
    note = next(t["note"] for t in out["trace"] if t["node"] == "rerank")
    assert "disabled" in note
    assert out["status"] == "answered"


def test_rerank_reorders_context_before_grading():
    chunks = [make_chunk("wrong.md#0", "irrelevant", 0.9),
              make_chunk("right.md#0", "Notice is 60 days.", 0.7)]

    def reranker(query, candidates, top_n):
        # cross-encoder promotes the genuinely relevant chunk
        return sorted(candidates, key=lambda c: "Notice" in c["metadata"]["text"], reverse=True)

    graph, _, _, answer_chat = build([True], ["Notice is 60 days [S1]."], hits=chunks, reranker=reranker)
    out = run(graph)
    assert out["context_used"][0]["metadata"]["chunk_id"] == "right.md#0"
    assert out["status"] == "answered"


def test_reranker_failure_falls_back_to_retrieval_order():
    """A reranker outage must degrade to plain retrieval, not fail the request."""
    def boom(query, candidates, top_n):
        raise RuntimeError("rerank service down")

    graph, _, _, _ = build([True], ["Notice is 60 days [S1]."], reranker=boom)
    out = run(graph)
    assert out["status"] == "answered"
    note = next(t["note"] for t in out["trace"] if t["node"] == "rerank")
    assert "error" in note


# --- citation marker bracket variants --------------------------------------


def test_full_width_cjk_brackets_are_recognised_as_citations():
    """The model emits 【S1】 (U+3010/U+3011) instead of [S1] often enough to
    matter. An ASCII-only pattern discarded correctly cited answers as
    ungrounded and returned a refusal instead."""
    graph, _, _, _ = build([True], ["The disputed invoice is INV-2024-0618 \u3010S1\u3011."])
    out = run(graph)

    assert out["status"] == "answered"
    assert [c["marker"] for c in out["citations"]] == ["[S1]"]
    assert "[S1]" in out["answer"], "marker should be normalised to ASCII in the answer"


def test_marker_normalisation_covers_known_variants():
    from app.graph.nodes import normalise_markers

    for raw in ("\u3010S2\u3011", "\uff3bS2\uff3d", "[ S2 ]", "[s2]"):
        assert normalise_markers(f"claim {raw}") == "claim [S2]", raw


def test_fabricated_marker_still_dropped_when_written_in_full_width():
    graph, _, _, _ = build([True], ["Real \u3010S1\u3011. Fake \u3010S9\u3011."])
    out = run(graph)
    assert [c["marker"] for c in out["citations"]] == ["[S1]"]
    assert "S9" not in out["answer"]
