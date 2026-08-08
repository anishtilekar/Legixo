"""StateGraph wiring.

    START -> normalize_question -> retrieve -> grade_context
                                                  |
                    +-----------------------------+-----------------------------+
                    | sufficient                  | insufficient &              | insufficient &
                    v                             v attempt < max               v attempt >= max
             generate_answer               rewrite_query --(loop)--> retrieve   no_answer
                    |                                                              |
                    v                                                              |
             verify_citations --(0 real citations)--------------------------------+
                    | >=1 real citation                                            |
                    v                                                              v
                 finalize <-------------------------------------------------- finalize
                    |
                   END

Two independent limits (see PROJECT_CONSTANTS.md):
  1. MAX_ATTEMPTS  — semantic cap enforced in route_after_grade
  2. recursion_limit — structural backstop passed to .invoke()
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.graph.nodes import GraphDeps, GraphNodes, route_after_grade, route_after_verify
from app.graph.state import AskState
from app.llm import E5Embeddings, make_chat_model
from app.vectorstore import VectorStore

# Runaway protection: caps output tokens per node. A malformed-output retry storm
# then costs cents rather than the whole balance.
ANSWER_MAX_TOKENS = 1024
UTILITY_MAX_TOKENS = 512


def recursion_limit_for(max_attempts: int) -> int:
    """Structural backstop, derived from the semantic cap rather than hardcoded.

    Worst path (every grade insufficient, then abstain) visits:
        normalize(1) + finalize(1) + no_answer(1)
        + (max_attempts + 1) rounds of retrieve+grade
        + max_attempts rewrites
      = 3 + 2*(max_attempts + 1) + max_attempts  = 5 + 3*max_attempts

    A fixed limit of 12 silently broke MAX_ATTEMPTS=3 (needs 14) — it surfaced as
    an HTTP 500 instead of a clean refusal. +8 leaves headroom for future nodes.
    """
    return 3 * max_attempts + 8


def build_graph(deps: GraphDeps):
    nodes = GraphNodes(deps)
    builder = StateGraph(AskState)

    builder.add_node("normalize_question", nodes.normalize_question)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("grade_context", nodes.grade_context)
    builder.add_node("rewrite_query", nodes.rewrite_query)
    builder.add_node("generate_answer", nodes.generate_answer)
    builder.add_node("verify_citations", nodes.verify_citations)
    builder.add_node("no_answer", nodes.no_answer)
    builder.add_node("finalize", nodes.finalize)

    builder.add_edge(START, "normalize_question")
    builder.add_edge("normalize_question", "retrieve")
    builder.add_edge("retrieve", "grade_context")

    # the branch: good path / retry path / give-up path
    builder.add_conditional_edges(
        "grade_context",
        route_after_grade,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "no_answer": "no_answer",
        },
    )
    builder.add_edge("rewrite_query", "retrieve")  # the loop
    builder.add_edge("generate_answer", "verify_citations")

    builder.add_conditional_edges(
        "verify_citations",
        route_after_verify,
        {"finalize": "finalize", "no_answer": "no_answer"},
    )
    builder.add_edge("no_answer", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


def build_production_deps(settings: Settings) -> GraphDeps:
    """Wire the real Together + Pinecone clients into the graph."""
    embedder = E5Embeddings(settings)
    store = VectorStore(settings)
    store.ensure_index()

    def retriever(query: str, top_k: int):
        return store.query(embedder.embed_query(query), top_k)

    return GraphDeps(
        settings=settings,
        retriever=retriever,
        chat=make_chat_model(settings, max_tokens=ANSWER_MAX_TOKENS),
        utility_chat=make_chat_model(
            settings,
            model_name=settings.together_utility_model,
            max_tokens=UTILITY_MAX_TOKENS,
        ),
    )
