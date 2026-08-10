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

import os

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.graph.nodes import GraphDeps, GraphNodes, route_after_grade, route_after_verify
from app.graph.state import AskState
from app.llm import E5Embeddings, make_chat_model
from app.retrieval import reciprocal_rank_fusion
from app.vectorstore import VectorStore

# Runaway protection: caps output tokens per node. A malformed-output retry storm
# then costs cents rather than the whole balance.
ANSWER_MAX_TOKENS = 1024
UTILITY_MAX_TOKENS = 512


def recursion_limit_for(max_attempts: int) -> int:
    """Structural backstop, derived from the semantic cap rather than hardcoded.

    Worst path (every grade insufficient, then abstain) visits:
        normalize(1) + finalize(1) + no_answer(1)
        + (max_attempts + 1) rounds of retrieve+rerank+grade
        + max_attempts rewrites
      = 3 + 3*(max_attempts + 1) + max_attempts  = 6 + 4*max_attempts

    A fixed limit of 12 silently broke MAX_ATTEMPTS=3 — it surfaced as an HTTP 500
    instead of a clean refusal. Adding the rerank node grew the per-round cost from
    2 nodes to 3, which is exactly why this is derived and not a constant.
    """
    return 4 * max_attempts + 8


def build_graph(deps: GraphDeps):
    nodes = GraphNodes(deps)
    builder = StateGraph(AskState)

    builder.add_node("normalize_question", nodes.normalize_question)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("rerank", nodes.rerank)
    builder.add_node("grade_context", nodes.grade_context)
    builder.add_node("rewrite_query", nodes.rewrite_query)
    builder.add_node("generate_answer", nodes.generate_answer)
    builder.add_node("verify_citations", nodes.verify_citations)
    builder.add_node("no_answer", nodes.no_answer)
    builder.add_node("finalize", nodes.finalize)

    builder.add_edge(START, "normalize_question")
    builder.add_edge("normalize_question", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "grade_context")

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


def _apply_langsmith_env(settings: Settings) -> None:
    """LangChain's tracer reads LANGCHAIN_* from os.environ directly, not from our
    Settings object — pydantic-settings' env_file loading only populates Settings,
    it never touches os.environ. Without this, LANGCHAIN_TRACING_V2=true in .env
    would silently do nothing when the app is started via uvicorn."""
    if not settings.langchain_tracing_v2:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


def build_production_deps(settings: Settings) -> GraphDeps:
    """Wire the real Together + Pinecone clients into the graph."""
    _apply_langsmith_env(settings)
    embedder = E5Embeddings(settings)
    store = VectorStore(settings)
    store.ensure_index()

    hybrid = settings.retrieval_mode == "hybrid"
    if hybrid:
        store.ensure_sparse_index()

    def retriever(query: str, top_k: int):
        # When reranking, retrieve wider than top_k so the cross-encoder has
        # something to actually reorder — a reranker over exactly top_k results
        # can only permute what dense already chose.
        fetch_k = settings.rerank_candidates if settings.rerank_enabled else top_k
        dense = store.query(embedder.embed_query(query), fetch_k)
        if not hybrid:
            return dense
        sparse_vec = store.embed_sparse([query], input_type="query")[0]
        sparse = store.query_sparse(sparse_vec, fetch_k)
        return reciprocal_rank_fusion([dense, sparse])

    reranker = None
    if settings.rerank_enabled:
        def reranker(query: str, candidates, top_n: int):  # noqa: F811
            return store.rerank(query, candidates, top_n)

    return GraphDeps(
        settings=settings,
        retriever=retriever,
        reranker=reranker,
        chat=make_chat_model(settings, max_tokens=ANSWER_MAX_TOKENS),
        utility_chat=make_chat_model(
            settings,
            model_name=settings.together_utility_model,
            max_tokens=UTILITY_MAX_TOKENS,
        ),
    )
