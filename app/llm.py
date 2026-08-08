"""Thin wrappers around Together's chat and embedding models.

Centralises the E5 asymmetric-prefix handling (see PROJECT_CONSTANTS.md "E5 prefixing") so
every caller gets it automatically instead of remembering to prefix by hand.
"""
from __future__ import annotations

from langchain_together import ChatTogether, TogetherEmbeddings

from app.config import E5_DOC_PREFIX, E5_QUERY_INSTRUCTION, Settings


def make_chat_model(settings: Settings, *, model_name: str | None = None, **kwargs) -> ChatTogether:
    return ChatTogether(
        model_name=model_name or settings.together_model,
        together_api_key=settings.together_api_key,
        **kwargs,
    )


class E5Embeddings:
    """Wraps TogetherEmbeddings and applies the correct E5-instruct prefix per side.

    Embedding a raw query or passage without its prefix works but measurably hurts
    retrieval quality on E5-family models — this class makes it impossible to forget.
    """

    def __init__(self, settings: Settings):
        self._client = TogetherEmbeddings(
            model=settings.together_embed_model,
            together_api_key=settings.together_api_key,
        )

    def embed_query(self, query: str) -> list[float]:
        prefixed = E5_QUERY_INSTRUCTION.format(query=query)
        return self._client.embed_query(prefixed)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [E5_DOC_PREFIX.format(text=t) for t in texts]
        return self._client.embed_documents(prefixed)
