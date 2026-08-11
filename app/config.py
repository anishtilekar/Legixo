"""Central config. Fails loudly on missing required keys rather than limping along."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Together.ai
    # Left optional (not `str` required) so --dry-run and offline tests don't need
    # real keys. Any code path that actually calls Together/Pinecone still fails
    # loudly on an empty key — the client libraries reject "" themselves.
    together_api_key: str = ""
    together_model: str = "openai/gpt-oss-20b"
    together_utility_model: str = "openai/gpt-oss-20b"
    together_embed_model: str = "intfloat/multilingual-e5-large-instruct"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "legixo-qa"
    pinecone_namespace: str = "legixo-demo"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Retrieval strategy (see eval/ablation.md for the measurements behind the defaults)
    retrieval_mode: str = "dense"  # "dense" | "hybrid"
    rerank_enabled: bool = False
    rerank_model: str = "bge-reranker-v2-m3"  # free tier: 500 req/month
    rerank_candidates: int = 15  # widen recall before the reranker narrows it
    pinecone_sparse_index_name: str = "legixo-qa-sparse"
    pinecone_sparse_model: str = "pinecone-sparse-english-v0"

    # App
    corpus_dir: str = "corpus"
    chunk_tokens: int = 350
    chunk_overlap_tokens: int = 60
    # 8, not 5 — measured. See eval/ablation.md: k=5 scored 32/33, k=8 scored 33/33
    # in two independent runs *and* was faster, because a narrow window makes the
    # answering chunk go missing and the question burns the full rewrite loop.
    top_k: int = 8
    max_attempts: int = 2
    relevance_floor: float = 0.75
    ingest_token: str = "dummy-ingest-token"

    # LangSmith (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "legixo-qa-takehome"


# Embedding dimension is fixed by the model choice (see PROJECT_CONSTANTS.md) — not env-configurable,
# because changing it silently would desync the Pinecone index.
EMBED_DIMENSION = 1024
EMBED_MAX_TOKENS = 514
CHUNK_HARD_CEILING_TOKENS = 380

# E5-instruct asymmetric prefixes. See PROJECT_CONSTANTS.md "E5 prefixing".
E5_QUERY_INSTRUCTION = (
    "Instruct: Given a legal/business question, retrieve the document passage "
    "that answers it.\nQuery: {query}"
)
E5_DOC_PREFIX = "passage: {text}"


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


class MissingKeysError(RuntimeError):
    """Raised instead of letting a provider SDK emit a bare 401 traceback."""


def require_live_keys(settings: Settings) -> None:
    """Fail with an actionable message before any network call.

    Without this, forgetting to replace the placeholders in .env surfaces as a
    14-line SDK traceback ending in `[401] Invalid API key`, which tells a new
    reviewer nothing about what to actually do.
    """
    sources = {
        "TOGETHER_API_KEY": "https://api.together.xyz  (Settings -> API Keys)",
        "PINECONE_API_KEY": "https://app.pinecone.io   (API Keys)",
    }
    problems, fixes = [], []
    for name, value in (
        ("TOGETHER_API_KEY", settings.together_api_key),
        ("PINECONE_API_KEY", settings.pinecone_api_key),
    ):
        if not value:
            problems.append(f"  - {name} is not set")
        elif value.startswith("dummy"):
            problems.append(f"  - {name} is still the placeholder from .env.example")
        else:
            continue
        fixes.append(f"  {name}  -> {sources[name]}")

    if problems:
        raise MissingKeysError(
            "Missing or placeholder API keys:\n"
            + "\n".join(problems)
            + "\n\nFix: copy .env.example to .env (if you haven't), then set:\n"
            + "\n".join(fixes)
            + "\n\nNo keys needed to explore chunking: python -m scripts.ingest --dry-run"
        )
