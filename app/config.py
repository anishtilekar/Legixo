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

    # App
    corpus_dir: str = "corpus"
    chunk_tokens: int = 350
    chunk_overlap_tokens: int = 60
    top_k: int = 5
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
