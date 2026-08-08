from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = None
    include_trace: bool = False


class Citation(BaseModel):
    marker: str
    chunk_id: str
    source_path: str
    heading_path: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    status: str  # "answered" | "not_found"
    citations: list[Citation]
    attempts: int
    trace: list[dict] | None = None


class IngestResponse(BaseModel):
    files_processed: int
    chunks_upserted: int
    chunks_pruned: int
    per_file: dict[str, int]
