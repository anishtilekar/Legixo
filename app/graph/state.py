"""Graph state.

`trace` and `queries` use additive reducers so every node can append without
having to read-modify-write the whole list.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AskState(TypedDict, total=False):
    # inputs
    question: str
    top_k: int
    max_attempts: int

    # retrieval loop
    queries: Annotated[list[str], operator.add]
    attempt: int
    candidates: list[dict[str, Any]]  # this round's raw hits
    context: list[dict[str, Any]]  # accumulated, deduped by chunk_id
    context_used: list[dict[str, Any]]  # the exact chunks shown as [S1..Sn]

    # grading
    grade: str  # "sufficient" | "insufficient"
    grade_reason: str

    # output
    answer: str
    citations: list[dict[str, Any]]
    status: str  # "answered" | "not_found"

    trace: Annotated[list[dict[str, Any]], operator.add]
