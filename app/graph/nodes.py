"""The eight graph nodes plus the two routing functions.

Dependencies (retriever, chat models) are injected via GraphDeps rather than
constructed inline, so tests can drive the whole graph with a stub retriever and
no network. See tests/test_graph_branch.py.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app import prompts
from app.config import Settings
from app.graph.state import AskState
from app.retrieval import max_dense_score

# (query_text, top_k) -> [{id, score, metadata}, ...]
Retriever = Callable[[str, int], list[dict[str, Any]]]
# (query_text, candidates, top_n) -> reordered candidates
Reranker = Callable[[str, list[dict[str, Any]], int], list[dict[str, Any]]]

_MARKER_RE = re.compile(r"\[S(\d+)\]")
# Split after sentence-final punctuation, keeping the delimiter with the sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# The grader sees FULL chunk text.
#
# This was previously truncated to 300 chars to cut grader input tokens. That was
# a premature optimisation and it caused false refusals at corpus scale: a chunk
# whose answer sits past the cut (e.g. the court-fee cap in "Item 4", after the
# Item 1 preamble) looks genuinely incomplete to the grader, which then correctly
# reports it cannot see the fact — and the question is refused despite the right
# chunk having been retrieved at rank 1.
#
# Chunks are hard-capped at 380 tokens, so full text is bounded and the extra
# cost is fractions of a cent per query. Correctness wins.
_GRADER_SNIPPET_CHARS: int | None = None


class GradeVerdict(BaseModel):
    sufficient: bool = Field(description="true only if the excerpts state the specific fact asked about")
    reason: str = Field(description="one short sentence explaining the verdict")


@dataclass
class GraphDeps:
    settings: Settings
    retriever: Retriever
    chat: BaseChatModel  # answering — quality matters most here
    utility_chat: BaseChatModel  # grading + rewriting
    reranker: Reranker | None = None  # None -> rerank node is a pass-through


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class GraphNodes:
    def __init__(self, deps: GraphDeps):
        self.deps = deps

    # ---- 1. normalize ---------------------------------------------------

    def normalize_question(self, state: AskState) -> dict[str, Any]:
        q = (state.get("question") or "").strip()
        if not q:
            raise ValueError("question must not be empty")
        settings = self.deps.settings
        return {
            "question": q,
            "queries": [q],
            "attempt": 0,
            "top_k": state.get("top_k") or settings.top_k,
            "max_attempts": state.get("max_attempts") or settings.max_attempts,
            "context": [],
            "trace": [{"node": "normalize_question", "note": f"question normalized ({len(q)} chars)"}],
        }

    # ---- 2. retrieve ----------------------------------------------------

    def retrieve(self, state: AskState) -> dict[str, Any]:
        started = _now_ms()
        query = state["queries"][-1]
        top_k = state.get("top_k") or self.deps.settings.top_k

        matches = self.deps.retriever(query, top_k)

        # merge into accumulated context, keeping the best score per chunk
        merged: dict[str, dict[str, Any]] = {
            c["metadata"]["chunk_id"]: c for c in state.get("context", [])
        }
        for m in matches:
            cid = m["metadata"]["chunk_id"]
            if cid not in merged or m["score"] > merged[cid]["score"]:
                merged[cid] = m
        context = sorted(merged.values(), key=lambda c: c["score"], reverse=True)

        return {
            "candidates": matches,
            "context": context,
            "trace": [
                {
                    "node": "retrieve",
                    "note": f"query={query!r} hits={len(matches)} context={len(context)}",
                    "top_score": round(matches[0]["score"], 4) if matches else None,
                    "ms": round(_now_ms() - started, 1),
                }
            ],
        }

    # ---- 2b. rerank (optional; pass-through when disabled) --------------

    def rerank(self, state: AskState) -> dict[str, Any]:
        """Cross-encoder rerank of the accumulated context.

        Retrieval is a bi-encoder: query and passage are embedded separately, so
        a chunk can rank highly on entity-name overlap alone ("Copperline") while
        missing the discriminating term ("retainer"). A cross-encoder reads both
        together and reorders accordingly.

        The node always exists in the graph so the shape stays stable and the
        trace shows explicitly whether reranking ran.
        """
        started = _now_ms()
        context = state.get("context", [])
        if self.deps.reranker is None or not context:
            return {
                "trace": [{"node": "rerank", "note": "disabled (pass-through)", "ms": 0.0}]
            }

        top_k = state.get("top_k") or self.deps.settings.top_k
        before = [c["metadata"]["chunk_id"] for c in context[:3]]
        try:
            reranked = self.deps.reranker(state["queries"][-1], context, top_k)
        except Exception as exc:
            return {
                "trace": [
                    {
                        "node": "rerank",
                        "note": f"reranker error, keeping retrieval order: {type(exc).__name__}",
                        "ms": round(_now_ms() - started, 1),
                    }
                ]
            }

        after = [c["metadata"]["chunk_id"] for c in reranked[:3]]
        return {
            "context": reranked,
            "trace": [
                {
                    "node": "rerank",
                    "note": f"{len(context)} -> {len(reranked)} | top3 {before} -> {after}",
                    "ms": round(_now_ms() - started, 1),
                }
            ],
        }

    # ---- 3. grade (the branch decision) ---------------------------------

    def grade_context(self, state: AskState) -> dict[str, Any]:
        started = _now_ms()
        context = state.get("context", [])
        floor = self.deps.settings.relevance_floor

        # Floor was calibrated against dense cosine, so it must keep reading dense
        # cosine in every retrieval mode — otherwise enabling hybrid or rerank
        # would silently redefine the threshold.
        scores = [c["score"] for c in context]
        top = max_dense_score(context) if context else 0.0
        rest = sorted(scores, reverse=True)[1:]
        margin = top - (sum(rest) / len(rest)) if rest else top

        # Layers 1+2: deterministic short-circuit. If nothing clears the floor
        # there is nothing worth spending a grader call on.
        if not context or top < floor:
            return {
                "grade": "insufficient",
                "grade_reason": f"no chunk cleared relevance floor (top={top:.4f} < {floor})",
                "trace": [
                    {
                        "node": "grade_context",
                        "note": "short-circuit: below relevance floor (no LLM call)",
                        "top_score": round(top, 4),
                        "margin": round(margin, 4),
                        "grade": "insufficient",
                        "ms": round(_now_ms() - started, 1),
                    }
                ],
            }

        # Layer 3: the LLM grader is the real decision-maker.
        blocks = []
        for i, c in enumerate(context, start=1):
            md = c["metadata"]
            text = md.get("text", "")
            snippet = text if _GRADER_SNIPPET_CHARS is None else text[:_GRADER_SNIPPET_CHARS]
            blocks.append(f"[S{i}] ({md.get('heading_path', '')}) {snippet}")
        rendered = "\n".join(blocks)

        grader = self.deps.utility_chat.with_structured_output(GradeVerdict)
        messages = [
            ("system", prompts.GRADE_SYSTEM),
            ("user", prompts.GRADE_USER.format(context=rendered, question=state["question"])),
        ]

        # Transient API errors and the occasional malformed structured output are
        # both worth one retry. Only after that do we fail safe.
        grade = reason = note = None
        last_error = ""
        for attempt_no in range(2):
            try:
                verdict: GradeVerdict = grader.invoke(messages)
                grade = "sufficient" if verdict.sufficient else "insufficient"
                reason = verdict.reason
                note = "LLM grader" if attempt_no == 0 else "LLM grader (succeeded on retry)"
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        if grade is None:
            # Fail safe toward refusing: a wrong refusal is recoverable, a
            # confidently wrong grounded-looking answer is not.
            grade = "insufficient"
            reason = f"grader failed twice, failing safe to insufficient — {last_error}"
            note = "grader exception (x2) -> insufficient"

        return {
            "grade": grade,
            "grade_reason": reason,
            "trace": [
                {
                    "node": "grade_context",
                    "note": note,
                    "top_score": round(top, 4),
                    "margin": round(margin, 4),
                    "grade": grade,
                    "reason": reason,
                    "ms": round(_now_ms() - started, 1),
                }
            ],
        }

    # ---- 4. rewrite (bad path, loops back to retrieve) ------------------

    def rewrite_query(self, state: AskState) -> dict[str, Any]:
        started = _now_ms()
        tried = "; ".join(state.get("queries", []))
        try:
            resp = self.deps.utility_chat.invoke(
                [
                    ("system", prompts.REWRITE_SYSTEM),
                    (
                        "user",
                        prompts.REWRITE_USER.format(
                            question=state["question"],
                            tried=tried,
                            reason=state.get("grade_reason", "unknown"),
                        ),
                    ),
                ]
            )
            new_query = (resp.content or "").strip().strip('"')
        except Exception:
            new_query = ""

        if not new_query:
            new_query = state["question"]  # degrade gracefully; loop cap still applies

        return {
            "attempt": state.get("attempt", 0) + 1,
            "queries": [new_query],
            "trace": [
                {
                    "node": "rewrite_query",
                    "note": f"attempt {state.get('attempt', 0) + 1} -> {new_query!r}",
                    "ms": round(_now_ms() - started, 1),
                }
            ],
        }

    # ---- 5. generate (good path) ----------------------------------------

    def generate_answer(self, state: AskState) -> dict[str, Any]:
        started = _now_ms()
        # Use the SAME context the grader just approved — not a top_k slice of it.
        #
        # Context accumulates across retrieval attempts, and grade_context judges
        # the whole accumulated list. Slicing to top_k here meant the grader could
        # answer "sufficient, excerpt S6 states it" and then S6 was never shown to
        # the answerer, which duly replied INSUFFICIENT_CONTEXT and turned a good
        # answer into a refusal. The list is bounded by (max_attempts+1)*top_k,
        # so this stays small.
        used = state.get("context", [])

        blocks = []
        for i, c in enumerate(used, start=1):
            md = c["metadata"]
            blocks.append(
                f"[S{i}] (source: {md.get('source_path')} > {md.get('heading_path')})\n{md.get('text', '')}"
            )
        rendered = "\n\n".join(blocks)

        messages = [
            ("system", prompts.ANSWER_SYSTEM),
            ("user", prompts.ANSWER_USER.format(context=rendered, question=state["question"])),
        ]
        resp = self.deps.chat.invoke(messages)
        answer = (resp.content or "").strip()
        note = f"generated over {len(used)} chunks"

        # The model occasionally produces a correct answer but forgets the marker.
        # Verification would then (rightly) discard it as ungrounded, turning a
        # good answer into a false refusal — so give it one corrective retry
        # rather than inventing a citation for it.
        needs_marker = answer and not answer.upper().startswith("INSUFFICIENT_CONTEXT")
        if needs_marker and not _MARKER_RE.search(answer):
            retry = self.deps.chat.invoke(
                messages
                + [
                    ("assistant", answer),
                    (
                        "user",
                        "That answer has no [S#] citation marker, so it is invalid. "
                        "Reply again with the same facts, appending the correct [S#] "
                        "marker to each claim. Use only markers shown above.",
                    ),
                ]
            )
            retry_text = (retry.content or "").strip()
            if _MARKER_RE.search(retry_text):
                answer = retry_text
                note += " (+1 retry to add missing markers)"
            else:
                note += " (retry still had no marker)"

        return {
            "answer": answer,
            "context_used": used,
            "trace": [
                {
                    "node": "generate_answer",
                    "note": note,
                    "ms": round(_now_ms() - started, 1),
                }
            ],
        }

    # ---- 6. verify citations --------------------------------------------

    def verify_citations(self, state: AskState) -> dict[str, Any]:
        """Map every [S#] in the answer back to a real retrieved chunk.

        Any marker that doesn't correspond to a chunk we actually supplied is
        stripped from the answer text — this is what makes a fabricated citation
        structurally impossible to return, rather than merely unlikely.
        """
        started = _now_ms()
        answer = state.get("answer", "")
        used = state.get("context_used", [])

        if answer.strip().upper().startswith("INSUFFICIENT_CONTEXT"):
            return {
                "citations": [],
                "trace": [
                    {
                        "node": "verify_citations",
                        "note": "model declared INSUFFICIENT_CONTEXT",
                        "ms": round(_now_ms() - started, 1),
                    }
                ],
            }

        found = [int(n) for n in _MARKER_RE.findall(answer)]
        invalid = sorted({n for n in found if not (1 <= n <= len(used))})

        # Drop whole sentences that carried a fabricated marker, not just the
        # marker itself: stripping only the marker would leave an unsupported
        # claim sitting in the answer with no citation at all, which is worse
        # than a visibly fake one.
        if invalid:
            kept = [
                s for s in _SENTENCE_RE.split(answer)
                if not any(f"[S{n}]" in s for n in invalid)
            ]
            cleaned = " ".join(part.strip() for part in kept if part.strip())
        else:
            cleaned = answer
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        # recompute against what actually survived
        surviving = [int(n) for n in _MARKER_RE.findall(cleaned)]
        valid_idx = sorted({n for n in surviving if 1 <= n <= len(used)})

        citations = []
        for n in valid_idx:
            md = used[n - 1]["metadata"]
            text = md.get("text", "")
            citations.append(
                {
                    "marker": f"[S{n}]",
                    "chunk_id": md.get("chunk_id", ""),
                    "source_path": md.get("source_path", ""),
                    "heading_path": md.get("heading_path", ""),
                    "score": round(float(used[n - 1]["score"]), 4),
                    "snippet": text[:300] + ("…" if len(text) > 300 else ""),
                }
            )

        return {
            "answer": cleaned,
            "citations": citations,
            "trace": [
                {
                    "node": "verify_citations",
                    "note": f"valid={valid_idx} stripped_fabricated={invalid}",
                    "ms": round(_now_ms() - started, 1),
                }
            ],
        }

    # ---- 7. no answer ---------------------------------------------------

    def no_answer(self, state: AskState) -> dict[str, Any]:
        searched = ", ".join(repr(q) for q in state.get("queries", []))
        return {
            "answer": prompts.NO_ANSWER_TEXT,
            "citations": [],
            "status": "not_found",
            "trace": [
                {
                    "node": "no_answer",
                    "note": f"abstained after {state.get('attempt', 0) + 1} retrieval(s); searched: {searched}",
                    "reason": state.get("grade_reason", ""),
                }
            ],
        }

    # ---- 8. finalize ----------------------------------------------------

    def finalize(self, state: AskState) -> dict[str, Any]:
        status = state.get("status") or "answered"
        return {
            "status": status,
            "trace": [{"node": "finalize", "note": f"status={status}"}],
        }


# ---- routing --------------------------------------------------------------


def route_after_grade(state: AskState) -> str:
    """The branch: good path vs bad path, with the loop cap enforced here."""
    if state.get("grade") == "sufficient":
        return "generate_answer"
    if state.get("attempt", 0) < state.get("max_attempts", 2):
        return "rewrite_query"
    return "no_answer"


def route_after_verify(state: AskState) -> str:
    """No real citation survived verification -> refuse rather than assert."""
    return "finalize" if state.get("citations") else "no_answer"
