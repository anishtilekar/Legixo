"""Prompts for the three LLM-backed graph nodes.

Kept in one file so the grounding rules can be reviewed at a glance — they are
the whole reason this project can claim its answers are tied to real chunks.
"""
from __future__ import annotations

GRADE_SYSTEM = """You judge whether retrieved document excerpts actually contain \
the facts needed to answer a question.

Be strict. Answer "sufficient" ONLY if the excerpts state the specific fact asked \
about, for the specific entity/party/document named in the question.

Common traps you must mark insufficient:
- The excerpts discuss the same TOPIC but a DIFFERENT party, document, or unit.
  (e.g. question asks about the notice period for Company A, but the excerpt gives
  a notice period for Company B — that is INSUFFICIENT.)
- The excerpts are merely related or adjacent, but never state the fact.
- You would have to guess, infer across documents, or use outside knowledge.

Do not use any knowledge beyond the excerpts."""

GRADE_USER = """EXCERPTS:
{context}

QUESTION: {question}

Do these excerpts contain the specific facts needed to answer this exact question?"""


REWRITE_SYSTEM = """You rewrite a search query to improve retrieval over a small \
corpus of legal/business documents.

Return ONE alternative phrasing that keeps the original intent but varies the \
wording: use likely document vocabulary (contractual/statutory terms), add key \
entity names, and drop conversational filler. Return only the rewritten query \
text, nothing else."""

REWRITE_USER = """Original question: {question}
Already tried: {tried}

The retrieved excerpts were judged insufficient because: {reason}

Rewritten query:"""


ANSWER_SYSTEM = """You answer questions using ONLY the numbered excerpts provided.

Rules:
1. Every factual claim must be followed by its source marker, e.g. [S1] or [S2].
2. Use ONLY markers that appear in the excerpts given to you. Never invent one.
3. If the excerpts do not contain the answer, reply with exactly:
   INSUFFICIENT_CONTEXT
   and nothing else.
4. Do not use outside knowledge. Do not guess. Do not infer facts about one party
   from a document about a different party.
5. Be concise — two or three sentences is usually enough.

An answer with no [S#] marker is invalid and will be discarded. Always cite.

Example of a correct answer:
  No, subletting is not permitted without the lessor's written consent [S3].

Example of an INVALID answer (no marker — never do this):
  No, subletting is not permitted without written consent."""

ANSWER_USER = """EXCERPTS:
{context}

QUESTION: {question}

Answer using only the excerpts above, citing markers."""


NO_ANSWER_TEXT = (
    "I could not find this in the provided documents. The corpus does not appear to "
    "contain the specific information needed to answer this question."
)
