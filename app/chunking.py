"""Heading-aware markdown chunker.

Splits on `#`/`##` headings first (so a chunk never straddles two unrelated
sections), then token-windows any section that's still too big for the
embedding model's 514-token ceiling. See PROJECT_CONSTANTS.md "Chunking".
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken

from app.config import CHUNK_HARD_CEILING_TOKENS

_ENCODING = tiktoken.get_encoding("cl100k_base")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_index: int
    heading_path: str
    text: str
    char_start: int
    char_end: int
    content_hash: str


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_into_sections(markdown: str) -> list[tuple[str, str, int]]:
    """Group lines under their nearest heading stack.

    Returns list of (heading_path, section_text, char_start).
    """
    lines = markdown.splitlines(keepends=True)
    sections: list[tuple[str, str, int]] = []
    heading_stack: list[str] = []
    buf: list[str] = []
    buf_start = 0
    pos = 0

    def flush():
        text = "".join(buf).strip()
        if text:
            sections.append((" > ".join(heading_stack) or "(untitled)", text, buf_start))

    for line in lines:
        m = _HEADING_RE.match(line.strip("\n"))
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            # truncate stack to this level, then append
            heading_stack = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack = heading_stack[: level - 1] + [title]
            buf = []
            buf_start = pos + len(line)
        else:
            if not buf:
                buf_start = pos
            buf.append(line)
        pos += len(line)
    flush()
    return sections


def _token_window_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split a text block into windows of <= max_tokens, with overlap, by token count."""
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    windows: list[str] = []
    start = 0
    step = max(max_tokens - overlap_tokens, 1)
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        windows.append(_ENCODING.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return windows


def chunk_markdown(
    markdown: str,
    *,
    chunk_tokens: int = 350,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Chunk one markdown document. Enforces the 380-token hard ceiling (see config)."""
    assert chunk_tokens <= CHUNK_HARD_CEILING_TOKENS, (
        f"chunk_tokens={chunk_tokens} exceeds the {CHUNK_HARD_CEILING_TOKENS}-token "
        "hard ceiling required by the 514-token embedding context window"
    )

    chunks: list[Chunk] = []
    idx = 0
    for heading_path, section_text, char_start in _split_into_sections(markdown):
        for window in _token_window_split(section_text, chunk_tokens, overlap_tokens):
            tok_count = count_tokens(window)
            assert tok_count <= CHUNK_HARD_CEILING_TOKENS, (
                f"chunk exceeds hard ceiling: {tok_count} tokens in section "
                f"'{heading_path}'"
            )
            char_end = char_start + len(window)
            content_hash = hashlib.sha256(window.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_index=idx,
                    heading_path=heading_path,
                    text=window.strip(),
                    char_start=char_start,
                    char_end=char_end,
                    content_hash=content_hash,
                )
            )
            idx += 1
    return chunks
