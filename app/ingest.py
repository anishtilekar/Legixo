"""Load corpus files -> chunk -> embed -> upsert into Pinecone.

Deterministic chunk ids (`{source_path}#{chunk_index}`) make this safe to
re-run: unchanged chunks overwrite themselves in place. See PROJECT_CONSTANTS.md.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.chunking import chunk_markdown
from app.config import Settings
from app.llm import E5Embeddings
from app.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    files_processed: int
    chunks_upserted: int
    chunks_pruned: int
    per_file: dict[str, int]


def _source_path_for(file_path: Path, corpus_root: Path) -> str:
    rel = file_path.relative_to(corpus_root.parent)
    return str(rel).replace("\\", "/")


def ingest_corpus(
    settings: Settings,
    *,
    corpus_dir: str | None = None,
    reset: bool = False,
    prune: bool = True,
    dry_run: bool = False,
) -> IngestResult:
    corpus_root = Path(corpus_dir or settings.corpus_dir).resolve()
    if not corpus_root.is_dir():
        raise FileNotFoundError(f"corpus dir not found: {corpus_root}")

    files = sorted(corpus_root.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md files found in {corpus_root}")

    store = None
    if not dry_run:
        store = VectorStore(settings)
        store.ensure_index()
        if reset:
            store.reset_namespace()

    embedder = E5Embeddings(settings) if not dry_run else None

    per_file: dict[str, int] = {}
    total_upserted = 0
    total_pruned = 0

    for fp in files:
        source_path = _source_path_for(fp, corpus_root)
        text = fp.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            text,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        per_file[source_path] = len(chunks)
        logger.info("chunked %s -> %d chunks", source_path, len(chunks))

        if dry_run:
            continue

        chunk_ids = {f"{source_path}#{c.chunk_index}" for c in chunks}
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_documents(texts)

        vectors = []
        for c, emb in zip(chunks, embeddings):
            chunk_id = f"{source_path}#{c.chunk_index}"
            vectors.append(
                {
                    "id": chunk_id,
                    "values": emb,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "source_path": source_path,
                        "source_file": fp.name,
                        "heading_path": c.heading_path,
                        "chunk_index": c.chunk_index,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "content_hash": c.content_hash,
                        "text": c.text,
                    },
                }
            )

        total_upserted += store.upsert(vectors)
        if prune and not reset:
            total_pruned += store.prune_stale(source_path, chunk_ids)

    return IngestResult(
        files_processed=len(files),
        chunks_upserted=total_upserted,
        chunks_pruned=total_pruned,
        per_file=per_file,
    )
