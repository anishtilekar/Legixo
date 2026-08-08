"""Ingest CLI — the documented, no-REPL way to load the corpus into Pinecone.

Usage:
    python -m scripts.ingest                  # normal run: upsert + prune stale
    python -m scripts.ingest --reset           # wipe namespace first, then load
    python -m scripts.ingest --no-prune        # skip stale-id cleanup
    python -m scripts.ingest --dry-run         # chunk only, no embed/upsert, no keys needed
    python -m scripts.ingest --path other_dir  # ingest a different folder
"""
from __future__ import annotations

import argparse
import logging
import sys

from app.config import MissingKeysError, get_settings, require_live_keys
from app.ingest import ingest_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=None, help="corpus directory (default: CORPUS_DIR env)")
    parser.add_argument("--reset", action="store_true", help="delete the namespace before loading")
    parser.add_argument("--no-prune", action="store_true", help="skip pruning stale chunk ids")
    parser.add_argument("--dry-run", action="store_true", help="chunk only; no Pinecone/embedding calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = get_settings()
    if not args.dry_run:  # --dry-run deliberately needs no keys
        try:
            require_live_keys(settings)
        except MissingKeysError as exc:
            print(f"\nERROR: {exc}")
            return 2

    result = ingest_corpus(
        settings,
        corpus_dir=args.path,
        reset=args.reset,
        prune=not args.no_prune,
        dry_run=args.dry_run,
    )

    print(f"\nfiles processed:  {result.files_processed}")
    for source, n in result.per_file.items():
        print(f"  {source}: {n} chunks")
    if not args.dry_run:
        print(f"chunks upserted:  {result.chunks_upserted}")
        print(f"stale pruned:     {result.chunks_pruned}")
    else:
        print("(dry run — nothing embedded or upserted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
