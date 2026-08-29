"""
Step 2 orchestration script: raw MS MARCO-XI -> chunked -> embedded ->
written to data/processed/ for Step 3's Qdrant upsert.

Usage:
    python -m scripts.ingest
    python -m scripts.ingest --languages hi,gu
    python -m scripts.ingest --languages hi,gu --output-dir ./data/processed

Writes two files:
    parents.parquet          — parent_id -> full passage text + metadata
                                (Qdrant only stores child vectors; the
                                generation layer looks up parent text by
                                parent_id at answer time)
    embedded_chunks.parquet  — one row per child chunk, dense + sparse
                                vectors included, ready for Qdrant upsert
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.core.embedding import BGEM3BulkEmbedder, get_bulk_embedder
from app.core.ingestion import ingest_languages
from app.models.ingestion_schemas import EmbeddedChunk, IngestionStats, ParentChunk

logging.basicConfig(level="INFO")
logger = logging.getLogger("voice_rag.ingest_script")

settings = get_settings()


def run_ingestion(
    languages: list[str] | None,
    output_dir: str | None = None,
    embedder: BGEM3BulkEmbedder | None = None,
) -> IngestionStats:
    """
    Runs the full Step 2 pipeline and writes outputs to disk.

    `embedder` is injectable so this function is unit-testable without
    loading the real multi-GB BGE-M3 model (see tests/test_ingest_script.py).
    """
    out_dir = Path(output_dir or settings.processed_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 2: Ingestion starting (languages=%s) ===", languages or "all configured")
    parents, children, stats = ingest_languages(languages)
    logger.info(
        "Ingestion complete: %d raw examples -> %d parents -> %d children",
        stats.raw_examples_seen,
        stats.parent_chunks_created,
        stats.child_chunks_created,
    )

    embedder = embedder or get_bulk_embedder()
    logger.info("=== Embedding %d child chunks ===", len(children))
    embedded_chunks = embedder.embed_children(children)
    stats.child_chunks_embedded = len(embedded_chunks)

    parents_path = out_dir / "parents.parquet"
    chunks_path = out_dir / "embedded_chunks.parquet"

    _write_parents(parents, parents_path)
    _write_embedded_chunks(embedded_chunks, chunks_path)

    stats.output_path = str(chunks_path)

    logger.info("=== Step 2 complete ===")
    logger.info("Parents written to:  %s (%d rows)", parents_path, len(parents))
    logger.info("Chunks written to:   %s (%d rows)", chunks_path, len(embedded_chunks))

    return stats


def _write_parents(parents: list[ParentChunk], path: Path) -> None:
    df = pd.DataFrame([p.model_dump() for p in parents])
    df.to_parquet(path, index=False)


def _write_embedded_chunks(chunks: list[EmbeddedChunk], path: Path) -> None:
    df = pd.DataFrame([c.model_dump() for c in chunks])
    df.to_parquet(path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice RAG Step 2: ingest, chunk, embed MS MARCO-XI.")
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes (e.g. 'hi,gu'). Defaults to all configured in raw_data_paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override settings.processed_data_dir for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    languages = args.languages.split(",") if args.languages else None
    stats = run_ingestion(languages=languages, output_dir=args.output_dir)
    print(stats.model_dump_json(indent=2))


if __name__ == "__main__":
    main()