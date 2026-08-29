"""
Step 3 upsert script: reads Step 2's parents.parquet + embedded_chunks.parquet,
denormalizes each child's parent passage text into its payload (so a single
hybrid retrieval query returns everything the generation layer needs — no
second round-trip, preserving the ~15ms retrieval latency budget), and
upserts points into Qdrant in batches.

Point ID note: Qdrant requires point IDs to be an unsigned int or a UUID.
Our chunk_id is a truncated content-hash string, not UUID-shaped, so we
derive a deterministic UUID from it via uuid.uuid5() for the Qdrant point
ID while keeping the original chunk_id in the payload for cross-referencing.

Usage:
    python -m scripts.upsert_to_qdrant
    python -m scripts.upsert_to_qdrant --input-dir ./data/processed --batch-size 256
"""

import argparse
import logging
import uuid

import pandas as pd
from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.core.qdrant_client import get_qdrant_client

logging.basicConfig(level="INFO")
logger = logging.getLogger("voice_rag.upsert_qdrant")

settings = get_settings()

# Stable namespace for deriving Qdrant point UUIDs from our chunk_id hashes.
_QDRANT_ID_NAMESPACE = uuid.UUID("f1e8b6c4-6f6a-4c9e-9c9a-2e7a6b9d5c31")


def chunk_id_to_qdrant_id(chunk_id: str) -> str:
    """Deterministic: same chunk_id always maps to the same Qdrant point ID (idempotent re-upserts)."""
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, chunk_id))


def load_processed(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    parents = pd.read_parquet(f"{input_dir}/parents.parquet")
    chunks = pd.read_parquet(f"{input_dir}/embedded_chunks.parquet")
    return parents, chunks


def build_points(parents: pd.DataFrame, chunks: pd.DataFrame) -> list[models.PointStruct]:
    """Join each child chunk with its parent's full passage text, build typed Qdrant points."""
    parent_text_by_id = dict(zip(parents["parent_id"], parents["text"]))

    points: list[models.PointStruct] = []
    missing_parents = 0

    for row in chunks.itertuples(index=False):
        parent_text = parent_text_by_id.get(row.parent_id)
        if parent_text is None:
            missing_parents += 1
            continue

        points.append(
            models.PointStruct(
                id=chunk_id_to_qdrant_id(row.chunk_id),
                vector={
                    "dense": list(row.dense_vector),
                    "sparse": models.SparseVector(
                        indices=list(row.sparse_indices),
                        values=list(row.sparse_values),
                    ),
                },
                payload={
                    "chunk_id": row.chunk_id,
                    "parent_id": row.parent_id,
                    "child_text": row.text,
                    "parent_text": parent_text,  # denormalized — see module docstring
                    "query_id": row.query_id,
                    "source_lang": row.source_lang,
                    "is_selected": bool(row.is_selected),
                },
            )
        )

    if missing_parents:
        logger.warning("%d chunks skipped — no matching parent_id found in parents.parquet", missing_parents)

    return points


def upsert_points(
    client: QdrantClient,
    points: list[models.PointStruct],
    collection_name: str | None = None,
    batch_size: int = 256,
) -> int:
    name = collection_name or settings.qdrant_collection_name
    total = 0

    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=name, points=batch, wait=True)
        total += len(batch)
        logger.info("Upserted %d/%d points", total, len(points))

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsert embedded chunks into Qdrant.")
    parser.add_argument("--input-dir", type=str, default=None, help="Defaults to settings.processed_data_dir")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    input_dir = args.input_dir or settings.processed_data_dir
    parents, chunks = load_processed(input_dir)
    logger.info("Loaded %d parents, %d chunks from %s", len(parents), len(chunks), input_dir)

    points = build_points(parents, chunks)
    logger.info("Built %d Qdrant points", len(points))

    client = get_qdrant_client()
    total = upsert_points(client, points, batch_size=args.batch_size)
    logger.info("Done — %d points upserted into '%s'", total, settings.qdrant_collection_name)


if __name__ == "__main__":
    main()