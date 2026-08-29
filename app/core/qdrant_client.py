"""
Qdrant Client & Collection Schema (Step 3, architecture doc Layer 3).

Defines the hybrid Sparse-Dense collection schema:
  - "dense"  named vector: 1024-dim (bge-m3), Cosine distance, SQ8 scalar
             quantization (4x memory reduction — fits 100k chunks in the
             1GB free tier per PRD cost constraint).
  - "sparse" named vector: bge-m3 learned sparse weights.

Transport defaults to gRPC with persistent connection pooling per the
architecture doc's "skip TLS handshake overhead" optimization.
"""

import logging

from qdrant_client import QdrantClient, models

from app.config import get_settings

logger = logging.getLogger("voice_rag.qdrant")
settings = get_settings()


def get_qdrant_client(location: str | None = None) -> QdrantClient:
    """
    Real usage: connects to Qdrant Cloud over gRPC using configured URL/key.
    `location=":memory:"` is supported for local testing without a server
    (used by our sanity checks, not in production).
    """
    if location:
        return QdrantClient(location=location)

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_use_grpc,
    )


def collection_exists(client: QdrantClient, collection_name: str | None = None) -> bool:
    name = collection_name or settings.qdrant_collection_name
    return client.collection_exists(collection_name=name)


def create_collection(client: QdrantClient, collection_name: str | None = None, recreate: bool = False) -> None:
    """
    Creates the hybrid dense+sparse collection if it doesn't already exist.
    Set recreate=True to drop and rebuild (destructive — used only for
    schema changes during development, never in the upsert/ingest path).
    """
    name = collection_name or settings.qdrant_collection_name

    if recreate and client.collection_exists(collection_name=name):
        logger.warning("Dropping existing collection '%s' (recreate=True)", name)
        client.delete_collection(collection_name=name)

    if client.collection_exists(collection_name=name):
        logger.info("Collection '%s' already exists — skipping creation.", name)
        return

    logger.info("Creating collection '%s' (dense dim=%d, SQ8 quantization)...", name, settings.embedding_dim)

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=settings.embedding_dim,
                distance=models.Distance.COSINE,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
            ),
        },
    )

    logger.info("Collection '%s' created.", name)


def get_collection_info(client: QdrantClient, collection_name: str | None = None):
    name = collection_name or settings.qdrant_collection_name
    return client.get_collection(collection_name=name)