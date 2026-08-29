"""
Retrieval Layer (Step 3, architecture doc Layer 3).

Runs hybrid dense+sparse search with Distribution-Based Score Fusion
(DBSF) against the Qdrant collection built in Step 2/3. Takes an already-
computed query embedding (produced by Step 4's Tier-1 guardrail at
runtime) and returns a typed RetrievalResult for the FSM.

Each returned RetrievedChunk.text is the PARENT passage text (not the
child match text) — the child vector is what matched, but the parent's
full ~150-word passage is what generation needs for context, per the
architecture doc's parent-child chunking design. Parent text is read
directly from the payload (denormalized at upsert time), so this is a
single round-trip query.
"""

import logging
import time

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.core.qdrant_client import get_qdrant_client
from app.models.schemas import EmbeddingResult, RetrievalResult, RetrievedChunk

logger = logging.getLogger("voice_rag.retrieval")
settings = get_settings()

_PREFETCH_LIMIT = 20  # candidates per sub-query before DBSF fusion narrows to top_k


def hybrid_search(
    embedding: EmbeddingResult,
    top_k: int = 5,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
) -> RetrievalResult:
    """
    Hybrid dense+sparse search with DBSF fusion.

    No source_lang filter is applied: the index holds English passages
    only (per "Zero-Translation Search"), and a passage's source_lang
    metadata just records which query language it was originally paired
    with in MS MARCO-XI — it doesn't restrict which query languages may
    retrieve it, so filtering on it would just reduce recall.
    """
    client = client or get_qdrant_client()
    name = collection_name or settings.qdrant_collection_name

    start = time.perf_counter()

    response = client.query_points(
        collection_name=name,
        prefetch=[
            models.Prefetch(
                query=embedding.dense_vector,
                using="dense",
                limit=_PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=embedding.sparse_indices,
                    values=embedding.sparse_values,
                ),
                using="sparse",
                limit=_PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        limit=top_k,
        with_payload=True,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    chunks: list[RetrievedChunk] = []
    for point in response.points:
        payload = point.payload or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=payload.get("chunk_id", str(point.id)),
                parent_id=payload.get("parent_id", ""),
                text=payload.get("parent_text", ""),
                score=point.score,
                source_lang=payload.get("source_lang", ""),
                is_selected=payload.get("is_selected", False),
            )
        )

    top_score = chunks[0].score if chunks else 0.0

    logger.info(
        "Hybrid search: %d results, top_score=%.4f, latency=%.2fms",
        len(chunks),
        top_score,
        latency_ms,
    )

    return RetrievalResult(
        chunks=chunks,
        top_score=top_score,
        retrieval_latency_ms=latency_ms,
    )