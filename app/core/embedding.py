"""
BGE-M3 Bulk Embedding (Step 2, architecture doc Layer 2).

Generates dense (1024-dim) + learned-sparse vector representations for
every ChildChunk produced by ingestion.py, using the official
FlagEmbedding BGEM3FlagModel in a single forward pass per batch.

IMPORTANT — this is the OFFLINE / BULK embedder, distinct from the
real-time query encoder used in Step 4's Tier-1 guardrail:
  - Here (Step 2, ingestion):  full-precision BAAI/bge-m3 via FlagEmbedding,
                                run once per corpus, latency doesn't matter.
  - There (Step 4, runtime):   int8-quantized ONNX bge-m3, ~40ms budget,
                                run on every live user query.
Both must produce vectors in the same space (same base model, same dim)
so query vectors and indexed vectors are comparable — but the code paths
are intentionally separate because their performance constraints differ.
"""

import logging

from app.config import get_settings
from app.models.ingestion_schemas import ChildChunk, EmbeddedChunk

logger = logging.getLogger("voice_rag.embedding")
settings = get_settings()


class BGEM3BulkEmbedder:
    """
    Lazy-loading wrapper around FlagEmbedding's BGEM3FlagModel.

    Model load is deferred to first use (not __init__) so importing this
    module elsewhere (e.g. for type hints) doesn't trigger a multi-GB
    model download/load.
    """

    def __init__(self, model_name: str | None = None, use_fp16: bool = True):
        self.model_name = model_name or settings.embedding_model_name
        self.use_fp16 = use_fp16
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            logger.info("Loading BGE-M3 model '%s' (fp16=%s)...", self.model_name, self.use_fp16)
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
            logger.info("BGE-M3 model loaded.")
        return self._model

    @staticmethod
    def _lexical_weights_to_sparse(lexical_weights: dict) -> tuple[list[int], list[float]]:
        """
        BGEM3FlagModel returns sparse weights as {token_id (str): weight (float)}.
        Qdrant's sparse vector format wants parallel (indices, values) lists.
        """
        if not lexical_weights:
            return [], []
        indices = [int(token_id) for token_id in lexical_weights.keys()]
        values = [float(v) for v in lexical_weights.values()]
        return indices, values

    def embed_texts(self, texts: list[str], batch_size: int | None = None) -> list[dict]:
        """
        Embed a list of raw strings, returning dense + sparse per text.
        Low-level helper — prefer embed_children() for the typed pipeline.
        """
        if not texts:
            return []

        batch_size = batch_size or settings.embedding_batch_size
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # not used in V1 — reserved for V2 late-interaction re-ranking
        )

        dense_vecs = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]

        results = []
        for i in range(len(texts)):
            indices, values = self._lexical_weights_to_sparse(lexical_weights[i])
            results.append(
                {
                    "dense_vector": dense_vecs[i].tolist(),
                    "sparse_indices": indices,
                    "sparse_values": values,
                }
            )
        return results

    def embed_children(
        self,
        children: list[ChildChunk],
        batch_size: int | None = None,
    ) -> list[EmbeddedChunk]:
        """
        Embed a batch of typed ChildChunks, returning typed EmbeddedChunks
        ready for Qdrant upsert (Step 3). Batches internally per
        settings.embedding_batch_size to bound peak memory on large corpora.
        """
        if not children:
            return []

        batch_size = batch_size or settings.embedding_batch_size
        embedded: list[EmbeddedChunk] = []

        for start in range(0, len(children), batch_size):
            batch = children[start : start + batch_size]
            texts = [c.text for c in batch]

            logger.info(
                "Embedding batch %d-%d of %d child chunks...",
                start,
                start + len(batch),
                len(children),
            )

            vec_results = self.embed_texts(texts, batch_size=batch_size)

            for chunk, vec in zip(batch, vec_results):
                embedded.append(
                    EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        parent_id=chunk.parent_id,
                        text=chunk.text,
                        query_id=chunk.query_id,
                        source_lang=chunk.source_lang,
                        is_selected=chunk.is_selected,
                        dense_vector=vec["dense_vector"],
                        sparse_indices=vec["sparse_indices"],
                        sparse_values=vec["sparse_values"],
                    )
                )

        return embedded


def get_bulk_embedder() -> BGEM3BulkEmbedder:
    """Convenience factory — one embedder instance per ingestion run is enough."""
    return BGEM3BulkEmbedder()