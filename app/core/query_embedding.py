"""
Runtime Query Embedding (Step 4, architecture doc Layer 2, "~40ms" step).

This is the REAL-TIME counterpart to Step 2's BGEM3BulkEmbedder: encodes
one live user query into dense + sparse vectors, using the int8-quantized
ONNX export of bge-m3 for speed (target ~40ms on CPU) instead of the
full-precision FlagEmbedding model used offline for bulk ingestion.

CAVEAT (flagged deliberately): a proper ONNX export of BGE-M3's sparse
(lexical-weight) head is nontrivial and most quantization toolchains only
export the dense head. This module uses ONNX for the dense vector (the
part that actually needs the speed) and derives an approximate sparse
vector from tokenizer term frequencies as a placeholder — not the same
sparse quality as Step 2's FlagEmbedding output. Revisit once you've
confirmed how your ONNX export was built; if it does include a sparse
head, swap _approximate_sparse() for a real session output read.
"""

import logging
import time
from collections import Counter

import numpy as np

from app.config import get_settings
from app.models.schemas import EmbeddingResult

logger = logging.getLogger("voice_rag.query_embedding")
settings = get_settings()


class ONNXQueryEmbedder:
    """Lazy-loading wrapper around the quantized ONNX bge-m3 session + tokenizer."""

    def __init__(self, onnx_path: str | None = None, tokenizer_name: str | None = None):
        self.onnx_path = onnx_path or settings.embedding_onnx_path
        self.tokenizer_name = tokenizer_name or settings.embedding_model_name
        self._session = None
        self._tokenizer = None

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort

            logger.info("Loading ONNX session from %s...", self.onnx_path)
            self._session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
            logger.info("ONNX session loaded.")
        return self._session

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            logger.info("Loading tokenizer '%s'...", self.tokenizer_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    @staticmethod
    def _approximate_sparse(token_ids: list[int]) -> tuple[list[int], list[float]]:
        """
        Placeholder sparse vector: normalized term-frequency over token ids.
        See module docstring caveat — replace with a real sparse ONNX head
        if/when your export supports one.
        """
        counts = Counter(tid for tid in token_ids if tid not in (0,))  # skip pad token id 0
        if not counts:
            return [], []
        total = sum(counts.values())
        indices = list(counts.keys())
        values = [c / total for c in counts.values()]
        return indices, values

    def embed_query(self, text: str) -> EmbeddingResult:
        """Encode one live query into a typed EmbeddingResult (dense + approximate sparse)."""
        start = time.perf_counter()

        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512,
            padding=True,
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        outputs = self.session.run(None, onnx_inputs)
        # Convention: first output is the pooled dense embedding, shape (1, embedding_dim).
        dense_vector = outputs[0][0].tolist()

        token_ids = inputs["input_ids"][0].tolist()
        sparse_indices, sparse_values = self._approximate_sparse(token_ids)

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info("Query embedded in %.2fms (dense_dim=%d)", latency_ms, len(dense_vector))

        return EmbeddingResult(
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            embedding_latency_ms=latency_ms,
        )


def get_query_embedder() -> ONNXQueryEmbedder:
    """Convenience factory — one embedder instance per app process (session load is expensive)."""
    return ONNXQueryEmbedder()