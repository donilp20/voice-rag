"""
Typed data contracts for the ingestion / chunking pipeline (Step 2).

These are distinct from app/models/schemas.py (the runtime FSM contracts).
Ingestion runs offline/batch, not per-query, so it gets its own typed
models — but the field names (parent_id, source_lang, is_selected)
intentionally match the metadata tags specified in the architecture doc
so they carry straight through into Qdrant payloads in Step 3.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ParentChunk(BaseModel):
    """
    An original MS MARCO-XI English passage (~150 words), preserved
    verbatim for LLM generation context. Deduplicated across queries
    that share the same passage.
    """

    parent_id: str = Field(..., description="Deterministic hash of passage text")
    text: str
    word_count: int
    query_id: str = Field(..., description="Source query_id this passage was linked to")
    source_lang: str = Field(..., description="Target Indic language code, e.g. 'hi', 'gu'")
    is_selected: bool = Field(
        ..., description="Whether MS MARCO annotators marked this passage as the answer source"
    )


class ChildChunk(BaseModel):
    """
    A sentence-level sub-chunk (1-2 sentences) of a ParentChunk, indexed
    for fine-grained similarity matching. Retrieval hits on children;
    generation context is pulled from the parent via parent_id.
    """

    chunk_id: str = Field(..., description="Deterministic hash of parent_id + child text")
    parent_id: str
    text: str
    sentence_span: int
    query_id: str
    source_lang: str
    is_selected: bool


class EmbeddedChunk(BaseModel):
    """A ChildChunk with its bge-m3 dense + sparse vector representation, ready for Qdrant upsert."""

    chunk_id: str
    parent_id: str
    text: str
    query_id: str
    source_lang: str
    is_selected: bool

    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class IngestionStats(BaseModel):
    """Summary emitted at the end of a Step 2 ingestion run."""

    languages_processed: list[str]
    raw_examples_seen: int
    parent_chunks_created: int
    child_chunks_created: int
    child_chunks_embedded: int
    output_path: Optional[str] = None