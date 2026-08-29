"""
Typed data contracts for the Voice RAG FSM.

Every boundary between pipeline layers (STT -> Guardrail -> Embedding ->
Retrieval -> Guardrail -> Generation -> Guardrail -> Output) passes data
through one of these models. This is what makes the orchestrator a
deterministic, strictly-typed FSM rather than a free-form agent loop.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────
# FSM State Enum
# ──────────────────────────────────────────────────────
class PipelineStage(str, Enum):
    RECEIVED_AUDIO = "RECEIVED_AUDIO"
    TRANSCRIBED = "TRANSCRIBED"
    TIER1_CHECKED = "TIER1_CHECKED"
    EMBEDDED = "EMBEDDED"
    RETRIEVED = "RETRIEVED"
    TIER2_CHECKED = "TIER2_CHECKED"
    GENERATED = "GENERATED"
    TIER3_CHECKED = "TIER3_CHECKED"
    STREAMING = "STREAMING"
    COMPLETE = "COMPLETE"
    ABSTAINED = "ABSTAINED"
    ERROR = "ERROR"


class AbstainReason(str, Enum):
    UNSAFE_INPUT = "UNSAFE_INPUT"
    LOW_RELEVANCE_SCORE = "LOW_RELEVANCE_SCORE"
    FAITHFULNESS_CHECK_FAILED = "FAITHFULNESS_CHECK_FAILED"


# ──────────────────────────────────────────────────────
# Layer 1: STT Output
# ──────────────────────────────────────────────────────
class TranscriptionResult(BaseModel):
    query_id: str
    raw_text: str
    source_lang: str = Field(..., description="ISO code, e.g. 'hi', 'gu', 'en'")
    stt_latency_ms: float


# ──────────────────────────────────────────────────────
# Layer 2 (Tier-1 Guardrail) + Embedding Output
# ──────────────────────────────────────────────────────
class SafetyCheckResult(BaseModel):
    is_safe: bool
    reason: Optional[str] = None
    check_latency_ms: float


class EmbeddingResult(BaseModel):
    dense_vector: list[float] = Field(..., min_length=1)
    sparse_indices: list[int] = Field(default_factory=list)
    sparse_values: list[float] = Field(default_factory=list)
    embedding_latency_ms: float


# ──────────────────────────────────────────────────────
# Layer 3: Retrieval Output
# ──────────────────────────────────────────────────────
class RetrievedChunk(BaseModel):
    chunk_id: str
    parent_id: str
    text: str
    score: float
    source_lang: str
    is_selected: bool


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    top_score: float
    retrieval_latency_ms: float


# ──────────────────────────────────────────────────────
# Tier-2 Grounding Guardrail Output
# ──────────────────────────────────────────────────────
class GroundingGateResult(BaseModel):
    passed: bool
    threshold_used: float
    top_score: float


# ──────────────────────────────────────────────────────
# Layer 4: Generation Output
# ──────────────────────────────────────────────────────
class GenerationResult(BaseModel):
    answer_text: str
    ttft_ms: float
    tokens_per_sec: float


# ──────────────────────────────────────────────────────
# Tier-3 Faithfulness Judge Output
# ──────────────────────────────────────────────────────
class FaithfulnessCheckResult(BaseModel):
    is_faithful: bool
    judge_latency_ms: float


# ──────────────────────────────────────────────────────
# Top-Level Pipeline State (threaded through the FSM)
# ──────────────────────────────────────────────────────
class PipelineState(BaseModel):
    query_id: str
    stage: PipelineStage = PipelineStage.RECEIVED_AUDIO

    transcription: Optional[TranscriptionResult] = None
    safety_check: Optional[SafetyCheckResult] = None
    embedding: Optional[EmbeddingResult] = None
    retrieval: Optional[RetrievalResult] = None
    grounding_gate: Optional[GroundingGateResult] = None
    generation: Optional[GenerationResult] = None
    faithfulness_check: Optional[FaithfulnessCheckResult] = None

    abstain_reason: Optional[AbstainReason] = None
    error_detail: Optional[str] = None

    total_latency_ms: Optional[float] = None


# ──────────────────────────────────────────────────────
# Final Client-Facing Response (what goes over the WebSocket)
# ──────────────────────────────────────────────────────
class VoiceRAGResponse(BaseModel):
    query_id: str
    status: PipelineStage  # COMPLETE, ABSTAINED, or ERROR
    answer_text: Optional[str] = None
    abstain_reason: Optional[AbstainReason] = None
    source_lang: Optional[str] = None
    latency_breakdown_ms: dict[str, float] = Field(default_factory=dict)