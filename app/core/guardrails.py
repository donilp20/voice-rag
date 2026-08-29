"""
Guardrail & Abstention Engine (Step 4, architecture doc Layer 5).

Tier-1: fast regex + keyword safety check on raw transcribed text,
        BEFORE embedding/retrieval/LLM are invoked (<5ms budget).
Tier-2: grounding gate comparing retrieval's top DBSF score against
        settings.hybrid_abstain_threshold, BEFORE the LLM is invoked.

Both tiers return typed results and never raise on a "fail" — failing
a guardrail is a normal, expected FSM transition to ABSTAINED, not an
error. Threshold VALUES here are placeholders/defaults; real calibration
against a validation set (per architecture doc, APR >= 95%) is a
separate offline exercise, not part of this module.
"""

import logging
import re
import time

from app.config import get_settings
from app.models.schemas import GroundingGateResult, SafetyCheckResult

logger = logging.getLogger("voice_rag.guardrails")
settings = get_settings()

# Minimal illustrative blocklist — extend/replace with a real moderation
# list or model in production. Kept intentionally small and generic here;
# this is a mechanism, not a curated safety policy.
_UNSAFE_PATTERNS = [
    r"\bignore (all|previous|prior) instructions\b",
    r"\bsystem prompt\b",
    r"\bhow (to|do i) (make|build|synthesize) (a bomb|explosives?|nerve agent)\b",
    r"\bchild (sexual|porn)\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), flags=re.IGNORECASE)

# Extremely short or empty transcriptions are treated as unsafe/unusable
# input rather than sent through the full pipeline for nothing.
_MIN_QUERY_LENGTH = 2


def check_input_safety(text: str) -> SafetyCheckResult:
    """
    Tier-1 guardrail. Fast, synchronous, no model calls — regex +
    length heuristics only, per the <5ms latency budget.
    """
    start = time.perf_counter()

    if not settings.safety_regex_enabled:
        latency_ms = (time.perf_counter() - start) * 1000
        return SafetyCheckResult(is_safe=True, reason=None, check_latency_ms=latency_ms)

    stripped = text.strip()
    is_safe = True
    reason = None

    if len(stripped) < _MIN_QUERY_LENGTH:
        is_safe = False
        reason = "QUERY_TOO_SHORT"
    elif _UNSAFE_RE.search(stripped):
        is_safe = False
        reason = "UNSAFE_PATTERN_MATCH"

    latency_ms = (time.perf_counter() - start) * 1000

    if not is_safe:
        logger.info("Tier-1 guardrail blocked input (reason=%s, latency=%.3fms)", reason, latency_ms)

    return SafetyCheckResult(is_safe=is_safe, reason=reason, check_latency_ms=latency_ms)


def check_grounding_threshold(
    top_score: float,
    threshold: float | None = None,
) -> GroundingGateResult:
    """
    Tier-2 guardrail. Pure comparison against the calibrated threshold —
    no I/O, so latency is negligible (the actual cost was the retrieval
    call itself, already measured separately in RetrievalResult).
    """
    threshold_used = threshold if threshold is not None else settings.hybrid_abstain_threshold
    passed = top_score >= threshold_used

    if not passed:
        logger.info(
            "Tier-2 guardrail: ABSTAIN (top_score=%.4f < threshold=%.4f)",
            top_score,
            threshold_used,
        )

    return GroundingGateResult(
        passed=passed,
        threshold_used=threshold_used,
        top_score=top_score,
    )