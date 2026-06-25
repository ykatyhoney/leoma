"""
Participant (miner) commitment, info, result and score types.
"""

from typing import Optional

from pydantic import BaseModel

from leoma.domain.comparison import EvaluationResult


class MinerCommitment(BaseModel):
    """Miner chain commitment payload."""

    hotkey: str
    model_name: str
    model_revision: str
    chute_id: str
    commit_block: int


class MinerInfo(BaseModel):
    """Validated miner with status."""

    uid: int
    hotkey: str
    model_name: str = ""
    model_revision: str = ""
    chute_id: str = ""
    chute_slug: str = ""
    block: int = 0
    is_valid: bool = False
    invalid_reason: Optional[str] = None
    model_hash: str = ""
    chute_status: str = ""


class MinerResult(BaseModel):
    """Single miner result for one sample."""

    hotkey: str
    slug: str
    video_filename: str
    evaluation: EvaluationResult


class MinerScore(BaseModel):
    """Aggregated miner score."""

    passed_count: int = 0
    total: int = 0
    pass_rate: float = 0.0
    slug: str = "unknown"
