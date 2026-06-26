"""
Pydantic models for API requests and responses.
"""

import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


# SS58 address regex pattern (Substrate addresses are 47-48 characters)
SS58_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{47,48}$")


def validate_ss58_hotkey(value: str) -> str:
    """Validate SS58 address format."""
    if not SS58_PATTERN.match(value):
        raise ValueError("Invalid SS58 address format")
    return value


class ORMResponseModel(BaseModel):
    """Base response model configured for ORM attribute loading."""

    model_config = ConfigDict(from_attributes=True)


class SampleSubmission(BaseModel):
    """Request model for submitting sample metadata (validator evaluation result)."""
    
    task_id: int = Field(..., description="Numeric task id for the evaluation task")
    miner_hotkey: str = Field(..., max_length=48, description="Miner's SS58 hotkey")
    prompt: str = Field(..., max_length=4096, description="Prompt used for generation")
    s3_bucket: str = Field(..., max_length=255, description="S3 bucket containing sample files")
    s3_prefix: str = Field(..., max_length=512, description="S3 prefix for sample files")
    passed: bool = Field(..., description="Whether generated video passed benchmark evaluation")
    confidence: Optional[int] = Field(None, ge=0, le=100, description="Confidence percentage")
    reasoning: Optional[str] = Field(None, max_length=4096, description="Evaluation reasoning")
    latency_ms: Optional[int] = Field(None, ge=0, description="Time from task creation to receiving result from miner (ms)")
    original_artifacts: Optional[str] = Field(None, max_length=8192, description="JSON or text for original artifacts")
    generated_artifacts: Optional[str] = Field(None, max_length=8192, description="JSON or text for generated artifacts")
    presentation_order: Optional[str] = Field(None, max_length=64, description="Presentation order")
    evaluation_signature: Optional[str] = Field(None, max_length=256, description="Validator signature over the evaluation payload for this task (for S3 verification)")
    
    @field_validator("miner_hotkey")
    @classmethod
    def validate_miner_hotkey(cls, v: str) -> str:
        return validate_ss58_hotkey(v)

class SampleBatchSubmission(BaseModel):
    """Request model for batch sample submission with optional validator signature over the evaluation payload."""
    signature: Optional[str] = Field(None, max_length=256, description="Validator signature over the evaluation result payload (for S3 verification)")
    samples: List[SampleSubmission] = Field(..., description="List of sample submissions (same task_id when signature is set)")


class BlacklistEntry(BaseModel):
    """Request model for adding a miner to blacklist."""
    
    hotkey: str = Field(..., max_length=48, description="Miner hotkey to blacklist")
    reason: Optional[str] = Field(None, max_length=255, description="Reason for blacklisting")
    
    @field_validator("hotkey")
    @classmethod
    def validate_hotkey(cls, v: str) -> str:
        return validate_ss58_hotkey(v)


class ValidatorInfo(ORMResponseModel):
    """Response model for validator information."""
    
    uid: int
    hotkey: str
    stake: float
    s3_bucket: Optional[str] = None
    last_seen_at: Optional[datetime] = None
    
class MinerResponse(ORMResponseModel):
    """Response model for miner information."""
    
    uid: int
    hotkey: str
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    model_hash: Optional[str] = None
    chute_id: Optional[str] = None
    chute_slug: Optional[str] = None
    is_valid: bool = False
    invalid_reason: Optional[str] = None
    block: Optional[int] = None
    last_validated_at: Optional[datetime] = None


class MinersListResponse(BaseModel):
    """Response model for list of miners."""
    
    miners: List[MinerResponse]
    total: int
    valid_count: int


class ScoreResponse(ORMResponseModel):
    """Response model for score information."""
    
    miner_hotkey: str
    validator_hotkey: str
    score: float
    total_samples: int = 0
    total_passed: int = 0
    pass_rate: float = 0.0
    updated_at: Optional[datetime] = None
    
class AggregatedScoreResponse(BaseModel):
    """Response model for aggregated scores."""
    
    scores: Dict[str, Dict[str, Any]]
    total_validators: int
    updated_at: Optional[datetime] = None


class MinerWeightEntry(BaseModel):
    """Per-miner entry for weights response (validator weight setting)."""
    miner_hotkey: str
    uid: int
    pass_rate: float
    weight: float  # 1.0 for top-ranked miner, 0.0 for others


class WeightsResponse(BaseModel):
    """Response for validators: top-ranked UID (winner_uid) and per-miner scores/weights."""
    winner_uid: int
    miners: List[MinerWeightEntry] = []


class MinerRankEntry(BaseModel):
    """One miner entry in the rank list (dashboard / get-rank)."""
    miner_hotkey: str
    uid: Optional[int] = None
    rank: int
    passed_count: int
    pass_rate: float
    block: Optional[int] = None
    eligible: bool = False  # True when completeness (evaluated tasks / window size) >= threshold (e.g. 80%)


class RankResponse(BaseModel):
    """Rank list calculated by dominance algorithm (block + 5% threshold)."""
    ranks: List[MinerRankEntry]


class SampleResponse(ORMResponseModel):
    """Response model for sample information."""
    
    id: int
    task_id: int
    validator_hotkey: str
    miner_hotkey: str
    prompt: Optional[str] = None
    passed: bool
    confidence: Optional[int] = None
    reasoning: Optional[str] = None
    evaluated_at: Optional[datetime] = None
    latency_ms: Optional[int] = None
    
class BlacklistResponse(ORMResponseModel):
    """Response model for blacklist entries."""
    
    hotkey: str
    reason: Optional[str] = None
    added_by: Optional[str] = None
    created_at: Optional[datetime] = None
    
class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    version: str
    database: bool
    metagraph_synced: bool
    last_sync: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str
    detail: Optional[str] = None


class ValidatorScoreDetail(BaseModel):
    """Score detail from a single validator."""
    
    validator_hotkey: str
    score: float
    total_samples: int
    total_passed: int
    pass_rate: float
    updated_at: Optional[str] = None


class AggregatedStats(BaseModel):
    """Aggregated statistics for a miner."""
    
    total_samples: int
    total_passed: int
    avg_score: float
    pass_rate: float
    validator_count: int


class MinerScoresResponse(BaseModel):
    """Response model for miner scores endpoint."""
    
    miner_hotkey: str
    by_validator: List[ValidatorScoreDetail]
    aggregated: AggregatedStats


class MinerScoreEntry(BaseModel):
    """Score entry for a miner from a validator."""
    
    miner_hotkey: str
    score: float
    total_samples: int
    total_passed: int
    pass_rate: float
    updated_at: Optional[str] = None


class ValidatorScoresResponse(BaseModel):
    """Response model for validator scores endpoint."""
    
    validator_hotkey: str
    scores: List[MinerScoreEntry]
    total_miners: int


class ScoreStatsResponse(BaseModel):
    """Response model for score statistics endpoint."""
    
    total_validators: int
    total_miners: int
    total_samples: int
    total_score_entries: int
    overall_pass_rate: float


class ValidatorSummaryResponse(BaseModel):
    """Response model for validator summary (dashboard list)."""
    
    validator_hotkey: str
    total_samples: int
    total_passed: int
    avg_score: float
    pass_rate: float
    last_updated: Optional[str] = None


class MinerTaskEntry(BaseModel):
    """One task entry for miner task list."""
    task_id: int
    passed: bool
    validator_hotkey: Optional[str] = None
    overall_score: Optional[int] = None  # 0-100, None if not recorded
    latency_ms: Optional[int] = None
    updated: Optional[datetime] = None


class TaskDetailMinerEntry(BaseModel):
    """One miner entry in a task-level detail response."""

    miner_hotkey: str
    passed: bool
    latency_ms: Optional[int] = None
    updated: Optional[datetime] = None


class TaskDetailResponse(BaseModel):
    """Task-level detail response with one entry per miner."""

    task_id: int
    description: Optional[str] = None
    s3_prefix: str
    first_frame_path: str
    original_clip_path: str
    first_frame_url: Optional[str] = None
    original_clip_url: Optional[str] = None
    miner_count: int
    sampler_hotkey: Optional[str] = None  # validator that sampled this task
    miners: List[TaskDetailMinerEntry]


class TaskMinerValidatorResult(BaseModel):
    """A validator's evaluation for a task/miner."""
    validator_hotkey: str
    passed: bool
    evaluated_at: Optional[datetime] = None
    confidence: Optional[int] = None
    reasoning: Optional[str] = None
    aspect_scores: Optional[Dict[str, int]] = None
    overall_score: Optional[int] = None


class TaskMinerDetailResponse(BaseModel):
    """Task detail for a specific miner (description, paths, optional presigned media URLs, validator results, final pass/fail)."""
    task_id: int
    miner_hotkey: str
    description: Optional[str] = None
    s3_prefix: str
    first_frame_path: str
    original_clip_path: str
    generated_video_path: str
    # Presigned GET URLs for dashboard (bucket stays private); None if not available
    first_frame_url: Optional[str] = None
    original_clip_url: Optional[str] = None
    generated_video_url: Optional[str] = None
    validators: List[TaskMinerValidatorResult]
    final_passed: bool
    latency_ms: Optional[int] = None


# ── Decentralized dashboard (active miners, validators, overview) ──

class ActiveMinerEntry(BaseModel):
    """A miner shown on the active leaderboard (valid + chute hot)."""
    uid: int
    hotkey: str
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    chute_slug: Optional[str] = None
    block: Optional[int] = None
    active: bool                       # valid + chute currently hot
    eligible: bool                     # completeness >= threshold over the scoring window
    rank: Optional[int] = None
    weight: float = 0.0                # 1.0 for rank-1 winner, else 0
    score: float = 0.0                 # per-validator-average pass rate
    tasks_passed: int = 0
    tasks_evaluated: int = 0
    validators_evaluating: int = 0
    last_evaluated_at: Optional[str] = None
    avg_latency_ms: Optional[int] = None
    # Each validator's own pass-rate for this miner (the values the equal-weight score averages),
    # for the per-row consensus/spread viz. Ascending.
    validator_scores: List[float] = []


class ValidatorCard(BaseModel):
    """Per-validator dashboard card (identity, liveness, rotation participation).

    Public-facing: the R2 bucket name is omitted (operational/infra) and liveness is coarsened to a
    boolean ``online`` rather than an exact ``last_seen_at`` timestamp.
    """
    uid: int
    hotkey: str
    stake: float                       # informational only — NOT used for weighting (equal weight)
    permissioned: bool                 # in the rotation/sampling allowlist
    online: bool = False               # seen within the liveness window (coarse, not an exact time)
    last_sampled_task_id: Optional[int] = None
    last_sampled_at: Optional[str] = None
    tasks_sampled: int = 0             # tasks sampled+evaluated in the window
    expected_turns: int = 0            # rotation turns assigned in the window
    participation_rate: float = 0.0    # tasks_sampled / expected_turns
    evaluations: int = 0
    avg_latency_ms: Optional[int] = None


class ValidatorMinerScore(BaseModel):
    miner_hotkey: str
    pass_rate: float
    total_samples: int
    total_passed: int


class ValidatorDetailResponse(ValidatorCard):
    """Validator card + the per-miner pass rates this validator recorded."""
    miner_scores: List[ValidatorMinerScore] = []


class OverviewResponse(BaseModel):
    """Network snapshot for the dashboard header."""
    active_miners: int
    valid_miners: int
    total_miners: int
    total_validators: int
    permissioned_validators: int
    rotation_interval: int
    current_sampler: Optional[str] = None
    latest_task_id: Optional[int] = None
    window_start: Optional[int] = None
    window_end: Optional[int] = None


# ── Decentralized miner validation (validators report; owner-api tallies consensus) ──

class MinerReportEntry(BaseModel):
    """One miner's validation result as judged by a single validator."""
    uid: int
    miner_hotkey: str = Field(..., max_length=64)
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    model_hash: Optional[str] = None
    chute_id: Optional[str] = None
    chute_slug: Optional[str] = None
    block: Optional[int] = None
    is_valid: bool = False
    invalid_reason: Optional[str] = None


class MinerReportSubmission(BaseModel):
    """A validator's full miner-validation report (replaces its previous report)."""
    miners: List[MinerReportEntry]
