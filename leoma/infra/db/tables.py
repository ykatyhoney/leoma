"""SQLAlchemy ORM models for the Leoma database schema."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    Index,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _utc_timestamp_column(*, nullable: bool = False, with_onupdate: bool = False):
    """Create standard UTC timestamp column definition."""
    kwargs = {"onupdate": func.now()} if with_onupdate else {}
    return mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=nullable,
        **kwargs,
    )


class ValidMiner(Base):
    """Centrally validated miners (synced from metagraph with HF and Chutes)."""
    __tablename__ = "valid_miners"

    uid: Mapped[int] = mapped_column(Integer, primary_key=True)
    miner_hotkey: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    block: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    model_revision: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    chute_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    chute_slug: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_valid: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    invalid_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_validated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = _utc_timestamp_column()
    updated_at: Mapped[datetime] = _utc_timestamp_column(with_onupdate=True)

    __table_args__ = (
        Index("idx_valid_miners_is_valid", "is_valid"),
        Index("idx_valid_miners_hotkey", "miner_hotkey"),
    )


class SamplingState(Base):
    """Key-value store for owner/sampler state (e.g. latest_task_id)."""
    __tablename__ = "sampling_state"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = _utc_timestamp_column(with_onupdate=True)


class ValidatorSample(Base):
    """Sample submitted by a validator (evaluation result)."""
    __tablename__ = "validator_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    validator_hotkey: Mapped[str] = mapped_column(String(64), nullable=False)
    task_id: Mapped[int] = mapped_column(Integer, nullable=False)
    miner_hotkey: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    s3_prefix: Mapped[str] = mapped_column(String(512), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    original_artifacts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    generated_artifacts: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    presentation_order: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    evaluated_at: Mapped[datetime] = _utc_timestamp_column()

    __table_args__ = (
        Index("idx_validator_samples_validator", "validator_hotkey"),
        Index("idx_validator_samples_miner", "miner_hotkey"),
        Index("idx_validator_samples_task_id", "task_id"),
        Index("idx_validator_samples_date", "evaluated_at"),
        Index("idx_validator_samples_unique", "validator_hotkey", "task_id", "miner_hotkey", unique=True),
    )


class RankScore(Base):
    """Per-validator scores (calculated from validator_samples)."""
    __tablename__ = "rank_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    miner_hotkey: Mapped[str] = mapped_column(String(64), nullable=False)
    validator_hotkey: Mapped[str] = mapped_column(String(64), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    total_samples: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_passed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    updated_at: Mapped[datetime] = _utc_timestamp_column(with_onupdate=True)

    __table_args__ = (
        Index("idx_rank_miner", "miner_hotkey"),
        Index("idx_rank_validator", "validator_hotkey"),
        Index("idx_rank_miner_validator", "miner_hotkey", "validator_hotkey", unique=True),
    )


class Blacklist(Base):
    """Blacklisted miner hotkeys."""
    __tablename__ = "blacklist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hotkey: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    added_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = _utc_timestamp_column()


class EvaluationSignature(Base):
    """Validator signature over evaluation result payload for a task."""
    __tablename__ = "evaluation_signatures"

    task_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    validator_hotkey: Mapped[str] = mapped_column(String(64), primary_key=True)
    signature: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = _utc_timestamp_column()


class MinerRank(Base):
    """Per-miner rank for dashboard and weights."""
    __tablename__ = "miner_ranks"

    miner_hotkey: Mapped[str] = mapped_column(String(64), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    passed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    block: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = _utc_timestamp_column(with_onupdate=True)

    __table_args__ = (Index("idx_miner_ranks_rank", "rank"),)


class ValidatorMinerReport(Base):
    """A validator's reported validation result for one miner (decentralized validation).

    Each permissioned validator validates miners itself and reports the result here; the
    owner-api tallies a majority consensus into ``valid_miners`` for the dashboard.
    """
    __tablename__ = "validator_miner_reports"

    validator_hotkey: Mapped[str] = mapped_column(String(64), primary_key=True)
    miner_hotkey: Mapped[str] = mapped_column(String(64), primary_key=True)
    uid: Mapped[int] = mapped_column(Integer, nullable=False)
    block: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    model_revision: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    chute_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    chute_slug: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_valid: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    invalid_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    reported_at: Mapped[datetime] = _utc_timestamp_column(with_onupdate=True)

    __table_args__ = (
        Index("idx_miner_reports_validator", "validator_hotkey"),
        Index("idx_miner_reports_miner", "miner_hotkey"),
    )


class MinerTaskRank(Base):
    """Per-miner task-pass ranking (stake-weighted; consecutive window + completeness threshold)."""
    __tablename__ = "miner_task_ranks"

    miner_hotkey: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_passed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tasks_evaluated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completeness: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = _utc_timestamp_column(with_onupdate=True)
