"""
Scores routes for Leoma API (Dashboard).

Provides endpoints for aggregated scores and miner rank (dashboard / get-rank).
Rank is determined by dominance algorithm (block order + 5% threshold), updated every 1h.
Validators use GET /weights for weight setting, not scores.
"""

from datetime import datetime
from typing import Any, List

from fastapi import APIRouter

from leoma.delivery.http.contracts import (
    AggregatedScoreResponse,
    MinerScoresResponse,
    ValidatorScoreDetail,
    ValidatorSummaryResponse,
    AggregatedStats,
    ValidatorScoresResponse,
    MinerScoreEntry,
    ScoreStatsResponse,
    RankResponse,
    MinerRankEntry,
)
from leoma.delivery.http.validators import validate_miner_hotkey, validate_validator_hotkey
from leoma.infra.db.stores import (
    MinerRankStore,
    MinerTaskRankStore,
    ParticipantStore,
    RankStore,
    SampleStore,
)
from leoma.infra.scorer_constants import COMPLETENESS_ELIGIBILITY_THRESHOLD

router = APIRouter()
rank_scores_dao = RankStore()
validator_samples_dao = SampleStore()
miner_rank_dao = MinerRankStore()
miner_task_rank_dao = MinerTaskRankStore()
valid_miners_dao = ParticipantStore()


def _format_timestamp(timestamp: datetime | None) -> str | None:
    """Format optional timestamp as ISO string."""
    return timestamp.isoformat() if timestamp else None


def _validator_score_detail(score: Any) -> ValidatorScoreDetail:
    """Convert score entity to validator detail response entry."""
    return ValidatorScoreDetail(
        validator_hotkey=score.validator_hotkey,
        score=score.score,
        total_samples=score.total_samples,
        total_passed=score.total_passed,
        pass_rate=score.pass_rate,
        updated_at=_format_timestamp(score.updated_at),
    )


def _miner_score_entry(score: Any) -> MinerScoreEntry:
    """Convert score entity to miner score response entry."""
    return MinerScoreEntry(
        miner_hotkey=score.miner_hotkey,
        score=score.score,
        total_samples=score.total_samples,
        total_passed=score.total_passed,
        pass_rate=score.pass_rate,
        updated_at=_format_timestamp(score.updated_at),
    )


def _aggregate_miner_scores(scores: list[Any]) -> AggregatedStats:
    """Aggregate validator-reported scores for one miner (per-validator rows only).

    Do not use total_samples/total_passed from this for cross-validator "sampling count":
    those fields sum evaluation rows; use :meth:`SampleStore.get_miner_sampling_stats_by_hotkeys`
    for distinct tasks per miner.
    """
    if not scores:
        return AggregatedStats(
            total_samples=0,
            total_passed=0,
            avg_score=0.0,
            pass_rate=0.0,
            validator_count=0,
        )

    total_samples = sum(score.total_samples for score in scores)
    total_passed = sum(score.total_passed for score in scores)
    avg_score = sum(score.score for score in scores) / len(scores)
    pass_rate = total_passed / total_samples if total_samples > 0 else 0.0

    return AggregatedStats(
        total_samples=total_samples,
        total_passed=total_passed,
        avg_score=avg_score,
        pass_rate=pass_rate,
        validator_count=len(scores),
    )

@router.get("/validators", response_model=List[ValidatorSummaryResponse])
async def get_validator_summaries() -> List[ValidatorSummaryResponse]:
    """Get validator summaries for dashboard (total samples, passed_count, avg score, last updated). Public."""
    summaries = await rank_scores_dao.get_validator_summaries()
    return [
        ValidatorSummaryResponse(
            validator_hotkey=s["validator_hotkey"],
            total_samples=s["total_samples"],
            total_passed=s["total_passed"],
            avg_score=s["avg_score"],
            pass_rate=s["pass_rate"],
            last_updated=_format_timestamp(s["last_updated"]) if s.get("last_updated") else None,
        )
        for s in summaries
    ]


@router.get("", response_model=AggregatedScoreResponse)
async def get_aggregated_scores() -> AggregatedScoreResponse:
    """Get aggregated dashboard scores across all validators. Public.

    Scores are calculated server-side from submitted sample metadata by the score calculation
    task (which also updates miner rank); leaderboard sample counts reflect this aggregated data only.
    """
    # Equal weight: aggregated score per miner = AVG of each validator's pass-rate
    # (RankStore.get_aggregated_scores), i.e. the per-validator-average. No stake weighting.
    scores = await rank_scores_dao.get_aggregated_scores()
    all_scores = await rank_scores_dao.get_all_scores()

    validators = set(s.validator_hotkey for s in all_scores)
    valid_timestamps = [score.updated_at for score in all_scores if score.updated_at is not None]
    latest_update = max(valid_timestamps, default=datetime.utcnow())

    return AggregatedScoreResponse(
        scores=scores,
        total_validators=len(validators),
        updated_at=latest_update,
    )


@router.get("/miner/{miner_hotkey}", response_model=MinerScoresResponse)
async def get_miner_scores(
    miner_hotkey: str,
) -> MinerScoresResponse:
    """Get dashboard scores for a specific miner from all validators. Public."""
    miner_hotkey = validate_miner_hotkey(miner_hotkey)
    scores = await rank_scores_dao.get_scores_by_miner(miner_hotkey)
    # Per-validator-average: each validator's pass-rate is one row; avg_score is their mean.
    agg = _aggregate_miner_scores(scores)
    return MinerScoresResponse(
        miner_hotkey=miner_hotkey,
        by_validator=[_validator_score_detail(score) for score in scores],
        aggregated=agg,
    )


@router.get("/validator/{validator_hotkey}", response_model=ValidatorScoresResponse)
async def get_validator_scores(
    validator_hotkey: str,
) -> ValidatorScoresResponse:
    """Get dashboard scores reported by a specific validator. Public."""
    validator_hotkey = validate_validator_hotkey(validator_hotkey)
    scores = await rank_scores_dao.get_scores_by_validator(validator_hotkey)
    
    return ValidatorScoresResponse(
        validator_hotkey=validator_hotkey,
        scores=[_miner_score_entry(score) for score in scores],
        total_miners=len(scores),
    )


def _uid_for_rank_miner(miner_from_db: Any | None) -> int | None:
    """Return UID for a miner in rank list."""
    return miner_from_db.uid if miner_from_db is not None else None


@router.get("/rank", response_model=RankResponse)
async def get_rank() -> RankResponse:
    """Get miner rank list (dashboard / get-rank CLI).
    
    Rank is calculated by dominance: block order first; to dominate earlier miners,
    pass_rate must exceed theirs by 5%. Rank 1 = top-ranked miner. Updated every 1h with score calculation.
    eligible = True when completeness (evaluated tasks / window size) >= threshold
    for the consecutive scoring window ending at max task_id (default threshold 80%).
    """
    rows = await miner_rank_dao.get_all_ordered_by_rank()
    # Batch-load miners + task ranks once (avoid a per-row N+1).
    uid_by_hotkey = {m.miner_hotkey: m.uid for m in await valid_miners_dao.get_all_miners()}
    completeness_by_hotkey = {
        t.miner_hotkey: float(t.completeness or 0.0)
        for t in await miner_task_rank_dao.get_all_ranked()
    }
    entries: List[MinerRankEntry] = []
    for r in rows:
        comp = completeness_by_hotkey.get(r.miner_hotkey)
        eligible = comp is not None and comp >= COMPLETENESS_ELIGIBILITY_THRESHOLD - 1e-9
        entries.append(
            MinerRankEntry(
                miner_hotkey=r.miner_hotkey,
                uid=uid_by_hotkey.get(r.miner_hotkey),
                rank=r.rank,
                passed_count=r.passed_count,
                pass_rate=r.pass_rate,
                block=r.block,
                eligible=eligible,
            )
        )
    return RankResponse(ranks=entries)


@router.get("/stats", response_model=ScoreStatsResponse)
async def get_score_stats() -> ScoreStatsResponse:
    """Get dashboard score statistics. Public."""
    all_scores = await rank_scores_dao.get_all_scores()
    total_samples_count = await validator_samples_dao.get_total_sample_count()

    validators = set(s.validator_hotkey for s in all_scores)
    miners = set(s.miner_hotkey for s in all_scores)
    
    total_passed = sum(s.total_passed for s in all_scores)
    total_samples = sum(s.total_samples for s in all_scores)
    
    return ScoreStatsResponse(
        total_validators=len(validators),
        total_miners=len(miners),
        total_samples=total_samples_count,
        total_score_entries=len(all_scores),
        overall_pass_rate=total_passed / total_samples if total_samples > 0 else 0.0,
    )
