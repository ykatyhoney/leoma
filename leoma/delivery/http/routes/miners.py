"""Miners routes for Leoma API."""
from typing import Annotated, Any, List

from fastapi import APIRouter, Depends, HTTPException

from leoma.delivery.http.verifier import verify_signature, verify_permissioned_validator
from leoma.delivery.http.contracts import (
    ActiveMinerEntry,
    MinerReportSubmission,
    MinerResponse,
    MinersListResponse,
    MinerTaskEntry,
)
from leoma.delivery.http.routes._task_utils import build_miner_task_entries
from leoma.delivery.http.dashboard_service import compute_miner_activity
from leoma.infra.aggregate import compute_miner_aggregates
from leoma.delivery.http.validators import validate_miner_hotkey
from leoma.infra.db.stores import (
    MinerRankStore,
    MinerReportStore,
    ParticipantStore,
    SampleStore,
)
from leoma.infra.chute_status import probe_hot_chutes
from leoma.infra.scorer_constants import COMPLETENESS_ELIGIBILITY_THRESHOLD
from leoma.delivery.http.routes.rotation import current_scoring_window


router = APIRouter()
valid_miners_dao = ParticipantStore()
validator_samples_dao = SampleStore()
miner_rank_dao = MinerRankStore()
miner_report_dao = MinerReportStore()
MAX_REPORT_MINERS = 2048  # generous cap on miners per report


def _to_miner_response(miner: Any) -> MinerResponse:
    """Convert miner ORM/entity model to API response model."""
    return MinerResponse(
        uid=miner.uid,
        hotkey=miner.miner_hotkey,
        model_name=miner.model_name,
        model_revision=miner.model_revision,
        model_hash=miner.model_hash,
        chute_id=miner.chute_id,
        chute_slug=miner.chute_slug,
        is_valid=miner.is_valid,
        invalid_reason=miner.invalid_reason,
        block=miner.block,
        last_validated_at=miner.last_validated_at,
    )


def _to_miners_list_response(
    miners: list[Any],
    *,
    total: int,
    valid_count: int,
) -> MinersListResponse:
    """Convert miner entities into list response payload."""
    return MinersListResponse(
        miners=[_to_miner_response(miner) for miner in miners],
        total=total,
        valid_count=valid_count,
    )


@router.get("/uid/{uid}", response_model=MinerResponse)
async def get_miner_by_uid(uid: int) -> MinerResponse:
    """Get miner by UID (public dashboard endpoint)."""
    miner = await valid_miners_dao.get_miner_by_uid(uid)
    if not miner:
        raise HTTPException(status_code=404, detail="Miner not found")
    return _to_miner_response(miner)


@router.get("/list", response_model=MinersListResponse)
async def get_miners_list() -> MinersListResponse:
    """Get list of all miners (valid and invalid) for dashboard display. Public; no auth."""
    miners = await valid_miners_dao.get_all_miners()
    valid_count = await valid_miners_dao.get_valid_count()
    return _to_miners_list_response(
        miners,
        total=len(miners),
        valid_count=valid_count,
    )


@router.get("/valid", response_model=MinersListResponse)
async def get_valid_miners(
    _hotkey: Annotated[str, Depends(verify_signature)],
) -> MinersListResponse:
    """Get list of valid miners. Requires validator signature authentication."""
    miners = await valid_miners_dao.get_valid_miners()
    all_miners = await valid_miners_dao.get_all_miners()
    miner_responses = [_to_miner_response(m) for m in miners]
    return MinersListResponse(
        miners=miner_responses,
        total=len(all_miners),
        valid_count=len(miners),
    )


@router.get("/all", response_model=MinersListResponse)
async def get_all_miners(
    _hotkey: Annotated[str, Depends(verify_signature)],
) -> MinersListResponse:
    """Get all miners (valid and invalid). Requires validator signature authentication."""
    miners = await valid_miners_dao.get_all_miners()
    valid_count = await valid_miners_dao.get_valid_count()
    
    return _to_miners_list_response(
        miners,
        total=len(miners),
        valid_count=valid_count,
    )


@router.get("/active", response_model=List[ActiveMinerEntry])
async def get_active_miners() -> List[ActiveMinerEntry]:
    """Active miners for the dashboard: valid AND chute currently hot, enriched with score/rank.

    Public. 'active' = passed validation and the Chute is reachable now (cold/dead chutes drop
    off). Score is the per-validator-average pass rate over the scoring window.
    """
    valid = await valid_miners_dao.get_valid_miners()
    hot = await probe_hot_chutes([m.chute_id for m in valid if m.chute_id])
    active = [m for m in valid if m.chute_id and hot.get(m.chute_id)]
    if not active:
        return []

    window = await current_scoring_window() or []
    samples = await validator_samples_dao.get_samples_in_task_window(window) if window else []
    activity = compute_miner_activity(samples)
    # Windowed pooled pass-rate for EVERY active miner (ranked or not), so the score column
    # is one consistent metric — not a lifetime mean for unranked rows.
    aggregates = compute_miner_aggregates(
        {(s.validator_hotkey, s.task_id, s.miner_hotkey): bool(s.passed) for s in samples},
        window,
    )

    ranks = {r.miner_hotkey: r for r in await miner_rank_dao.get_all_ordered_by_rank()}
    winner = await miner_rank_dao.get_winner_hotkey()

    entries: List[ActiveMinerEntry] = []
    for m in active:
        hk = m.miner_hotkey
        a = activity.get(hk)
        r = ranks.get(hk)
        ag = aggregates.get(hk)
        completeness = ag.completeness if ag else 0.0
        eligible = completeness >= COMPLETENESS_ELIGIBILITY_THRESHOLD - 1e-9
        score = float(r.pass_rate) if r is not None else (float(ag.score) if ag else 0.0)
        entries.append(
            ActiveMinerEntry(
                uid=m.uid,
                hotkey=hk,
                model_name=m.model_name,
                model_revision=m.model_revision,
                chute_slug=m.chute_slug,
                block=m.block,
                active=True,
                eligible=eligible,
                rank=r.rank if r else None,
                weight=1.0 if (winner and hk == winner and r and r.rank == 1) else 0.0,
                score=score,
                tasks_passed=a.passed_tasks if a else 0,
                tasks_evaluated=a.tasks_evaluated if a else 0,
                validators_evaluating=a.validators_evaluating if a else 0,
                last_evaluated_at=a.last_evaluated_at.isoformat() if a and a.last_evaluated_at else None,
                avg_latency_ms=a.avg_latency_ms if a else None,
                validator_scores=sorted(ag.per_validator_rate.values()) if ag else [],
            )
        )
    # Ranked miners first (by rank), then the rest by score desc.
    entries.sort(key=lambda e: (e.rank if e.rank is not None else 10**9, -e.score))
    return entries


@router.post("/report")
async def report_miners(
    body: MinerReportSubmission,
    hotkey: Annotated[str, Depends(verify_permissioned_validator)],
) -> dict:
    """A permissioned validator reports its miner-validation results (replaces its prior report).

    The owner-api does NOT validate miners itself; it tallies validators' reports into a majority
    consensus (see MinerConsensusTask) which populates valid_miners for the dashboard.
    """
    if len(body.miners) > MAX_REPORT_MINERS:
        raise HTTPException(status_code=400, detail=f"Too many miners (max {MAX_REPORT_MINERS})")
    count = await miner_report_dao.replace_validator_report(
        hotkey, [m.model_dump() for m in body.miners]
    )
    return {"reported": count}


@router.get("/{miner_hotkey}/tasks", response_model=list[MinerTaskEntry])
async def get_miner_tasks(
    miner_hotkey: str,
) -> list[MinerTaskEntry]:
    """List a miner's recent tasks: pass/fail, the validator that sampled each, latency, updated.

    Drives the miner detail page. Each task is evaluated by one validator (the sampler for that
    window), so each entry names that validator's hotkey.
    """
    miner_hotkey = validate_miner_hotkey(miner_hotkey)
    samples = await validator_samples_dao.get_samples_by_miner_and_task_ids(miner_hotkey)
    if not samples:
        return []
    return build_miner_task_entries(samples)


@router.get("/info/{miner_hotkey}", response_model=MinerResponse)
async def get_miner_info(
    miner_hotkey: str,
) -> MinerResponse:
    """Get details for a specific miner (public dashboard endpoint)."""
    miner_hotkey = validate_miner_hotkey(miner_hotkey)
    miner = await valid_miners_dao.get_miner_by_hotkey(miner_hotkey)
    if not miner:
        raise HTTPException(status_code=404, detail="Miner not found")
    return _to_miner_response(miner)


@router.get("/{miner_hotkey}", response_model=MinerResponse)
async def get_miner(
    miner_hotkey: str,
    _hotkey: Annotated[str, Depends(verify_signature)],
) -> MinerResponse:
    """Get details for a specific miner. Requires validator signature authentication."""
    miner_hotkey = validate_miner_hotkey(miner_hotkey)
    miner = await valid_miners_dao.get_miner_by_hotkey(miner_hotkey)
    
    if not miner:
        raise HTTPException(status_code=404, detail="Miner not found")
    
    return _to_miner_response(miner)
