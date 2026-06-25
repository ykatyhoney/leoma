"""
Validator dashboard routes (first-class in the decentralized design).

Every validator now equally determines outcomes (it samples and self-evaluates its own tasks),
so the dashboard surfaces each validator's identity, liveness, and rotation participation —
"did this validator sample on its turns?" is the key health signal. Stake is shown for
information only; it does NOT affect weighting (aggregation is one-validator-one-weight).
"""
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Optional, Tuple

import bittensor as bt
from fastapi import APIRouter, Depends, HTTPException

from leoma.bootstrap import NETUID, NETWORK
from leoma.delivery.http.contracts import (
    ValidatorCard,
    ValidatorDetailResponse,
    ValidatorMinerScore,
    ValidatorRegistration,
)
from leoma.delivery.http.dashboard_service import compute_validator_participation
from leoma.delivery.http.routes.rotation import current_scoring_window, ordered_validators
from leoma.delivery.http.validators import validate_validator_hotkey
from leoma.delivery.http.verifier import verify_admin_signature
from leoma.infra.db.stores import RankStore, SampleStore, ValidatorStore


router = APIRouter()
validators_dao = ValidatorStore()
validator_samples_dao = SampleStore()
rank_scores_dao = RankStore()


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts else None


# Liveness window for the public ``online`` flag (coarse — we don't expose the exact last-seen time).
ONLINE_WINDOW = timedelta(minutes=15)


def _is_online(ts: datetime | None) -> bool:
    if ts is None:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - ts < ONLINE_WINDOW


async def _window_participation():
    """Compute per-validator participation over the current (block-derived) scoring window."""
    window = await current_scoring_window() or []
    samples = await validator_samples_dao.get_samples_in_task_window(window) if window else []
    return compute_validator_participation(samples, await ordered_validators(), window)


def _card(validator, participation, permissioned: bool) -> ValidatorCard:
    p = participation.get(validator.hotkey)
    return ValidatorCard(
        uid=validator.uid,
        hotkey=validator.hotkey,
        stake=float(validator.stake),
        permissioned=permissioned,
        online=_is_online(validator.last_seen_at),
        last_sampled_task_id=p.last_task_id if p else None,
        last_sampled_at=_iso(p.last_evaluated_at) if p else None,
        tasks_sampled=p.tasks_sampled if p else 0,
        expected_turns=p.expected_turns if p else 0,
        participation_rate=p.participation_rate if p else 0.0,
        evaluations=p.evaluations if p else 0,
        avg_latency_ms=p.avg_latency_ms if p else None,
    )


@router.get("", response_model=List[ValidatorCard])
async def list_validators() -> List[ValidatorCard]:
    """All registered validators with liveness + rotation participation. Public.

    Every registered validator is owner-managed and permissioned (it samples + sets weights).
    """
    validators = await validators_dao.get_all_validators()
    participation = await _window_participation()
    cards = [_card(v, participation, True) for v in validators]
    cards.sort(key=lambda c: -c.participation_rate)
    return cards


async def _resolve_from_metagraph(hotkey: str) -> Optional[Tuple[int, float]]:
    """Look up ``(uid, stake)`` for a hotkey from the subnet metagraph; ``None`` if not registered."""
    subtensor = bt.AsyncSubtensor(network=NETWORK)
    try:
        meta = await subtensor.metagraph(NETUID)
    finally:
        await subtensor.close()
    hotkeys = list(getattr(meta, "hotkeys", []) or [])
    for uid, hk in enumerate(hotkeys):
        if hk == hotkey:
            stake_vec = getattr(meta, "S", None)
            if stake_vec is None:
                stake_vec = getattr(meta, "stake", None)
            stake = float(stake_vec[uid]) if stake_vec is not None and uid < len(stake_vec) else 0.0
            return uid, stake
    return None


@router.post("")
async def register_validator(
    body: ValidatorRegistration,
    admin_hotkey: Annotated[str, Depends(verify_admin_signature)],
) -> dict:
    """Add (or update) a validator in the owner-managed allowlist. Admin only.

    The validator is then permissioned: it can authenticate, sample on its rotation turns, and set
    weights. ``uid`` and ``stake`` are resolved from the metagraph by hotkey when omitted (the hotkey
    must be registered on the subnet). Re-registering the same uid updates its hotkey/stake.
    """
    hotkey = validate_validator_hotkey(body.hotkey)
    uid = body.uid
    stake = body.stake if body.stake is not None else 0.0
    if uid is None:
        resolved = await _resolve_from_metagraph(hotkey)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail="Hotkey not found on the subnet metagraph; pass uid explicitly to add it anyway.",
            )
        uid, meta_stake = resolved
        if body.stake is None:
            stake = meta_stake

    v = await validators_dao.save_validator(uid=uid, hotkey=hotkey, stake=max(0.0, stake))
    return {
        "success": True,
        "uid": v.uid,
        "hotkey": v.hotkey,
        "stake": float(v.stake),
        "added_by": admin_hotkey,
    }


@router.delete("/{validator_hotkey}")
async def remove_validator(
    validator_hotkey: str,
    admin_hotkey: Annotated[str, Depends(verify_admin_signature)],
) -> dict:
    """Remove a validator from the owner-managed allowlist by hotkey. Admin only.

    Once removed it can no longer authenticate, sample, or rotate (it drops out of the permissioned
    set immediately).
    """
    validator_hotkey = validate_validator_hotkey(validator_hotkey)
    removed = await validators_dao.delete_validator_by_hotkey(validator_hotkey)
    if not removed:
        raise HTTPException(status_code=404, detail="Validator not registered")
    return {
        "success": True,
        "message": f"Removed validator {validator_hotkey[:8]}... from allowlist",
        "removed_by": admin_hotkey,
    }


@router.get("/{validator_hotkey}", response_model=ValidatorDetailResponse)
async def get_validator_detail(validator_hotkey: str) -> ValidatorDetailResponse:
    """One validator's card + the per-miner pass rates it recorded. Public."""
    validator_hotkey = validate_validator_hotkey(validator_hotkey)
    validator = await validators_dao.get_validator_by_hotkey(validator_hotkey)
    participation = await _window_participation()

    if validator is not None:
        card = _card(validator, participation, True)
    else:
        # Not registered (owner hasn't added it / was removed) but may have participation history.
        p = participation.get(validator_hotkey)
        card = ValidatorCard(
            uid=-1,
            hotkey=validator_hotkey,
            stake=0.0,
            permissioned=False,
            tasks_sampled=p.tasks_sampled if p else 0,
            expected_turns=p.expected_turns if p else 0,
            participation_rate=p.participation_rate if p else 0.0,
            evaluations=p.evaluations if p else 0,
            last_sampled_task_id=p.last_task_id if p else None,
            last_sampled_at=_iso(p.last_evaluated_at) if p else None,
            avg_latency_ms=p.avg_latency_ms if p else None,
        )

    scores = await rank_scores_dao.get_scores_by_validator(validator_hotkey)
    miner_scores = [
        ValidatorMinerScore(
            miner_hotkey=s.miner_hotkey,
            pass_rate=s.pass_rate,
            total_samples=s.total_samples,
            total_passed=s.total_passed,
        )
        for s in scores
    ]
    return ValidatorDetailResponse(**card.model_dump(), miner_scores=miner_scores)
