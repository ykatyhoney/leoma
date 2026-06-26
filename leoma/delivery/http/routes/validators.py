"""
Validator dashboard routes (first-class in the decentralized design).

Every validator equally determines outcomes (it samples and self-evaluates its own tasks), so the
dashboard surfaces each validator's identity, liveness, and rotation participation — "did this
validator sample on its turns?" is the key health signal. Membership is the hardcoded repo allowlist;
uid/stake come from the metagraph (cached) for display only and do NOT affect weighting.
"""
from datetime import datetime, timedelta, timezone
import asyncio
import os
import time
from typing import Dict, List, Tuple

import bittensor as bt
from fastapi import APIRouter, HTTPException

from leoma.bootstrap import NETUID, NETWORK, emit_log as log
from leoma.delivery.http.contracts import (
    ValidatorCard,
    ValidatorDetailResponse,
    ValidatorMinerScore,
)
from leoma.delivery.http.dashboard_service import compute_validator_participation
from leoma.delivery.http.routes.rotation import current_scoring_window, ordered_validators
from leoma.delivery.http.validators import validate_validator_hotkey
from leoma.infra.allowlist import VALIDATOR_ALLOWLIST
from leoma.infra.db.stores import RankStore, SampleStore


router = APIRouter()
validator_samples_dao = SampleStore()
rank_scores_dao = RankStore()

# Liveness window for the public ``online`` flag (coarse — we don't expose the exact last-seen time).
ONLINE_WINDOW = timedelta(minutes=15)

# Cached {hotkey: (uid, stake)} from the metagraph (display only); TTL-refreshed.
_META_TTL = float(os.environ.get("VALIDATOR_META_CACHE_TTL", "300"))
_meta_cache: dict = {"data": None, "ts": 0.0}
_meta_lock = asyncio.Lock()


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts else None


def _is_online(ts: datetime | None) -> bool:
    if ts is None:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - ts < ONLINE_WINDOW


async def _metagraph_info() -> Dict[str, Tuple[int, float]]:
    """Cached ``{hotkey: (uid, stake)}`` from the subnet metagraph (display only).

    Refreshed at most every ``_META_TTL`` seconds; on a read error the last good snapshot is reused
    (empty if none yet), so the validators dashboard never fails just because the chain is briefly
    unreachable.
    """
    if _meta_cache["data"] is not None and time.time() - _meta_cache["ts"] < _META_TTL:
        return _meta_cache["data"]
    async with _meta_lock:
        if _meta_cache["data"] is not None and time.time() - _meta_cache["ts"] < _META_TTL:
            return _meta_cache["data"]
        try:
            subtensor = bt.AsyncSubtensor(network=NETWORK)
            try:
                meta = await subtensor.metagraph(NETUID)
            finally:
                await subtensor.close()
            hotkeys = list(getattr(meta, "hotkeys", []) or [])
            stake_vec = getattr(meta, "S", None)
            if stake_vec is None:
                stake_vec = getattr(meta, "stake", None)
            info: Dict[str, Tuple[int, float]] = {}
            for uid, hk in enumerate(hotkeys):
                stake = float(stake_vec[uid]) if stake_vec is not None and uid < len(stake_vec) else 0.0
                info[hk] = (uid, stake)
        except Exception as e:
            log(f"Could not read validator metagraph info: {e}", "warn")
            return _meta_cache["data"] or {}
        _meta_cache["data"] = info
        _meta_cache["ts"] = time.time()
        return info


async def _window_participation():
    """Compute per-validator participation over the current scoring window."""
    window = await current_scoring_window() or []
    samples = await validator_samples_dao.get_samples_in_task_window(window) if window else []
    return compute_validator_participation(samples, await ordered_validators(), window)


def _card(hotkey: str, uid: int, stake: float, participation) -> ValidatorCard:
    """Build a card for an allowlisted hotkey; uid/stake from the metagraph, liveness from samples."""
    p = participation.get(hotkey)
    return ValidatorCard(
        uid=uid,
        hotkey=hotkey,
        stake=stake,
        permissioned=True,
        online=_is_online(p.last_evaluated_at) if p else False,
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
    """The hardcoded permissioned validator allowlist with liveness + rotation participation. Public.

    Membership is the repo allowlist; uid/stake come from the metagraph (cached) for display only.
    """
    info = await _metagraph_info()
    participation = await _window_participation()
    cards = []
    for hk in sorted(set(VALIDATOR_ALLOWLIST)):
        uid, stake = info.get(hk, (0, 0.0))
        cards.append(_card(hk, uid, stake, participation))
    cards.sort(key=lambda c: -c.participation_rate)
    return cards


@router.get("/{validator_hotkey}", response_model=ValidatorDetailResponse)
async def get_validator_detail(validator_hotkey: str) -> ValidatorDetailResponse:
    """One permissioned validator's card + the per-miner pass rates it recorded. Public."""
    validator_hotkey = validate_validator_hotkey(validator_hotkey)
    if validator_hotkey not in set(VALIDATOR_ALLOWLIST):
        raise HTTPException(status_code=404, detail="Not a permissioned validator")
    info = await _metagraph_info()
    uid, stake = info.get(validator_hotkey, (0, 0.0))
    participation = await _window_participation()
    card = _card(validator_hotkey, uid, stake, participation)

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
