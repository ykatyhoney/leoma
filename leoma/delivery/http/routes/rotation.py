"""
Rotation schedule endpoints (decentralized sampling coordination).

The owner-api is the authoritative clock for *whose turn it is to sample*. Sampling
rotates across the permissioned-validator allowlist every ``interval`` blocks
(default ``SAMPLING_ROTATION_INTERVAL``); since miners can't serve concurrent
requests, only one validator samples per window. ``task_id == current_block // interval``.

- ``GET  /rotation``          (permissioned) current schedule + whether it's your turn
- ``POST /rotation/interval`` (admin)        change the rotation interval
"""
import asyncio
import os
import time
from typing import Annotated, List, Optional, Tuple

import bittensor as bt
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from leoma.bootstrap import NETWORK, SAMPLING_ROTATION_INTERVAL, emit_log as log
from leoma.delivery.http.verifier import (
    get_current_admin,
    verify_permissioned_validator,
)
from leoma.infra.db.stores import ProducedTaskStore, SamplingStateStore, ValidatorStore
from leoma.infra.scorer_constants import settled_window_end, window_task_ids_ending_at


router = APIRouter()
sampling_state_dao = SamplingStateStore()
produced_task_dao = ProducedTaskStore()
validator_store = ValidatorStore()

# Short-lived block cache (single-flight) so concurrent calls don't each hit the chain.
_BLOCK_CACHE_TTL = 6.0
_block_cache: dict = {"block": None, "ts": 0.0}
_block_lock = asyncio.Lock()

# Failover: if the scheduled sampler hasn't produced this window's task within a grace period of
# blocks, the turn deterministically advances to the next validator(s) in rotation order so a
# missed turn rarely leaves a window empty. 0 derives the grace from the interval (half a window).
FAILOVER_GRACE_BLOCKS = int(os.environ.get("SAMPLER_FAILOVER_GRACE_BLOCKS", "0"))


def failover_step(block_offset: int, grace_blocks: int, n_validators: int) -> int:
    """How many rotation positions to advance, given blocks elapsed in the window without a task.

    ``0`` while within the first ``grace_blocks`` (primary's turn); ``1`` after one grace period,
    ``2`` after two, capped at ``n_validators - 1`` so it never wraps past the whole ring. Pure +
    deterministic, so every validator computes the same effective sampler.
    """
    if grace_blocks <= 0 or n_validators <= 1 or block_offset < grace_blocks:
        return 0
    return min(block_offset // grace_blocks, n_validators - 1)


class IntervalUpdate(BaseModel):
    interval: int = Field(..., gt=0, description="Rotation interval in blocks (must be > 0).")


async def ordered_validators() -> List[str]:
    """Stable, deterministic order of the owner-managed validator allowlist (the permissioned set).

    Read from the validators table (owner-managed via CLI), sorted by hotkey — every validator hits
    the same owner-api, so they all compute the identical rotation order.
    """
    validators = await validator_store.get_all_validators()
    return sorted(v.hotkey for v in validators)


async def _current_block() -> int:
    cached = _block_cache["block"]
    if cached is not None and time.time() - _block_cache["ts"] < _BLOCK_CACHE_TTL:
        return cached
    async with _block_lock:
        # Re-check after acquiring: another caller may have just refreshed the cache.
        cached = _block_cache["block"]
        if cached is not None and time.time() - _block_cache["ts"] < _BLOCK_CACHE_TTL:
            return cached
        subtensor = bt.AsyncSubtensor(network=NETWORK)
        try:
            block = int(await subtensor.get_current_block())
        finally:
            await subtensor.close()
        _block_cache["block"] = block
        _block_cache["ts"] = time.time()
        return block


def compute_sampler(validators: List[str], rotation_index: int) -> Optional[str]:
    if not validators:
        return None
    return validators[rotation_index % len(validators)]


async def current_scoring_window_rows() -> Tuple[List[int], Optional[List[str]]]:
    """The settled scoring window as ``(rotation_ids, active_validators)`` for the dashboard scorer.

    Production-based: the last N tasks in the produced-task ledger at the current block, so skipped
    rotation turns don't dilute the window. ``active_validators`` is the distinct samplers in the
    window (denominator for the min-distinct-validators gate). Falls back to the legacy block-derived
    consecutive window when the ledger is empty (e.g. before backfill on first deploy); ``active`` is
    ``None`` on that fallback (the gate is effectively disabled until the ledger fills).
    """
    block = await _current_block()
    rows = await produced_task_dao.window(as_of_block=block)
    if rows:
        ids = [r.rotation_id for r in rows]
        active = sorted({r.sampler_hotkey for r in rows})
        return ids, active
    interval = await sampling_state_dao.get_rotation_interval(SAMPLING_ROTATION_INTERVAL)
    ids = window_task_ids_ending_at(settled_window_end(block, interval)) or []
    return ids, None


async def current_scoring_window() -> Optional[List[int]]:
    """Scoring-window rotation_ids only (settled), shared by the dashboard + scorer."""
    ids, _ = await current_scoring_window_rows()
    return ids or None


@router.get("")
async def get_rotation(
    hotkey: Annotated[str, Depends(verify_permissioned_validator)],
) -> dict:
    """Return the current rotation schedule and whether it's the caller's turn to sample.

    If the primary sampler hasn't produced this window's task within the grace period, the turn
    advances deterministically to the next validator(s) so a missed turn rarely leaves an empty
    window. Once the task is produced, no failover applies.
    """
    interval = await sampling_state_dao.get_rotation_interval(SAMPLING_ROTATION_INTERVAL)
    validators = await ordered_validators()
    block = await _current_block()
    rotation_index = block // interval
    primary = compute_sampler(validators, rotation_index)

    produced = await produced_task_dao.has_rotation(rotation_index)
    if produced:
        step = 0
    else:
        grace = FAILOVER_GRACE_BLOCKS or max(1, interval // 2)
        step = failover_step(block - rotation_index * interval, grace, len(validators))
    sampler = compute_sampler(validators, rotation_index + step)
    return {
        "interval": interval,
        "current_block": block,
        "rotation_index": rotation_index,
        "validators": validators,
        "sampler_hotkey": sampler,
        "primary_sampler": primary,
        "failover_step": step,
        "is_your_turn": sampler == hotkey,
    }


@router.post("/interval")
async def set_rotation_interval(
    body: IntervalUpdate,
    hotkey: Annotated[str, Depends(get_current_admin)],
) -> dict:
    """Set the rotation interval (blocks). Admin only."""
    await sampling_state_dao.set_rotation_interval(body.interval)
    log(f"Rotation interval set to {body.interval} blocks by {hotkey[:12]}...", "success")
    return {"interval": body.interval}
