"""
Tasks routes for Leoma API.

Provides endpoints for task id (latest sampled task for validators),
miner task list, and task-miner detail.
"""

import asyncio
import os
import time
from typing import Annotated, Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from leoma.delivery.http.validators import validate_miner_hotkey
from leoma.delivery.http.verifier import verify_permissioned_validator
from leoma.delivery.http.routes._task_utils import (
    build_miner_task_entries,
    build_task_detail_entries,
    parse_aspect_and_overall,
    representative_sample,
)
from leoma.delivery.http.routes.rotation import _current_block
from leoma.infra.db.stores import (
    ProducedTaskStore,
    SampleStore,
    SamplingStateStore,
)
from leoma.infra.scorer_constants import SCORER_SETTLE_MARGIN, SCORER_TASK_WINDOW
from leoma.infra.storage_backend import get_task_media_presigned_urls
from leoma.delivery.http.contracts import (
    MinerTaskEntry,
    TaskDetailResponse,
    TaskMinerDetailResponse,
    TaskMinerValidatorResult,
)


router = APIRouter()
sampling_state_dao = SamplingStateStore()
validator_samples_dao = SampleStore()
produced_task_dao = ProducedTaskStore()


class TaskAnnouncement(BaseModel):
    task_id: int = Field(..., ge=0, description="Block-derived rotation index of the sampled task.")


# ---------------------------------------------------------------------------
# Sampling-turn lease (failover): the validator whose turn it is (primary, or a failover backup
# per GET /rotation) claims the rotation_id before sampling so a returning primary and a backup
# can't both sample the same window. The lease is PERSISTED in the DB (sampling_state) so it
# survives an owner-api restart; the lock serializes the read-modify-write within the process, and
# a wall-clock TTL frees the slot if a sampler crashes.
# ---------------------------------------------------------------------------
_CLAIM_TTL_SECONDS = float(os.environ.get("SAMPLER_CLAIM_TTL_SECONDS", "600"))
_claims_lock = asyncio.Lock()


class TaskClaim(BaseModel):
    rotation_id: int = Field(..., ge=0, description="Block-derived rotation index to claim.")


def _apply_claim(
    claims: Dict[int, Tuple[str, float]], rotation_id: int, hotkey: str, now: float, ttl: float
) -> dict:
    """Grant/refresh the lease for ``rotation_id`` to ``hotkey`` (or report the current holder).

    Pure aside from mutating ``claims``: purges expired leases, then grants if the slot is free or
    already held by this caller (refreshing its expiry), else denies with the current holder.
    """
    for rid in [r for r, (_h, exp) in claims.items() if exp <= now]:
        del claims[rid]
    cur = claims.get(rotation_id)
    if cur is None or cur[0] == hotkey:
        claims[rotation_id] = (hotkey, now + ttl)
        return {"granted": True, "holder": hotkey}
    return {"granted": False, "holder": cur[0]}


@router.get("/latest")
async def get_latest_task_id(
    _hotkey: Annotated[str, Depends(verify_permissioned_validator)],
) -> dict:
    """Return the latest sampled task id and the validator that sampled it.

    Validator sampling-coordination — permissioned-signature only. The public dashboard surfaces the
    same value via ``GET /overview`` (computed server-side), so this endpoint is validators-only.
    """
    task_id = await sampling_state_dao.get_latest_task_id()
    if task_id is None:
        raise HTTPException(status_code=404, detail="No task sampled yet")
    sampler_hotkey = await sampling_state_dao.get_latest_task_sampler()
    return {"task_id": task_id, "sampler_hotkey": sampler_hotkey}


@router.post("/announce")
async def announce_task(
    body: TaskAnnouncement,
    hotkey: Annotated[str, Depends(verify_permissioned_validator)],
) -> dict:
    """Announce a freshly sampled task (decentralized sampler → dashboard/peers).

    The authenticated permissioned validator records that it produced ``task_id`` (its block-derived
    rotation index) so `GET /tasks/latest` reflects it and peers know which bucket to read. It is
    also appended to the gap-free produced-task ledger (idempotent per rotation_id), which assigns a
    monotonic ``task_seq`` used to build the scoring window.
    """
    applied = await sampling_state_dao.announce_task(body.task_id, hotkey)
    block = await _current_block()
    ledger = await produced_task_dao.append(
        rotation_id=body.task_id, sampler_hotkey=hotkey, block=block
    )
    return {
        "task_id": body.task_id,
        "sampler_hotkey": hotkey,
        "applied": applied,
        "task_seq": ledger["task_seq"],
    }


@router.get("/window")
async def get_scoring_window(
    hotkey: Annotated[str, Depends(verify_permissioned_validator)],
    as_of_block: int | None = None,
    n: int = SCORER_TASK_WINDOW,
    margin: int = SCORER_SETTLE_MARGIN,
) -> dict:
    """Return the settled scoring window: the last ``n`` *produced* tasks at ``as_of_block``.

    Production-based (not block-number-based) so skipped rotation turns don't dilute the window.
    Validators pass their shared epoch-boundary block as ``as_of_block`` so every validator computes
    the identical window over immutable ledger rows. ``active_validators`` is the distinct set of
    samplers in the window (the denominator for the min-distinct-validators eligibility gate).
    """
    block = as_of_block if as_of_block is not None else await _current_block()
    rows = await produced_task_dao.window(as_of_block=block, n=n, margin=margin)
    active = sorted({r.sampler_hotkey for r in rows})
    return {
        "as_of_block": block,
        "window": [
            {"task_seq": r.task_seq, "rotation_id": r.rotation_id, "sampler_hotkey": r.sampler_hotkey}
            for r in rows
        ],
        "active_validators": active,
    }


@router.post("/claim")
async def claim_sampling_turn(
    body: TaskClaim,
    hotkey: Annotated[str, Depends(verify_permissioned_validator)],
) -> dict:
    """Lease a sampling turn before producing it (failover coordination).

    The validator whose turn it is (per `GET /rotation`) claims ``rotation_id`` first; only the
    grantee samples. An already-produced rotation_id can't be claimed (the window is done). The
    lease auto-expires (``SAMPLER_CLAIM_TTL_SECONDS``) so a crashed sampler frees the slot.
    """
    if await produced_task_dao.has_rotation(body.rotation_id):
        return {"granted": False, "holder": None, "already_produced": True,
                "rotation_id": body.rotation_id}
    async with _claims_lock:
        claims = await sampling_state_dao.load_claim_map()
        result = _apply_claim(claims, body.rotation_id, hotkey, time.time(), _CLAIM_TTL_SECONDS)
        await sampling_state_dao.save_claim_map(claims)
    result["rotation_id"] = body.rotation_id
    return result


@router.get("", response_model=List[MinerTaskEntry])
async def get_miner_tasks(
    miner_hotkey: str,
) -> List[MinerTaskEntry]:
    """List a miner's tasks: pass/fail, the validator that sampled each, latency, updated.

    Query param: miner_hotkey (required). Each task is evaluated by one validator (the sampler),
    so each entry names that validator's hotkey.
    """
    miner_hotkey = validate_miner_hotkey(miner_hotkey)
    samples = await validator_samples_dao.get_samples_by_miner_and_task_ids(miner_hotkey)
    if not samples:
        return []
    return build_miner_task_entries(samples)


@router.get("/{task_id:int}", response_model=TaskDetailResponse)
async def get_task_detail(task_id: int) -> TaskDetailResponse:
    """Task detail with one aggregated entry per miner for the given task."""
    samples = await validator_samples_dao.get_samples_by_task_id(task_id)
    if not samples:
        raise HTTPException(status_code=404, detail="Task not found")

    entries = build_task_detail_entries(samples)
    sampler_hotkey = samples[0].validator_hotkey
    presigned = await get_task_media_presigned_urls(
        task_id, samples[0].miner_hotkey, sampler_hotkey=sampler_hotkey
    )
    prefix = str(task_id)

    return TaskDetailResponse(
        task_id=task_id,
        description=(samples[0].prompt if samples[0].prompt else None),
        s3_prefix=prefix,
        first_frame_path=f"{prefix}/first_frame.png",
        original_clip_path=f"{prefix}/original_clip.mp4",
        first_frame_url=presigned.get("first_frame_url") if presigned else None,
        original_clip_url=presigned.get("original_clip_url") if presigned else None,
        miner_count=len(entries),
        sampler_hotkey=sampler_hotkey,
        miners=entries,
    )


@router.get("/{task_id:int}/miner/{miner_hotkey}", response_model=TaskMinerDetailResponse)
async def get_task_miner_detail(
    task_id: int,
    miner_hotkey: str,
) -> TaskMinerDetailResponse:
    """Task detail for a miner: description, S3 paths, the validator's evaluation, final pass/fail."""
    miner_hotkey = validate_miner_hotkey(miner_hotkey)
    samples = await validator_samples_dao.get_samples_by_task_and_miner(task_id, miner_hotkey)
    if not samples:
        raise HTTPException(status_code=404, detail="No evaluations for this task/miner")
    rep = representative_sample(samples)
    final_passed = bool(rep.passed)
    latency_ms = next((s.latency_ms for s in samples if getattr(s, "latency_ms", None) is not None), None)
    validator_results = []
    for s in samples:
        aspect_scores, overall_score = parse_aspect_and_overall(getattr(s, "generated_artifacts", None))
        validator_results.append(
            TaskMinerValidatorResult(
                validator_hotkey=s.validator_hotkey,
                passed=s.passed,
                evaluated_at=getattr(s, "evaluated_at", None),
                confidence=getattr(s, "confidence", None),
                reasoning=getattr(s, "reasoning", None),
                aspect_scores=aspect_scores,
                overall_score=overall_score,
            )
        )
    prefix = str(task_id)
    safe_hotkey = miner_hotkey.replace("/", "_").replace("\\", "_")
    presigned = await get_task_media_presigned_urls(
        task_id, miner_hotkey, sampler_hotkey=samples[0].validator_hotkey
    )
    return TaskMinerDetailResponse(
        task_id=task_id,
        miner_hotkey=miner_hotkey,
        description=(samples[0].prompt if samples and samples[0].prompt else None),
        s3_prefix=prefix,
        first_frame_path=f"{prefix}/first_frame.png",
        original_clip_path=f"{prefix}/original_clip.mp4",
        generated_video_path=f"{prefix}/generated_videos/{safe_hotkey}.mp4",
        first_frame_url=presigned.get("first_frame_url") if presigned else None,
        original_clip_url=presigned.get("original_clip_url") if presigned else None,
        generated_video_url=presigned.get("generated_video_url") if presigned else None,
        validators=validator_results,
        final_passed=final_passed,
        latency_ms=latency_ms,
    )
