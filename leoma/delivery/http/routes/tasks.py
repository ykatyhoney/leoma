"""Tasks routes for Leoma API (public dashboard: miner task list + task/miner detail)."""
from typing import List

from fastapi import APIRouter, HTTPException

from leoma.delivery.http.validators import validate_miner_hotkey
from leoma.delivery.http.routes._task_utils import (
    build_miner_task_entries,
    build_task_detail_entries,
    parse_aspect_and_overall,
    representative_sample,
)
from leoma.infra.db.stores import SampleStore
from leoma.infra.storage_backend import get_task_media_presigned_urls
from leoma.delivery.http.contracts import (
    MinerTaskEntry,
    TaskDetailResponse,
    TaskMinerDetailResponse,
    TaskMinerValidatorResult,
)


router = APIRouter()
validator_samples_dao = SampleStore()


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
