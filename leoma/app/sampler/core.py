"""
Shared sampling core (used by each validator's own sampler).

Pure task-generation logic with NO task_id allocation and NO storage upload — the caller
passes in a ``task_id`` (the block-derived rotation index) and uploads the returned artifacts
to its own bucket, so the exact same generation path runs in every permissioned validator.
"""

import os
import time
import base64
import random
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp
from minio import Minio

from leoma.bootstrap import (
    SOURCE_BUCKET,
    MIN_VIDEO_SIZE,
    MAX_VIDEO_SIZE,
    CLIP_DURATION,
    CHUTES_API_KEY,
    REQUEST_TIMEOUT,
    REQUIRED_VIDEO_HEIGHT,
    REQUIRED_VIDEO_WIDTH,
    VIDEO_RESOLUTION_TOLERANCE,
    MAX_VIDEO_HISTORY,
)
from leoma.bootstrap import emit_log as log
from leoma.infra.video_utils import (
    OneShotClipSelection,
    choose_one_shot_clip_start,
    extract_clip,
    extract_first_frame,
    get_video_resolution,
)
from leoma.infra.judge import get_description_async, GEMINI_DESCRIPTION_MODEL
from leoma.infra.chute_resolver import get_chute_info, build_chute_endpoint

# Concurrency / one-shot selection tunables (env-overridable, same names as before).
MAX_CONCURRENT_MINERS = int(os.environ.get("MAX_CONCURRENT_MINERS", "40"))
ONE_SHOT_SCENE_THRESHOLD = float(os.environ.get("ONE_SHOT_SCENE_THRESHOLD", "0.18"))
ONE_SHOT_BOUNDARY_MARGIN = float(os.environ.get("ONE_SHOT_BOUNDARY_MARGIN", "0.15"))
SAFE_CLIP_START_OFFSET_SECONDS = float(os.environ.get("SAFE_CLIP_START_OFFSET_SECONDS", "2.0"))

# Recently used source videos (in-memory, per sampler process).
USED_VIDEOS: list[str] = []


@dataclass
class SampledTask:
    """Artifacts produced by :func:`sample_once` (caller uploads + cleans up)."""

    task_id: int
    clip_path: str
    frame_path: str
    metadata: Dict[str, Any]
    miner_paths: Dict[str, str]
    temp_paths: list[str] = field(default_factory=list)


def _is_resolution_acceptable(width: int, height: int) -> bool:
    """Accept videos whose width and height are both within tolerance of canonical 480p."""
    return (
        abs(width - REQUIRED_VIDEO_WIDTH) <= VIDEO_RESOLUTION_TOLERANCE
        and abs(height - REQUIRED_VIDEO_HEIGHT) <= VIDEO_RESOLUTION_TOLERANCE
    )


def build_generation_miners(valid_miners: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert a validated miner list to a generation payload map (hotkey -> info)."""
    return {
        m["hotkey"]: {
            "chute_id": m.get("chute_id"),
            "model_name": m.get("model_name"),
            "model_revision": m.get("model_revision"),
            "slug": m.get("chute_slug"),
            "block": m.get("block"),
            "model_hash": m.get("model_hash"),
        }
        for m in valid_miners
    }


def remove_file(path: str | None) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError:
        pass


def cleanup(result: Optional["SampledTask"], *extra_paths: str | None) -> None:
    """Remove all temp files produced during a sampling round."""
    paths: list[str | None] = list(extra_paths)
    if result is not None:
        paths.extend([result.clip_path, result.frame_path])
        paths.extend(result.miner_paths.values())
        paths.extend(result.temp_paths)
    for p in paths:
        remove_file(p)


def _build_paths(task_id: int) -> tuple[str, str, str]:
    sid = str(task_id)
    return (
        f"/tmp/source_video_{sid}.mp4",
        f"/tmp/original_clip_{sid}.mp4",
        f"/tmp/first_frame_{sid}.png",
    )


async def _list_source_videos(minio_client: Minio) -> list[str]:
    objects = await asyncio.to_thread(
        lambda: list(minio_client.list_objects(SOURCE_BUCKET, recursive=True))
    )
    return [
        obj.object_name
        for obj in objects
        if obj.object_name.endswith(".mp4") and MIN_VIDEO_SIZE < obj.size < MAX_VIDEO_SIZE
    ]


def _prioritize_video_keys(video_keys: list[str]) -> list[str]:
    global USED_VIDEOS
    if not video_keys:
        return []
    available = [key for key in video_keys if key not in USED_VIDEOS]
    if not available:
        recent = USED_VIDEOS[-5:] if len(USED_VIDEOS) >= 5 else []
        USED_VIDEOS.clear()
        USED_VIDEOS.extend(recent)
        available = [key for key in video_keys if key not in USED_VIDEOS]
    if not available:
        available = list(video_keys)
    random.shuffle(available)
    return available


def _register_used_video(video_key: str) -> None:
    global USED_VIDEOS
    USED_VIDEOS.append(video_key)
    if len(USED_VIDEOS) > MAX_VIDEO_HISTORY:
        USED_VIDEOS[:] = USED_VIDEOS[-MAX_VIDEO_HISTORY:]


async def select_one_shot_video(
    minio_client: Minio,
    local_video_path: str,
) -> tuple[str, OneShotClipSelection] | None:
    """Iterate source videos until one has a one-shot segment holding the full clip duration."""
    video_keys = await _list_source_videos(minio_client)
    for video_key in _prioritize_video_keys(video_keys):
        try:
            remove_file(local_video_path)
            await asyncio.to_thread(
                minio_client.fget_object, SOURCE_BUCKET, video_key, local_video_path
            )
            selection = await choose_one_shot_clip_start(
                local_video_path,
                CLIP_DURATION,
                scene_threshold=ONE_SHOT_SCENE_THRESHOLD,
                boundary_margin=ONE_SHOT_BOUNDARY_MARGIN,
            )
            if selection is None:
                log("No one-shot segment found; retrying next video", "warn")
                continue
            safe_clip_start = max(
                selection.clip_start_seconds,
                selection.segment_start_seconds + SAFE_CLIP_START_OFFSET_SECONDS,
            )
            max_clip_start = selection.segment_end_seconds - CLIP_DURATION
            if safe_clip_start > max_clip_start:
                log("Safe clip start offset exceeds max clip start; retrying next video", "warn")
                continue
            selection = OneShotClipSelection(
                clip_start_seconds=round(safe_clip_start, 3),
                segment_start_seconds=selection.segment_start_seconds,
                segment_end_seconds=selection.segment_end_seconds,
                video_duration_seconds=selection.video_duration_seconds,
                scene_cuts=selection.scene_cuts,
            )
            _register_used_video(video_key)
            return video_key, selection
        except Exception:
            continue
    return None


async def _generate_video_for_miner(
    session: aiohttp.ClientSession,
    endpoint: str,
    image_b64: str,
    prompt: str,
) -> tuple[bytes | None, str | None]:
    headers = {"Content-Type": "application/json"}
    if CHUTES_API_KEY:
        headers["Authorization"] = f"Bearer {CHUTES_API_KEY}"
    try:
        async with session.post(
            endpoint,
            headers=headers,
            json={
                "prompt": prompt,
                "image": image_b64,
                "fps": 16,
                "frames": 81,
                "resolution": "480p",
                "fast": True,
            },
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        ) as resp:
            if resp.status == 200:
                return await resp.read(), None
            return None, f"status {resp.status}"
    except asyncio.TimeoutError:
        return None, "timeout"
    except Exception as e:
        return None, str(e)


async def generate_videos_for_miners(
    session: aiohttp.ClientSession,
    miners: Dict[str, Dict[str, Any]],
    image_b64: str,
    prompt: str,
    task_created_at: float,
) -> tuple[Dict[str, tuple[bytes | None, str | None, str | None]], Dict[str, int]]:
    """Generate videos from all miners concurrently. Returns (miner_videos, latencies_ms)."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_MINERS)

    async def process(hotkey: str, info: Dict[str, Any]):
        async with semaphore:
            chute_id = info.get("chute_id")
            if not chute_id:
                return hotkey, (None, "no chute_id", None), None
            chute = await get_chute_info(session, chute_id)
            if not chute or not chute.get("hot") or not chute.get("slug"):
                return hotkey, (None, "chute not available", None), None
            endpoint = build_chute_endpoint(chute["slug"])
            video_bytes, err = await _generate_video_for_miner(session, endpoint, image_b64, prompt)
            latency_ms = int((time.time() - task_created_at) * 1000) if video_bytes is not None else None
            return hotkey, (video_bytes, err, endpoint), latency_ms

    tasks = [process(hk, info) for hk, info in miners.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: Dict[str, tuple[bytes | None, str | None, str | None]] = {}
    latencies: Dict[str, int] = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        hotkey, data, latency_ms = r
        out[hotkey] = data
        if latency_ms is not None:
            latencies[hotkey] = latency_ms
    return out, latencies


async def sample_once(
    task_id: int,
    miners: Dict[str, Dict[str, Any]],
    source_client: Minio,
    gemini_client: Any,
) -> Optional[SampledTask]:
    """Run one sampling round for a given task_id: select source, describe, call miners, validate.

    Returns a :class:`SampledTask` on success (caller uploads + cleans up), or ``None`` if no
    usable source/video/miner output was produced this round.
    """
    video_path, clip_path, frame_path = _build_paths(task_id)
    # Every temp file created this round; cleaned here on ANY failure, handed to the caller on success.
    created: list[str] = [video_path, clip_path, frame_path]
    miner_paths: Dict[str, str] = {}
    success = False
    try:
        selected = await select_one_shot_video(source_client, video_path)
        if not selected:
            log("No source video contains a one-shot segment for the clip duration", "warn")
            return None
        video_key, one_shot = selected
        start_offset = one_shot.clip_start_seconds

        await extract_clip(video_path, clip_path, start_offset, CLIP_DURATION)
        await extract_first_frame(clip_path, frame_path, start_offset=0)

        if not gemini_client:
            log("No Gemini client; cannot generate description", "warn")
            return None

        description = await get_description_async(gemini_client, clip_path)
        log(f"Description: {description[:80]}...", "info")

        with open(frame_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        task_created_at = time.time()
        async with aiohttp.ClientSession() as session:
            miner_videos, miner_latencies_ms = await generate_videos_for_miners(
                session, miners, image_b64, description, task_created_at
            )

        successful = {hk for hk, (vb, _, _) in miner_videos.items() if vb is not None}
        if not successful:
            log("No miners produced a video", "warn")
            return None

        for hotkey in successful:
            video_bytes, _, _ = miner_videos[hotkey]
            safe = hotkey.replace("/", "_").replace("\\", "_")[:16]
            p = f"/tmp/miner_{safe}_{task_id}.mp4"
            created.append(p)  # tracked before write so a probe failure can't leak it
            with open(p, "wb") as fh:
                fh.write(video_bytes)
            width, height = await get_video_resolution(p)
            if not _is_resolution_acceptable(width, height):
                log(
                    f"Miner {hotkey[:12]}...: dropping video, resolution {width}x{height} "
                    f"(required {REQUIRED_VIDEO_WIDTH}x{REQUIRED_VIDEO_HEIGHT} ±{VIDEO_RESOLUTION_TOLERANCE}px)",
                    "warn",
                )
                remove_file(p)
                continue
            miner_paths[hotkey] = p

        if not miner_paths:
            log("All miner videos rejected on resolution; nothing to upload", "warn")
            return None

        metadata = {
            "task_id": task_id,
            "created_at": datetime.now().isoformat(),
            "source": {
                "bucket": SOURCE_BUCKET,
                "key": video_key,
                "full_duration_seconds": one_shot.video_duration_seconds,
                "clip_start_seconds": start_offset,
                "clip_duration_seconds": CLIP_DURATION,
                "one_shot_segment_start_seconds": one_shot.segment_start_seconds,
                "one_shot_segment_end_seconds": one_shot.segment_end_seconds,
                "scene_cut_count": len(one_shot.scene_cuts),
                "scene_detection_threshold": ONE_SHOT_SCENE_THRESHOLD,
            },
            "prompt": {
                "model": GEMINI_DESCRIPTION_MODEL,
                "text": description,
                "description_source": "full_clip_video",
            },
            "miners": list(miner_paths.keys()),
            "miner_latencies_ms": {
                hk: latency for hk, latency in miner_latencies_ms.items() if hk in miner_paths
            },
        }

        result = SampledTask(
            task_id=task_id,
            clip_path=clip_path,
            frame_path=frame_path,
            metadata=metadata,
            miner_paths=miner_paths,
            temp_paths=[video_path],
        )
        success = True
        return result
    finally:
        if not success:
            # Failure or early return: remove every temp file created this round.
            for p in created:
                remove_file(p)
