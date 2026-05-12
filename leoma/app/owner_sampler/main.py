"""
Owner sampler: creates tasks, calls miners, uploads results to S3.

Runs as a separate process (not inside FastAPI). Every 30 minutes:
- Allocates next task_id from DB
- Gets valid miners from API
- Finds a source video with a valid one-shot 5s segment, extracts clip and first frame, gets GPT-4o description
- Calls each miner via Chutes, collects generated videos
- Uploads to S3: task_id/generated_videos/{miner_hotkey}.mp4, task_id/original_clip.mp4, etc.
- Updates latest_sampled_task_id so validators can poll GET /tasks/latest
"""

import os
import random
import asyncio
import base64
import time
from datetime import datetime
from typing import Dict, Any

import aiohttp
from minio import Minio
from openai import AsyncOpenAI

from leoma.bootstrap import (
    SOURCE_BUCKET,
    SAMPLES_BUCKET,
    MIN_VIDEO_SIZE,
    MAX_VIDEO_SIZE,
    CLIP_DURATION,
    CHUTES_API_KEY,
    REQUEST_TIMEOUT,
    REQUIRED_VIDEO_HEIGHT,
    REQUIRED_VIDEO_WIDTH,
    VIDEO_RESOLUTION_TOLERANCE,
    WALLET_NAME,
    HOTKEY_NAME,
)
from leoma.bootstrap import emit_log as log, emit_header as log_header, log_exception
from leoma.infra.db.pool import init_database
from leoma.infra.db.stores import SamplingStateStore
from leoma.infra.storage_backend import (
    create_samples_write_client,
    create_source_read_client,
    ensure_bucket_exists,
    upload_task_artifacts,
)
from leoma.infra.video_utils import (
    OneShotClipSelection,
    choose_one_shot_clip_start,
    extract_frames,
    frames_to_base64,
    extract_clip,
    extract_first_frame,
    get_video_resolution,
)
from leoma.infra.judge import get_description_async
from leoma.infra.chute_resolver import get_chute_info, build_chute_endpoint

# Interval between sampling rounds (seconds)
OWNER_SAMPLING_INTERVAL = int(os.environ.get("OWNER_SAMPLING_INTERVAL", "1200"))  # 20 min
API_URL = os.environ.get("API_URL", "https://api.leoma.ai")

# Track recently used videos (in-memory)
USED_VIDEOS: list[str] = []
MAX_VIDEO_HISTORY = int(os.environ.get("MAX_VIDEO_HISTORY", "100"))

# Concurrency
MAX_CONCURRENT_MINERS = int(os.environ.get("MAX_CONCURRENT_MINERS", "40"))
DESCRIPTION_MAX_FRAMES = int(os.environ.get("DESCRIPTION_MAX_FRAMES", "12"))
DESCRIPTION_FRAME_FPS = float(os.environ.get("DESCRIPTION_FRAME_FPS", "3"))
ONE_SHOT_SCENE_THRESHOLD = float(os.environ.get("ONE_SHOT_SCENE_THRESHOLD", "0.18"))
ONE_SHOT_BOUNDARY_MARGIN = float(os.environ.get("ONE_SHOT_BOUNDARY_MARGIN", "0.15"))
SAFE_CLIP_START_OFFSET_SECONDS = float(os.environ.get("SAFE_CLIP_START_OFFSET_SECONDS", "2.0"))


def _is_resolution_acceptable(width: int, height: int) -> bool:
    """Accept videos whose width and height are both within tolerance of the canonical 480p target."""
    return (
        abs(width - REQUIRED_VIDEO_WIDTH) <= VIDEO_RESOLUTION_TOLERANCE
        and abs(height - REQUIRED_VIDEO_HEIGHT) <= VIDEO_RESOLUTION_TOLERANCE
    )


async def _get_valid_miners_via_api() -> list[Dict[str, Any]]:
    """Get valid miners from the centralized API (requires wallet for /miners/valid)."""
    from leoma.infra.remote_api import create_api_client_from_wallet

    client = create_api_client_from_wallet(
        wallet_name=WALLET_NAME,
        hotkey_name=HOTKEY_NAME,
        api_url=API_URL,
    )
    try:
        miners = await client.get_valid_miners()
        return [
            {
                "uid": m.uid,
                "hotkey": m.hotkey,
                "chute_id": m.chute_id or "",
                "chute_slug": m.chute_slug or "",
                "model_name": getattr(m, "model_name", None),
                "model_revision": getattr(m, "model_revision", None),
                "model_hash": getattr(m, "model_hash", None),
                "block": getattr(m, "block", None),
            }
            for m in miners
        ]
    finally:
        await client.close()


def _build_generation_miners(valid_miners: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert validated miner list to generation payload map (hotkey -> info)."""
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


def _build_paths(task_id: int) -> tuple[str, str, str, str]:
    """Build temporary filesystem paths for a sampling round."""
    sid = str(task_id)
    video_path = f"/tmp/source_video_{sid}.mp4"
    clip_path = f"/tmp/original_clip_{sid}.mp4"
    frame_path = f"/tmp/first_frame_{sid}.png"
    original_frames_dir = f"/tmp/original_frames_{sid}"
    return video_path, clip_path, frame_path, original_frames_dir


def _remove_file(path: str | None) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _remove_directory(path: str | None) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
        os.rmdir(path)
    except OSError:
        pass


async def _list_source_videos(minio_client: Minio) -> list[str]:
    """List candidate source videos from the source bucket."""
    objects = await asyncio.to_thread(
        lambda: list(minio_client.list_objects(SOURCE_BUCKET, recursive=True))
    )
    return [
        obj.object_name
        for obj in objects
        if obj.object_name.endswith(".mp4") and MIN_VIDEO_SIZE < obj.size < MAX_VIDEO_SIZE
    ]


def _prioritize_video_keys(video_keys: list[str]) -> list[str]:
    """Prioritize unseen videos while keeping fallback behavior deterministic."""
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


async def _select_one_shot_video(
    minio_client: Minio,
    local_video_path: str,
) -> tuple[str, OneShotClipSelection] | None:
    """
    Iterate source videos until a one-shot segment can hold the full clip duration.
    Returns (video_key, one_shot_selection) or None.
    """
    video_keys = await _list_source_videos(minio_client)
    for video_key in _prioritize_video_keys(video_keys):
        try:
            _remove_file(local_video_path)
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
                log("Safe clip start offset is greater than max clip start; retrying next video", "warn")
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
    *,
    use_chutes_auth: bool = True,
) -> tuple[bytes | None, str | None]:
    """Generate video using a miner's I2V endpoint."""
    headers = {"Content-Type": "application/json"}
    if use_chutes_auth and CHUTES_API_KEY:
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


async def _generate_videos_for_miners(
    session: aiohttp.ClientSession,
    miners: Dict[str, Dict[str, Any]],
    image_b64: str,
    prompt: str,
    task_created_at: float,
) -> tuple[Dict[str, tuple[bytes | None, str | None, str | None]], Dict[str, int]]:
    """Generate videos from all miners concurrently.
    
    Returns (miner_videos, miner_latencies_ms). Latency is time from task_created_at
    to receiving the miner's response (how long the miner took to generate the video).
    """
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


async def run_owner_sampler_loop() -> None:
    """
    Main loop: every OWNER_SAMPLING_INTERVAL seconds, create a task, call miners, upload to S3.
    """
    await init_database()
    sampling_state_dao = SamplingStateStore()
    await sampling_state_dao.ensure_next_task_id_synced()
    source_client = create_source_read_client()
    samples_client = create_samples_write_client()
    await ensure_bucket_exists(samples_client, SAMPLES_BUCKET)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        log("OPENAI_API_KEY not set; description generation will fail", "warn")
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None

    log_header("Owner Sampler Starting")
    log(f"Sampling interval: {OWNER_SAMPLING_INTERVAL}s", "info")
    log(f"API: {API_URL}", "info")

    round_num = 0
    while True:
        round_num += 1
        round_start = time.time()
        video_path = clip_path = frame_path = original_frames_dir = None
        miner_paths: Dict[str, str] = {}

        async def _sleep_until_next_round() -> None:
            elapsed = time.time() - round_start
            sleep_time = max(1, int(OWNER_SAMPLING_INTERVAL - elapsed))
            log(f"Sleeping {sleep_time}s (interval={OWNER_SAMPLING_INTERVAL}s, elapsed={elapsed:.1f}s)...", "info")
            await asyncio.sleep(sleep_time)

        try:
            log_header(f"Owner Sampler Round #{round_num}")

            valid_miners = await _get_valid_miners_via_api()
            if not valid_miners:
                log("No valid miners from API; skipping round (task_id not incremented)", "warn")
                await _sleep_until_next_round()
                continue

            task_id = await sampling_state_dao.peek_next_task_id()
            log(f"Attempting task_id={task_id} (latest_task_id advances only after successful upload)", "info")

            miners = _build_generation_miners(valid_miners)
            log(f"Found {len(miners)} valid miners", "info")

            video_path, clip_path, frame_path, original_frames_dir = _build_paths(task_id)
            selected = await _select_one_shot_video(source_client, video_path)
            if not selected:
                log("No source video contains a 5s one-shot segment; retrying next round", "warn")
                await _sleep_until_next_round()
                continue
            video_key, one_shot = selected
            duration = one_shot.video_duration_seconds
            start_offset = one_shot.clip_start_seconds
            log(
                (
                    f"Selected one-shot source: key={video_key} "
                    f"clip_start={start_offset:.3f}s "
                    f"segment=[{one_shot.segment_start_seconds:.3f}, {one_shot.segment_end_seconds:.3f}]"
                ),
                "info",
            )

            await extract_clip(video_path, clip_path, start_offset, CLIP_DURATION)
            await extract_first_frame(clip_path, frame_path, start_offset=0)

            if not openai_client:
                log("Skipping round: no OpenAI client", "warn")
                await _sleep_until_next_round()
                continue

            original_frames = await extract_frames(
                clip_path,
                original_frames_dir,
                max_frames=DESCRIPTION_MAX_FRAMES,
                fps=DESCRIPTION_FRAME_FPS,
            )
            if len(original_frames) < 5:
                log("Not enough frames, skipping", "warn")
                continue

            original_frames_b64 = frames_to_base64(original_frames)
            description = await get_description_async(openai_client, original_frames_b64)
            log(f"Description: {description[:80]}...", "info")

            with open(frame_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            task_created_at = time.time()
            async with aiohttp.ClientSession() as session:
                miner_videos, miner_latencies_ms = await _generate_videos_for_miners(
                    session, miners, image_b64, description, task_created_at
                )

            successful = {
                hk for hk, (vb, _, _) in miner_videos.items() if vb is not None
            }
            if not successful:
                log("No miners produced video", "warn")
                await _sleep_until_next_round()
                continue

            for hotkey in successful:
                video_bytes, _, _ = miner_videos[hotkey]
                safe = hotkey.replace("/", "_").replace("\\", "_")[:16]
                p = f"/tmp/miner_{safe}_{task_id}.mp4"
                with open(p, "wb") as f:
                    f.write(video_bytes)
                width, height = await get_video_resolution(p)
                if not _is_resolution_acceptable(width, height):
                    log(
                        f"Miner {hotkey[:12]}...: dropping video, resolution "
                        f"{width}x{height} (required "
                        f"{REQUIRED_VIDEO_WIDTH}x{REQUIRED_VIDEO_HEIGHT} "
                        f"±{VIDEO_RESOLUTION_TOLERANCE}px)",
                        "warn",
                    )
                    _remove_file(p)
                    continue
                miner_paths[hotkey] = p

            if not miner_paths:
                log(
                    f"All miner videos rejected on resolution (required "
                    f"{REQUIRED_VIDEO_WIDTH}x{REQUIRED_VIDEO_HEIGHT} "
                    f"±{VIDEO_RESOLUTION_TOLERANCE}px); skipping upload",
                    "warn",
                )
                await _sleep_until_next_round()
                continue

            metadata = {
                "task_id": task_id,
                "created_at": datetime.now().isoformat(),
                "source": {
                    "bucket": SOURCE_BUCKET,
                    "key": video_key,
                    "full_duration_seconds": duration,
                    "clip_start_seconds": start_offset,
                    "clip_duration_seconds": CLIP_DURATION,
                    "one_shot_segment_start_seconds": one_shot.segment_start_seconds,
                    "one_shot_segment_end_seconds": one_shot.segment_end_seconds,
                    "scene_cut_count": len(one_shot.scene_cuts),
                    "scene_detection_threshold": ONE_SHOT_SCENE_THRESHOLD,
                },
                "prompt": {
                    "model": "gpt-4o",
                    "text": description,
                    "description_frame_count": len(original_frames),
                    "description_frame_fps": DESCRIPTION_FRAME_FPS,
                },
                "miners": list(miner_paths.keys()),
                "miner_latencies_ms": {
                    hk: latency
                    for hk, latency in miner_latencies_ms.items()
                    if hk in miner_paths
                },
            }

            await upload_task_artifacts(
                samples_client,
                task_id,
                clip_path,
                frame_path,
                metadata,
                miner_paths,
            )

            await sampling_state_dao.set_latest_task_id(task_id)
            elapsed = time.time() - round_start
            log(f"Task {task_id} uploaded; latest_task_id set ({len(miner_paths)} miners, {elapsed:.1f}s)", "success")

        except Exception as e:
            log(f"Owner sampler error: {e}", "error")
            log_exception("Owner sampler error", e)
        finally:
            _remove_file(video_path)
            _remove_file(clip_path)
            _remove_file(frame_path)
            _remove_directory(original_frames_dir)
            for p in (miner_paths or {}).values():
                _remove_file(p)

        await _sleep_until_next_round()
