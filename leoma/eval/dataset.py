"""Deterministic held-out duel clips from the source-video corpus.

Builds the ``Clip`` list a duel runs on: from the master seed (block hash +
hotkey) it picks source videos, carves a one-shot window from each, and returns
the conditioning first frame + the **real continuation** (ground truth) + a
prompt. Because every validator lists the same sorted corpus and runs the same
seeded selection, they all build the identical clip set.

Corpus + ffmpeg bound — imports (PIL/numpy) are lazy and the source download +
frame extraction reuse the shared ``storage_backend`` / ``video_utils`` helpers
(which survive the Phase-3 sampler removal). Run via the sync ``build_duel_clips``
wrapper (the eval server calls it from a worker thread).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import List, Optional

from leoma.bootstrap import (
    SOURCE_BUCKET,
    MIN_VIDEO_SIZE,
    MAX_VIDEO_SIZE,
    emit_log as log,
)
from leoma.infra.storage_backend import create_source_read_client
from leoma.infra.video_utils import (
    choose_one_shot_clip_start,
    extract_clip,
    extract_frames,
)
from leoma.app.validator.seeds import clip_generation_seed
from leoma.eval.video_runner import Clip, GenParams

# Deterministic default prompt for the I2V pipeline. King and challenger receive
# the SAME prompt per clip, so fairness holds regardless; a per-clip caption
# sidecar can replace this later without breaking determinism.
DEFAULT_DUEL_PROMPT = os.environ.get("LEOMA_DUEL_PROMPT", "")


async def _list_source_videos(client) -> List[str]:
    """Eligible source keys, sorted for a stable cross-validator order."""
    objects = await asyncio.to_thread(
        lambda: list(client.list_objects(SOURCE_BUCKET, recursive=True))
    )
    keys = [
        obj.object_name
        for obj in objects
        if obj.object_name.endswith(".mp4") and MIN_VIDEO_SIZE < obj.size < MAX_VIDEO_SIZE
    ]
    return sorted(keys)


def _deterministic_index_order(master_seed: int, total: int) -> List[int]:
    """A deterministic permutation of ``range(total)`` for the try order."""
    import numpy as np

    rng = np.random.default_rng(master_seed)
    return [int(i) for i in rng.permutation(total)]


async def _load_frames(paths: List[str], width: int, height: int):
    import numpy as np
    from PIL import Image

    frames = [
        np.asarray(Image.open(p).convert("RGB").resize((width, height)))
        for p in paths
    ]
    return np.stack(frames) if frames else np.empty((0, height, width, 3), dtype="uint8")


async def _build_one_clip(client, key: str, clip_index: int, gseed: int, params: GenParams) -> Optional[Clip]:
    """Download a source video and carve one deterministic ground-truth clip."""
    clip_duration = params.num_frames / max(1, params.fps)
    tmpdir = tempfile.mkdtemp(prefix="leoma-duel-")
    src = os.path.join(tmpdir, "src.mp4")
    clip_mp4 = os.path.join(tmpdir, "clip.mp4")
    frames_dir = os.path.join(tmpdir, "frames")
    try:
        await asyncio.to_thread(client.fget_object, SOURCE_BUCKET, key, src)
        selection = await choose_one_shot_clip_start(src, clip_duration, seed=gseed)
        if selection is None:
            return None
        await extract_clip(src, clip_mp4, selection.clip_start_seconds, clip_duration)
        frame_paths = await extract_frames(clip_mp4, frames_dir, max_frames=params.num_frames, fps=params.fps)
        if not frame_paths:
            return None
        truth = await _load_frames(frame_paths, params.width, params.height)
        if truth.shape[0] == 0:
            return None
        return Clip(
            clip_index=clip_index,
            first_frame=truth[0],
            prompt=DEFAULT_DUEL_PROMPT,
            truth_frames=truth,
            params=params,
        )
    except Exception as e:
        log(f"duel clip {key} failed: {e}", "warn")
        return None
    finally:
        for p in (src, clip_mp4):
            try:
                os.remove(p)
            except OSError:
                pass


async def _collect(master_seed: int, n_clips: int, params: GenParams) -> List[Clip]:
    client = create_source_read_client()
    keys = await _list_source_videos(client)
    total = len(keys)
    if total == 0:
        return []

    order = _deterministic_index_order(master_seed, total)
    # Try more than n so transient one-shot failures still yield n clips; the
    # order is deterministic so every validator converges on the same set.
    max_attempts = min(total, max(n_clips * 4, n_clips + 20))

    clips: List[Clip] = []
    for idx in order[:max_attempts]:
        if len(clips) >= n_clips:
            break
        gseed = clip_generation_seed(master_seed, idx)
        clip = await _build_one_clip(client, keys[idx], idx, gseed, params)
        if clip is not None:
            clips.append(clip)
    return clips


def build_duel_clips(master_seed: int, n_clips: int, params: GenParams) -> List[Clip]:
    """Sync entry point (called from the eval server's worker thread)."""
    return asyncio.run(_collect(master_seed, n_clips, params))
