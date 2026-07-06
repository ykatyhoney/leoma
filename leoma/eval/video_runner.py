"""Video-generation king-of-the-hill duel runner.

On the same held-out clips, king and challenger each generate a continuation
from the clip's first frame + prompt using the SAME per-clip seed; each
generation is scored against the real continuation with a reference metric, and
the per-clip distances feed the paired bootstrap verdict.

``run_duel`` is the pure orchestration — it takes *generate* callables and a
*distance* function, so the whole scoring→verdict flow is unit-testable with
numpy fakes. The torch/diffusers loading + generation used in production is
imported lazily (``load_video_pipeline`` / ``generate``); ``duel`` wires those
into ``run_duel``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from leoma.eval.bootstrap import can_still_win, paired_bootstrap_verdict
from leoma.eval.metrics import Metric, get_metric
from leoma.app.validator.seeds import clip_generation_seed


@dataclass
class GenParams:
    """Generation knobs shared by king and challenger (kept identical for fairness)."""

    num_frames: int = 81
    fps: int = 16
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    width: int = 832
    height: int = 480
    negative_prompt: str = "low quality, blurry, distorted"


@dataclass
class Clip:
    """One held-out duel item."""

    clip_index: int
    first_frame: np.ndarray            # (H, W, C) — the conditioning image
    prompt: str
    truth_frames: np.ndarray           # (T, H, W, C) — the real continuation (ground truth)
    params: GenParams = field(default_factory=GenParams)


# A generate callable: (clip, seed) -> generated frames (T, H, W, C).
GenerateFn = Callable[[Clip, int], np.ndarray]


def run_duel(
    clips: Sequence[Clip],
    generate_king: GenerateFn,
    generate_challenger: GenerateFn,
    distance_fn: Metric,
    *,
    master_seed: int,
    delta_threshold: float,
    alpha: float,
    n_bootstrap: int,
    on_phase: Optional[Callable[[dict], None]] = None,
    early_stop_max_advantage: Optional[float] = None,
) -> dict:
    """Score every clip and return the verdict plus per-clip distances.

    ``master_seed`` drives the per-clip generation seed (identical for king and
    challenger on a given clip). If ``early_stop_max_advantage`` is set, the duel
    abandons early once the challenger provably can't clear the threshold — the
    verdict stays "king" in that case, so the outcome is unchanged.
    """
    if not clips:
        raise ValueError("no clips to duel on")

    king_scores: list[float] = []
    challenger_scores: list[float] = []
    per_clip: list[dict] = []

    for pos, clip in enumerate(clips):
        gseed = clip_generation_seed(master_seed, clip.clip_index)
        king_frames = generate_king(clip, gseed)
        chall_frames = generate_challenger(clip, gseed)
        kd = float(distance_fn(king_frames, clip.truth_frames))
        cd = float(distance_fn(chall_frames, clip.truth_frames))
        king_scores.append(kd)
        challenger_scores.append(cd)
        per_clip.append({"clip_index": clip.clip_index, "king_distance": kd, "challenger_distance": cd})

        if on_phase:
            on_phase({
                "phase": "scored_clip",
                "position": pos + 1,
                "total": len(clips),
                "king_distance": round(kd, 6),
                "challenger_distance": round(cd, 6),
            })

        if early_stop_max_advantage is not None and not can_still_win(
            king_scores, challenger_scores, remaining=len(clips) - (pos + 1),
            delta_threshold=delta_threshold, best_possible_advantage=early_stop_max_advantage,
        ):
            if on_phase:
                on_phase({"phase": "early_stop", "reason": "challenger cannot clear threshold"})
            verdict = paired_bootstrap_verdict(
                king_scores, challenger_scores,
                delta_threshold=delta_threshold, alpha=alpha, n_bootstrap=n_bootstrap, seed=master_seed,
            )
            verdict["early_stopped"] = True
            verdict["per_clip"] = per_clip
            return verdict

    verdict = paired_bootstrap_verdict(
        king_scores, challenger_scores,
        delta_threshold=delta_threshold, alpha=alpha, n_bootstrap=n_bootstrap, seed=master_seed,
    )
    verdict["early_stopped"] = False
    verdict["per_clip"] = per_clip
    return verdict


# ---------------------------------------------------------------------------
# Production generation path (torch + diffusers) — imported lazily.
# ---------------------------------------------------------------------------

def load_video_pipeline(snapshot_dir: str, *, device: Optional[str] = None):
    """Load a diffusers I2V pipeline for a materialized model snapshot.

    Resolves the pipeline class from ``chain_config.ARCH_PIPELINE`` (the pinned
    base architecture), falling back to diffusers' auto image-to-video pipeline.
    King and challenger are the SAME architecture, so both load this way.
    """
    import torch
    import diffusers

    from leoma.infra.chain_config import ARCH_PIPELINE

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipeline_cls = getattr(diffusers, ARCH_PIPELINE, None) if ARCH_PIPELINE else None
    if pipeline_cls is None:
        pipeline_cls = diffusers.AutoPipelineForImage2Video
    pipe = pipeline_cls.from_pretrained(snapshot_dir, torch_dtype=dtype)
    pipe = pipe.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return pipe


def generate(pipeline, clip: Clip, seed: int) -> np.ndarray:
    """Run the I2V pipeline for one clip; returns frames (T, H, W, C) uint8."""
    import torch
    from PIL import Image

    p = clip.params
    device = getattr(pipeline, "device", None)
    gen = torch.Generator(device=str(device) if device is not None else "cpu").manual_seed(int(seed) & 0x7FFFFFFF)

    first = clip.first_frame
    image = first if isinstance(first, Image.Image) else Image.fromarray(np.asarray(first).astype("uint8"))
    image = image.convert("RGB").resize((p.width, p.height))

    with torch.inference_mode():
        result = pipeline(
            image=image,
            prompt=clip.prompt,
            negative_prompt=p.negative_prompt,
            num_frames=p.num_frames,
            num_inference_steps=p.num_inference_steps,
            guidance_scale=p.guidance_scale,
            generator=gen,
        )
    frames = result.frames[0]  # list of PIL frames (or ndarray)
    return np.stack([np.asarray(f) for f in frames])


def duel(
    king_pipeline,
    challenger_pipeline,
    clips: Sequence[Clip],
    *,
    master_seed: int,
    metric: str,
    delta_threshold: float,
    alpha: float,
    n_bootstrap: int,
    generate_fn: GenerateFn = None,  # type: ignore[assignment]
    on_phase: Optional[Callable[[dict], None]] = None,
    early_stop_max_advantage: Optional[float] = None,
) -> dict:
    """Production wrapper: bind loaded pipelines + the named metric into ``run_duel``."""
    gen = generate_fn or generate
    distance_fn = get_metric(metric)
    return run_duel(
        clips,
        generate_king=lambda c, s: gen(king_pipeline, c, s),
        generate_challenger=lambda c, s: gen(challenger_pipeline, c, s),
        distance_fn=distance_fn,
        master_seed=master_seed,
        delta_threshold=delta_threshold,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        on_phase=on_phase,
        early_stop_max_advantage=early_stop_max_advantage,
    )
