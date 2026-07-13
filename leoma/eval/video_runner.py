"""Video-generation king-of-the-hill duel runner.

On the same held-out clips, king and challenger each generate a continuation from
the clip's first frame + prompt using the SAME per-clip seed; each generation is
scored against the real continuation with a reference metric, and the per-clip
distances feed the paired bootstrap verdict.

``run_duel`` is the pure orchestration — it takes *generate* callables and a
*distance* function, so the whole scoring→verdict flow is unit-testable with numpy
fakes. The torch/diffusers loading + generation used in production is imported
lazily (``load_video_pipeline`` / ``generate``); ``duel`` wires those into
``run_duel``.

Two things ``run_duel`` now records that it did not before, both so a disagreement
between validators can be *localized* instead of argued about:

* **per-clip generated-frame digests** — if two validators' distances differ, the
  digests say immediately whether the generations differed (GPU noise) or only the
  scoring did (a broken box);
* **the clip's manifest id**, alongside the index the seed was derived from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from leoma.eval.bootstrap import can_still_win, paired_bootstrap_verdict
from leoma.eval.digests import digest_frames
from leoma.eval.errors import DuelCancelled
from leoma.eval.guards import validate_generation
from leoma.eval.metrics import Metric, get_metric
from leoma.app.validator.seeds import clip_generation_seed


@dataclass(frozen=True)
class GenParams:
    """Generation knobs shared by king and challenger (identical for both, by construction).

    Built from the pinned :class:`~leoma.eval.spec.GenSpec` — the defaults here are
    only for tests. In production nothing constructs a ``GenParams`` out of thin
    air: ``from_spec`` is the only path, because a default that silently differs
    from the chain's pin is the exact bug this whole tier exists to kill.
    """

    num_frames: int = 81
    fps: int = 16
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    width: int = 832
    height: int = 480
    negative_prompt: str = "low quality, blurry, distorted"
    dtype: str = "bfloat16"
    offload: str = "none"

    @classmethod
    def from_spec(cls, gen) -> "GenParams":
        return cls(
            num_frames=gen.num_frames,
            fps=gen.fps,
            num_inference_steps=gen.num_inference_steps,
            guidance_scale=gen.guidance_scale,
            width=gen.width,
            height=gen.height,
            negative_prompt=gen.negative_prompt,
            dtype=gen.dtype,
            offload=gen.offload,
        )


@dataclass
class Clip:
    """One held-out duel item."""

    clip_index: int                    # index into the corpus manifest; the seed hangs off it
    first_frame: np.ndarray            # (H, W, C) — the conditioning image
    prompt: str
    truth_frames: np.ndarray           # (T, H, W, C) — the real continuation (ground truth)
    params: GenParams = field(default_factory=GenParams)
    clip_id: str = ""                  # the manifest's stable id, for the audit block


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
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Score every clip and return the verdict plus per-clip distances.

    ``master_seed`` drives the per-clip generation seed (identical for king and
    challenger on a given clip). If ``early_stop_max_advantage`` is set, the duel
    abandons early once the challenger provably can't clear the threshold — the
    verdict stays "king" in that case, so the outcome is unchanged and a validator
    running with early stop still agrees with one running without it.

    ``should_cancel`` is checked **between clips** — the only honest granularity. A
    single generation is one uninterruptible call into diffusers; pretending we can
    stop mid-clip would just be a lie that leaves the GPU busy anyway.
    """
    if not clips:
        raise ValueError("no clips to duel on")

    king_scores: list[float] = []
    challenger_scores: list[float] = []
    per_clip: list[dict] = []

    for pos, clip in enumerate(clips):
        if should_cancel and should_cancel():
            raise DuelCancelled(f"duel cancelled after {pos} of {len(clips)} clips")

        gseed = clip_generation_seed(master_seed, clip.clip_index)
        p = clip.params
        expected = int(np.asarray(clip.truth_frames).shape[0])

        king_frames = validate_generation(
            generate_king(clip, gseed),
            expected_frames=expected, width=p.width, height=p.height, who="king",
        )
        chall_frames = validate_generation(
            generate_challenger(clip, gseed),
            expected_frames=expected, width=p.width, height=p.height, who="challenger",
        )

        kd = float(distance_fn(king_frames, clip.truth_frames))
        cd = float(distance_fn(chall_frames, clip.truth_frames))
        king_scores.append(kd)
        challenger_scores.append(cd)
        per_clip.append({
            "clip_index": clip.clip_index,
            "clip_id": clip.clip_id,
            "gen_seed": gseed,
            "king_distance": kd,
            "challenger_distance": cd,
            "king_frames_digest": digest_frames(king_frames),
            "challenger_frames_digest": digest_frames(chall_frames),
        })

        if on_phase:
            on_phase({
                "phase": "scored_clip",
                "position": pos + 1,
                "total": len(clips),
                "clip_id": clip.clip_id,
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

_TORCH_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def load_video_pipeline(snapshot_dir: str, *, gen, device: Optional[str] = None):
    """Load the pinned diffusers I2V pipeline for a materialized model snapshot.

    The pipeline class comes from the **request's** :class:`~leoma.eval.spec.ArchSpec`,
    not from this box's environment, and an unresolvable class is a hard error.
    The old code fell back to ``AutoPipelineForImage2Video`` when the pinned class
    couldn't be resolved — which meant the "pinned" pipeline was not actually
    pinned: a box with the wrong diffusers version would quietly load a *different*
    pipeline and generate different video from the same weights.
    """
    import torch
    import diffusers

    from leoma.infra.chain_config import SPEC

    pipeline_name = SPEC.arch.pipeline
    pipeline_cls = getattr(diffusers, pipeline_name, None)
    if pipeline_cls is None:
        raise RuntimeError(
            f"pinned pipeline {pipeline_name!r} does not exist in diffusers "
            f"{diffusers.__version__}. This box cannot run the duel the chain specifies; "
            "upgrade diffusers rather than falling back to a different pipeline."
        )

    if torch.cuda.is_available():
        dtype = getattr(torch, _TORCH_DTYPES[gen.dtype])
    else:
        dtype = torch.float32  # CPU has no usable bf16 path; tests only

    pipe = pipeline_cls.from_pretrained(snapshot_dir, torch_dtype=dtype)
    pipe = pipe.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return pipe


def generate(pipeline, clip: Clip, seed: int) -> np.ndarray:
    """Run the I2V pipeline for one clip; returns frames (T, H, W, C) uint8."""
    import torch
    from PIL import Image

    from leoma.app.validator.seeds import torch_seed

    p = clip.params
    device = getattr(pipeline, "device", None)
    gen = torch.Generator(device=str(device) if device is not None else "cpu")
    gen.manual_seed(torch_seed(seed))

    first = clip.first_frame
    image = first if isinstance(first, Image.Image) else Image.fromarray(np.asarray(first).astype("uint8"))
    image = image.convert("RGB").resize((p.width, p.height))

    with torch.inference_mode():
        result = pipeline(
            image=image,
            prompt=clip.prompt,
            negative_prompt=p.negative_prompt,
            height=p.height,
            width=p.width,
            num_frames=p.num_frames,
            num_inference_steps=p.num_inference_steps,
            guidance_scale=p.guidance_scale,
            generator=gen,
            output_type="np",
        )
    frames = result.frames[0]
    arr = np.asarray(frames)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype("uint8")


def duel(
    king_pipeline,
    challenger_pipeline,
    clips: Sequence[Clip],
    *,
    master_seed: int,
    metric: str,
    metric_device: str,
    delta_threshold: float,
    alpha: float,
    n_bootstrap: int,
    generate_fn: GenerateFn = None,  # type: ignore[assignment]
    on_phase: Optional[Callable[[dict], None]] = None,
    early_stop_max_advantage: Optional[float] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Production wrapper: bind loaded pipelines + the named metric into ``run_duel``."""
    gen = generate_fn or generate
    distance_fn = get_metric(metric, device=metric_device)
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
        should_cancel=should_cancel,
    )
