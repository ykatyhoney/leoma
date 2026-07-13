"""Reference-distance metrics for the video duel.

Each metric scores how far a generated clip is from the **real ground-truth
continuation** — lower = better. king and challenger are scored with the same
metric on the same clips, and the per-clip distances feed the paired bootstrap.

Metrics span three axes:
  - spatial fidelity (per-frame): ``mse`` < ``ssim`` < ``lpips`` in sophistication
  - temporal fidelity (motion):   ``temporal`` (numpy) < ``flow`` (optical flow)
  - semantic fidelity:            ``clip`` (embedding cosine distance to reality)

``mse``/``ssim``/``temporal`` are pure-numpy (deterministic, dependency-light,
unit-testable). The learned/heavier metrics are imported lazily so this module
stays import-safe without their deps: ``lpips`` (torch), ``flow`` (OpenCV, CPU
and deterministic), ``clip`` (torch + open_clip). ``make_composite`` /
``"composite:lpips=1.0,flow=0.5"`` blends several into the single per-clip scalar
the duel needs. Select via ``get_metric(name)`` (from ``LEOMA_DUEL_METRIC``).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# A metric maps (generated_frames, truth_frames) -> mean per-frame distance.
# Frames are numpy arrays shaped (T, H, W, C) or (T, H, W), any numeric dtype.
Metric = Callable[[np.ndarray, np.ndarray], float]

_L = 255.0  # dynamic range for 8-bit frames


def _as_float_gray(frames: np.ndarray) -> np.ndarray:
    """Coerce frames to float64 grayscale, shape (T, H, W)."""
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim == 4:  # (T, H, W, C) -> average channels
        arr = arr.mean(axis=-1)
    elif arr.ndim == 2:  # (H, W) -> single frame
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected frames of rank 2/3/4, got shape {np.asarray(frames).shape}")
    return arr


class ShortGeneration(ValueError):
    """The generation has fewer frames than the ground truth."""


def require_frames(gen: np.ndarray, truth: np.ndarray) -> None:
    """A generation shorter than the truth is REJECTED, never truncated-to-fit.

    This is the root of the freeze/1-frame cheat, and it is the single
    highest-leverage guard in the scoring stack.

    ``_align`` used to truncate BOTH arrays to ``min(T_gen, T_truth)``. So a
    1-frame generation was compared against a 1-frame *truth* — and frame 0 of the
    truth is exactly **the conditioning frame the model was handed**. A model that
    emitted a single frame therefore scored near-perfectly on *every* metric,
    including the production default (lpips):

        flow / clip / temporal -> 0.0   (degenerate early-returns)
        mse / ssim / lpips     -> ~0    (truth truncated to the input frame)

    Requiring at least as many frames as the truth removes the exploit at its
    source; the degeneracy and freeze-baseline gates are defence in depth on top.

    A LONGER generation is fine — it is truncated to the truth's length.
    """
    n_gen = int(np.asarray(gen).shape[0]) if np.asarray(gen).ndim >= 3 else 1
    n_truth = int(np.asarray(truth).shape[0]) if np.asarray(truth).ndim >= 3 else 1
    if n_truth == 0:
        raise ValueError("no frames to compare")
    if n_gen < n_truth:
        raise ShortGeneration(
            f"generation too short: {n_gen} frames < {n_truth} ground-truth frames. "
            "A generation shorter than the truth cannot be scored (it would be graded "
            "against the conditioning frame it was given)."
        )


def _align(gen: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align a generation to the truth. The generation may not be SHORTER."""
    require_frames(gen, truth)
    g = _as_float_gray(gen)
    t = _as_float_gray(truth)
    n = t.shape[0]          # the truth's length is authoritative
    if n == 0:
        raise ValueError("no frames to compare")
    g, t = g[:n], t[:n]     # a longer generation is truncated; a shorter one already raised
    if g.shape[1:] != t.shape[1:]:
        raise ValueError(f"frame size mismatch: {g.shape[1:]} vs {t.shape[1:]}")
    return g, t


def mse(gen: np.ndarray, truth: np.ndarray) -> float:
    """Mean squared error per pixel, averaged over frames. 0 = identical."""
    g, t = _align(gen, truth)
    return float(np.mean((g - t) ** 2))


def _frame_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Global (single-window) SSIM in [-1, 1] for one grayscale frame."""
    c1 = (0.01 * _L) ** 2
    c2 = (0.03 * _L) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(num / den)


def ssim_distance(gen: np.ndarray, truth: np.ndarray) -> float:
    """1 - mean SSIM over frames. 0 = identical, larger = less similar."""
    g, t = _align(gen, truth)
    sims = [_frame_ssim(g[i], t[i]) for i in range(g.shape[0])]
    return float(1.0 - np.mean(sims))


def temporal_distance(gen: np.ndarray, truth: np.ndarray) -> float:
    """Motion-fidelity distance: how well the generation's frame-to-frame change
    matches the real continuation's.

    MSE/SSIM/LPIPS are spatial (per-frame) — they don't see whether the *motion*
    is right, so a frozen or flickering generation can still score well. This
    compares the temporal derivatives (Δframe) of gen vs truth, penalizing both
    too-static and too-jittery motion. 0 = identical motion dynamics.
    """
    g, t = _align(gen, truth)
    if g.shape[0] < 2:
        # Used to return 0.0 — a PERFECT score for a single-frame generation.
        raise ShortGeneration("temporal distance needs at least 2 frames")
    dg = np.diff(g, axis=0)  # (T-1, H, W): per-pixel motion of the generation
    dt = np.diff(t, axis=0)  # ... of the real continuation
    return float(np.mean((dg - dt) ** 2))


# The lazy metrics guard BEFORE their heavy import, so the anti-cheat check is
# unit-testable on a box with no torch / no OpenCV installed.
def lpips_distance(gen: np.ndarray, truth: np.ndarray) -> float:
    """Learned perceptual distance (LPIPS). Lazily loads torch + the lpips net."""
    require_frames(gen, truth)
    from leoma.eval._lpips import lpips_video_distance  # lazy heavy import

    return lpips_video_distance(gen, truth)


def flow_distance(gen: np.ndarray, truth: np.ndarray) -> float:
    """Optical-flow motion distance vs the real continuation. Lazily loads OpenCV."""
    require_frames(gen, truth)
    from leoma.eval._flow import flow_video_distance  # lazy heavy import

    return flow_video_distance(gen, truth)


def clip_distance(gen: np.ndarray, truth: np.ndarray) -> float:
    """CLIP semantic distance vs the real continuation. Lazily loads torch + open_clip."""
    require_frames(gen, truth)
    from leoma.eval._clip import clip_video_distance  # lazy heavy import

    return clip_video_distance(gen, truth)


_METRICS: dict[str, Metric] = {
    "mse": mse,
    "ssim": ssim_distance,
    "temporal": temporal_distance,
    "flow": flow_distance,
    "lpips": lpips_distance,
    "clip": clip_distance,
}

DEFAULT_METRIC = "lpips"


def make_composite(weights: dict) -> Metric:
    """Build a single per-clip metric that blends several registered metrics.

    ``weights`` maps metric name -> weight, e.g. ``{"lpips": 1.0, "temporal": 0.5}``
    scores spatial *and* motion fidelity in the one scalar the duel bootstrap
    needs. NOTE: the metrics live on different numeric scales, so the weights set
    both the relative importance AND the scale normalization — calibrate them on
    real data (this is part of the delta/alpha calibration step).
    """
    if not weights:
        raise ValueError("composite needs at least one weighted metric")
    parts = [(get_metric(name), float(w)) for name, w in weights.items()]

    def composite(gen: np.ndarray, truth: np.ndarray) -> float:
        return float(sum(w * fn(gen, truth) for fn, w in parts))

    return composite


def get_metric(name: str | None) -> Metric:
    """Resolve a metric by name.

    Besides the registered names, accepts a composite spec
    ``"composite:lpips=1.0,temporal=0.5"`` so a blended metric is selectable
    purely from config (``LEOMA_DUEL_METRIC``).
    """
    key = (name or DEFAULT_METRIC).strip()
    if key.lower().startswith("composite:"):
        spec = key.split(":", 1)[1]
        weights: dict = {}
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            metric_name, _, w = part.partition("=")
            weights[metric_name.strip().lower()] = float(w) if w else 1.0
        return make_composite(weights)

    key = key.lower()
    if key not in _METRICS:
        raise ValueError(f"unknown reference metric {name!r}; choices: {sorted(_METRICS)} or 'composite:...'")
    return _METRICS[key]
