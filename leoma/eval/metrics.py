"""Reference-distance metrics for the video duel.

Each metric scores how far a generated clip is from the **real ground-truth
continuation** — lower = better. king and challenger are scored with the same
metric on the same clips, and the per-clip distances feed the paired bootstrap.

``mse`` and ``ssim`` are pure-numpy (deterministic, dependency-light, and unit
testable). ``lpips`` is the perceptual metric that best tracks human video
quality but needs torch; it is imported lazily so this module stays import-safe
without a GPU. Select via ``get_metric(name)`` (name comes from config/env).
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


def _align(gen: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate both to the shortest common frame count; require equal H,W."""
    g = _as_float_gray(gen)
    t = _as_float_gray(truth)
    n = min(g.shape[0], t.shape[0])
    if n == 0:
        raise ValueError("no frames to compare")
    g, t = g[:n], t[:n]
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


def lpips_distance(gen: np.ndarray, truth: np.ndarray) -> float:
    """Learned perceptual distance (LPIPS). Lazily loads torch + the lpips net."""
    from leoma.eval._lpips import lpips_video_distance  # lazy heavy import

    return lpips_video_distance(gen, truth)


_METRICS: dict[str, Metric] = {
    "mse": mse,
    "ssim": ssim_distance,
    "lpips": lpips_distance,
}

DEFAULT_METRIC = "lpips"


def get_metric(name: str | None) -> Metric:
    key = (name or DEFAULT_METRIC).strip().lower()
    if key not in _METRICS:
        raise ValueError(f"unknown reference metric {name!r}; choices: {sorted(_METRICS)}")
    return _METRICS[key]
