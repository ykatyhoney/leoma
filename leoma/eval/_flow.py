"""Lazy optical-flow backend (OpenCV Farneback). Imported only for the ``flow`` metric.

Motion-fidelity distance: compute dense optical flow between consecutive frames
for both the generation and the real continuation, then compare the flow fields
(mean endpoint error). Stronger than the numpy ``temporal`` metric — it measures
actual per-pixel motion vectors, not just raw frame deltas.

Farneback runs on **CPU and is deterministic**, which is preferable for
cross-validator reproducibility over a GPU flow net. ``cv2`` is imported lazily
so this module stays import-safe without OpenCV installed; the endpoint-error
math (``flow_endpoint_error``) is pure numpy and unit-testable.
"""
from __future__ import annotations

import numpy as np


def _to_gray_uint8(frames: np.ndarray) -> np.ndarray:
    """(T, H, W, C) or (T, H, W) -> (T, H, W) uint8 grayscale."""
    arr = np.asarray(frames)
    if arr.ndim == 4:
        arr = arr.mean(axis=-1)
    return np.clip(arr, 0, 255).astype("uint8")


def flow_endpoint_error(flow_a: np.ndarray, flow_b: np.ndarray) -> float:
    """Mean endpoint error between two flow fields shaped (N, H, W, 2)."""
    a = np.asarray(flow_a, dtype=np.float64)
    b = np.asarray(flow_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.mean(np.sqrt(((a - b) ** 2).sum(axis=-1))))


def _farneback_flow(gray: np.ndarray) -> np.ndarray:
    """(T, H, W) uint8 -> (T-1, H, W, 2) dense flow between consecutive frames."""
    import cv2

    flows = []
    for i in range(gray.shape[0] - 1):
        flows.append(
            cv2.calcOpticalFlowFarneback(
                gray[i], gray[i + 1], None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
        )
    if not flows:
        return np.zeros((0, gray.shape[1], gray.shape[2], 2), dtype=np.float32)
    return np.stack(flows)


def flow_video_distance(gen: np.ndarray, truth: np.ndarray) -> float:
    """Mean optical-flow endpoint error between generation and truth motion."""
    g = _to_gray_uint8(gen)
    t = _to_gray_uint8(truth)
    n = min(g.shape[0], t.shape[0])
    if n < 2:
        return 0.0
    g, t = g[:n], t[:n]
    if g.shape[1:] != t.shape[1:]:
        raise ValueError(f"frame size mismatch: {g.shape[1:]} vs {t.shape[1:]}")
    return flow_endpoint_error(_farneback_flow(g), _farneback_flow(t))
