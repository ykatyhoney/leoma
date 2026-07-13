"""Degenerate-generation policy — reject unscoreable output before it is scored.

The metrics themselves now refuse a generation shorter than the truth (that was the
freeze cheat's root, fixed in ``metrics.require_frames``). This module is the layer
above: it turns "unscoreable" into a *typed, deterministic verdict* instead of a
``ValueError`` that surfaces as a generic duel error and gets the challenger
retried four times for output that will be exactly as broken next time.

Every check here is a pure function of the frames, so **every validator reaches the
same conclusion about the same generation** — a rejection is part of consensus, not
a local accident.

The motion floor is deliberately **not** here. It is a *cheat* gate, it needs the
truth's own motion to compare against, and it belongs with the freeze-baseline gate.
This module answers only: can this array be scored at all?
"""
from __future__ import annotations

from leoma.eval.errors import DegenerateGeneration


def validate_generation(frames, *, expected_frames: int, width: int, height: int, who: str = "challenger"):
    """Check a generation is scoreable; raise :class:`DegenerateGeneration` if not.

    Returns the frames as a ``(T, H, W, 3)`` uint8 array so callers can use the
    validated value directly.
    """
    import numpy as np

    arr = np.asarray(frames)

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise DegenerateGeneration(
            f"{who} produced frames of shape {arr.shape}; expected "
            f"({expected_frames}, {height}, {width}, 3)"
        )

    if arr.shape[0] < expected_frames:
        raise DegenerateGeneration(
            f"{who} produced {arr.shape[0]} frames, fewer than the {expected_frames} "
            "ground-truth frames it is scored against. A short generation is scored "
            "against a truncated truth, whose first frame is the conditioning frame "
            "the model was handed — i.e. a free near-perfect score. Rejected."
        )

    if arr.shape[1] != height or arr.shape[2] != width:
        raise DegenerateGeneration(
            f"{who} produced {arr.shape[2]}x{arr.shape[1]} frames; the duel is pinned "
            f"to {width}x{height}"
        )

    # NaN/Inf can't survive a uint8 cast, so only float output can carry them —
    # and a NaN distance would poison the bootstrap rather than lose the duel.
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        raise DegenerateGeneration(f"{who} produced non-finite frame values (NaN/Inf)")

    return np.ascontiguousarray(arr[:expected_frames].astype("uint8"))


__all__ = ["validate_generation"]
