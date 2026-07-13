"""Paired-bootstrap verdict for the king-of-the-hill duel.

Leoma's duel metric is a per-clip **reference distance** (LPIPS / SSIM / FVD /
optical-flow) between a generation and the real ground-truth continuation —
lower is better, exactly analogous to Teutonic's per-sequence cross-entropy
loss. On the same held-out clips, king and challenger each produce one distance
per clip. We test whether the challenger is *confidently* better by paired
bootstrap over the per-clip advantage ``d = king_distance - challenger_distance``
(positive ⇒ challenger closer to ground truth):

  - resample the paired advantages ``n_bootstrap`` times (seeded, deterministic),
  - take the lower-confidence bound ``lcb = quantile(bootstrap means, alpha)``,
  - **crown the challenger iff ``lcb > delta_threshold``**.

The seed is derived from the block hash + hotkey (see ``app.validator.seeds``),
so every validator computes the identical verdict from the same scores.

``paired_bootstrap_verdict`` is a **pure function of its arguments** — nothing in
what it returns depends on when it ran. That is deliberate: the verdict is hashed
into a ``verdict_digest`` that validators compare against each other, and a
wall-clock timestamp inside it would make every honest validator's digest differ.
The eval server stamps ``produced_at`` on the response, *outside* the digested
surface.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def paired_bootstrap_verdict(
    king_scores: Sequence[float],
    challenger_scores: Sequence[float],
    *,
    delta_threshold: float,
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Return the duel verdict dict.

    ``king_scores[i]`` and ``challenger_scores[i]`` are the reference distances
    (lower = better) for the same clip ``i``. Raises ``ValueError`` on empty or
    mismatched-length inputs so a broken eval never silently crowns anyone.
    """
    king = np.asarray(king_scores, dtype=np.float64)
    challenger = np.asarray(challenger_scores, dtype=np.float64)
    if king.size == 0 or challenger.size == 0:
        raise ValueError("empty score arrays")
    if king.shape != challenger.shape:
        raise ValueError(f"score length mismatch: {king.shape} vs {challenger.shape}")

    diff = king - challenger  # positive ⇒ challenger closer to ground truth
    n = diff.shape[0]

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()

    mu_hat = float(diff.mean())
    lcb = float(np.quantile(boot, alpha))
    accepted = bool(lcb > delta_threshold)

    return {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "mu_hat": round(mu_hat, 6),
        "lcb": round(lcb, 6),
        "delta_threshold": delta_threshold,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "n_clips": n,
        "avg_king_distance": round(float(king.mean()), 6),
        "avg_challenger_distance": round(float(challenger.mean()), 6),
    }


def can_still_win(
    king_scores: Sequence[float],
    challenger_scores: Sequence[float],
    remaining: int,
    *,
    delta_threshold: float,
    best_possible_advantage: float,
) -> bool:
    """Early-stop guard: can the challenger still clear the threshold?

    Given the paired advantages seen so far plus ``remaining`` clips each able to
    add at most ``best_possible_advantage`` to the mean, returns False when even
    the most optimistic remaining outcome can't push the mean advantage above
    ``delta_threshold``. Used to abandon a hopeless duel early (the bootstrap LCB
    is always ≤ the sample mean, so mean ≤ delta ⇒ lcb ≤ delta ⇒ reject).
    """
    king = np.asarray(king_scores, dtype=np.float64)
    challenger = np.asarray(challenger_scores, dtype=np.float64)
    seen = king - challenger
    total = seen.size + remaining
    if total == 0:
        return True
    best_mean = (float(seen.sum()) + remaining * best_possible_advantage) / total
    return best_mean > delta_threshold
