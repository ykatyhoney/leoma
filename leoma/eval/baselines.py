"""The freeze cheat, materialized as an opponent.

Leoma scores *closeness to the real continuation*. That creates a cheat class the
reference subnet structurally cannot have: a model that simply **holds the
conditioning frame** — emits it over and over — scores well on any clip that doesn't
move much, without having learned anything at all.

``metrics.require_frames`` already closed the crudest version (a 1-frame generation
was being scored against a 1-frame truth, i.e. against the very frame the model was
handed). But a *full-length* freeze is still a legal generation, and on a low-motion
clip it can genuinely beat a mediocre king.

**The design: make the cheat a duelist.** :func:`freeze_frames` is a ``GenerateFn``
like any other, so the freeze baseline runs through the *same* ``run_duel`` loop, the
*same* metric, the *same* clips, and the *same* ``paired_bootstrap_verdict`` as the
king and the challenger. There is no second statistical path to get wrong, and no new
threshold semantics to reason about: a challenger must beat **the king** *and* **the
cheat**, each with bootstrap confidence.

A challenger that fails the freeze gate is rejected **even if it beat the king** —
which is exactly the point. A freeze-cheater must not inherit a mediocre crown.

The margin is **scale-free** (a fraction of the freeze baseline's own mean distance),
so it survives any metric recalibration. Tightening it later means changing one
number, not re-deriving it per metric — LPIPS, MSE, flow and composites all live on
wildly different numeric scales, and a hard-coded absolute margin would silently mean
something different on each.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from leoma.eval.bootstrap import paired_bootstrap_verdict
from leoma.eval.video_runner import Clip


def freeze_frames(clip: Clip, seed: int = 0) -> np.ndarray:
    """The cheat: the conditioning frame, repeated to the truth's length.

    Signature-compatible with ``GenerateFn`` on purpose — it is a *duelist*, not a
    special case. ``seed`` is ignored: the whole point of the cheat is that it does
    not generate anything.
    """
    first = np.asarray(clip.first_frame, dtype="uint8")
    n = int(np.asarray(clip.truth_frames).shape[0])
    return np.repeat(first[None, ...], n, axis=0)


def freeze_scores(clips: Sequence[Clip], distance_fn) -> list[float]:
    """What the freeze cheat would score on exactly these clips.

    Cheap: no generation, no GPU. Just the metric, run against a repeated frame — a
    rounding error next to the hours the real generations take.
    """
    return [float(distance_fn(freeze_frames(clip), clip.truth_frames)) for clip in clips]


def freeze_gate(
    challenger_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    margin_fraction: float,
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Must the challenger be *confidently* better than the freeze cheat?

    Reuses ``paired_bootstrap_verdict`` verbatim, with the cheat standing in for the
    king. ``lcb > margin`` therefore means exactly what it means everywhere else in
    the subnet: "better than its opponent, with confidence, by at least this much".

    The margin is ``margin_fraction × mean(baseline)`` — a fraction of the cheat's own
    score, so it is dimensionless and moves with the metric. At the launch setting of
    **0.0** the gate is LCB-only: the challenger need only be *confidently better than
    freezing*, with no additional headroom demanded. That is the honest place to start
    — we have not yet measured, on real hardware, how much headroom a genuinely good
    model has over the cheat, and a margin guessed too high would reject good models
    before we ever saw one. ``avg_freeze_distance`` is published on the dashboard
    precisely so that number can be measured rather than invented.
    """
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    margin = float(margin_fraction) * float(baseline.mean()) if baseline.size else 0.0

    verdict = paired_bootstrap_verdict(
        baseline_scores,          # the "king" seat: the cheat
        challenger_scores,        # the challenger, unchanged
        delta_threshold=margin,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "passed": bool(verdict["accepted"]),
        "margin": round(margin, 6),
        "margin_fraction": float(margin_fraction),
        "lcb": verdict["lcb"],
        "mu_hat": verdict["mu_hat"],
        "avg_freeze_distance": verdict["avg_king_distance"],
        "avg_challenger_distance": verdict["avg_challenger_distance"],
    }


def evaluate_freeze_gates(
    clips: Sequence[Clip],
    king_scores: Sequence[float],
    challenger_scores: Sequence[float],
    distance_fn,
    *,
    margin_fraction: float,
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Run the freeze baseline against BOTH duelists.

    The king is checked too, but **report-only** — see :func:`king_gate_is_advisory`.
    """
    baseline = freeze_scores(clips, distance_fn)

    challenger = freeze_gate(
        challenger_scores, baseline,
        margin_fraction=margin_fraction, alpha=alpha, n_bootstrap=n_bootstrap, seed=seed,
    )
    king = freeze_gate(
        king_scores, baseline,
        margin_fraction=margin_fraction, alpha=alpha, n_bootstrap=n_bootstrap, seed=seed,
    )

    return {
        "baseline": "freeze",
        "avg_freeze_distance": challenger["avg_freeze_distance"],
        "challenger": challenger,
        "king": king,
        # The gate that actually decides anything.
        "challenger_passed": challenger["passed"],
        # An alarm, not a verdict. See below.
        "king_failed": not king["passed"],
    }


def king_gate_is_advisory() -> str:
    """Why a KING that fails the freeze gate is not automatically deposed.

    It is tempting: a king no better than a frozen frame is obviously not a good king.
    But auto-dethroning on a *gate* failure hands an attacker a lever. Push the corpus
    toward static clips (or wait for a corpus rotation that happens to be low-motion)
    and the incumbent "fails" — at which point the throne falls to whichever marginal
    challenger is next in the queue. The gate would become the attack.

    So a failing king raises a loud alarm — in the verdict, in the history, on the
    dashboard, in ``/health`` — and an operator responds by re-seeding via
    ``chain.toml``. Deposing a king is a decision with a human in it.
    """
    return (
        "A king that fails the freeze gate is reported, not deposed: auto-dethroning on a "
        "gate failure would let an attacker shift the corpus static and hand the crown to a "
        "marginal challenger. Re-seed via chain.toml instead."
    )


__all__ = [
    "evaluate_freeze_gates",
    "freeze_frames",
    "freeze_gate",
    "freeze_scores",
    "king_gate_is_advisory",
]
