"""Which physical GPU each duelist generates on — a throughput knob, not a consensus one.

Today king and challenger load onto the **same** default CUDA device and generate
**sequentially**: king's clip, then the challenger's clip, one at a time. On an 8×H100
box that wastes seven idle GPUs for the length of every duel. Since each clip is
independent and both duelists use the same per-clip seed, king and challenger can
generate on two *separate* devices at the same time — roughly halving duel wall-clock.

**Why this is deliberately NOT part of the consensus surface** (``chain.toml`` /
:class:`~leoma.eval.spec.ConsensusSpec`): generation was already conceded to be
non-bit-exact across GPU *architectures* (see ``determinism.py`` and
``eval/calibrate.py``), and ``delta_threshold`` exists precisely to absorb that noise
regardless of its source. Whether two duelists happen to run on the same physical
card or two different (but identical) cards in the same box is exactly that kind of
generation-side noise, not a fact about the exam. So it is a per-box performance
setting, resolved from the environment, exactly like ``metric_device``.

**The one thing operators must not forget:** enabling this changes *what* noise
``delta_threshold`` has to absorb. Even two identical H100s in one node are not
provably bit-identical generators. If you turn this on, re-run
``leoma calibrate`` with it enabled — you are calibrating against the runtime
configuration you intend to use in production, not the one you happened to measure
before you turned this on. See ``docs/TESTNET_RUNBOOK.md``.

This module is a pure function of already-known inputs (feature flag, device count,
operator overrides) so the *decision* is unit-testable with no GPU. The one line that
needs an actual CUDA runtime — ``torch.cuda.device_count()`` — lives in the eval
server's tiny, untested glue, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DuelDevices:
    """Where each duelist should load. ``None`` means "today's default device"."""

    king_device: Optional[str]
    challenger_device: Optional[str]
    #: True only when generation will ACTUALLY overlap. False whenever the request
    #: for concurrency couldn't be honored (fewer than 2 devices, or a collision) —
    #: callers branch on this, not on whether concurrency was merely *requested*.
    concurrent: bool
    #: Human-readable explanation of the decision, for logs and the verdict's audit block.
    note: str


def resolve_duel_devices(
    *,
    concurrent_enabled: bool,
    cuda_device_count: int,
    king_device_override: Optional[str] = None,
    challenger_device_override: Optional[str] = None,
) -> DuelDevices:
    """Decide where king and challenger generate. Falls back to today's behavior
    whenever concurrency can't be honored safely, rather than guessing.

    ``king_device_override`` / ``challenger_device_override`` let an operator pin
    specific devices (e.g. to dodge a card another process is using, or on a box
    where ``torch.cuda.device_count()`` undercounts what's actually usable). When
    both are given explicitly, they are trusted even if the detected device count
    looks insufficient.
    """
    if not concurrent_enabled:
        return DuelDevices(
            None, None, False,
            "concurrent generation disabled; king and challenger share the default device",
        )

    both_overridden = bool(king_device_override and challenger_device_override)
    if cuda_device_count < 2 and not both_overridden:
        return DuelDevices(
            king_device_override, challenger_device_override, False,
            f"concurrent generation requested but only {cuda_device_count} CUDA device(s) "
            "visible (need >=2, or set both LEOMA_KING_DEVICE and LEOMA_CHALLENGER_DEVICE) "
            "— falling back to sequential single-device generation",
        )

    king = king_device_override or "cuda:0"
    challenger = challenger_device_override or "cuda:1"
    if king == challenger:
        return DuelDevices(
            king, challenger, False,
            f"king and challenger both resolved to {king!r} — concurrent generation would "
            "not overlap anything; falling back to sequential",
        )
    return DuelDevices(
        king, challenger, True,
        f"generating concurrently: king on {king}, challenger on {challenger}",
    )


__all__ = ["DuelDevices", "resolve_duel_devices"]
