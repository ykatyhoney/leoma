"""Typed duel errors — who is at fault decides what happens next.

The validator's failure policy (``app/validator/failures.py``) turns an exception
into retry / quarantine / abort. That decision is only as good as the fault
attribution behind it, so the eval path raises errors that *say whose fault it
is* rather than a bare ``Exception`` the validator has to guess at from a string.

Three faults, three very different consequences:

* :class:`ChallengerFault` — the *submitted model* is bad (missing repo, wrong
  architecture, degenerate output). Quarantine the artifact; the subnet moves on.
* :class:`TransientDuelError` — the *environment* hiccuped (a corpus read failed,
  the GPU box died). Retry with backoff. **Never** blame the miner for this.
* :class:`CorpusIntegrityError` / :class:`ConsensusConfigError` — *this validator*
  is wrong: its corpus does not match the pinned manifest, or its config does not
  match the chain. Both are **fail-closed**: refuse to duel rather than emit a
  verdict that other validators cannot reproduce. Blaming a miner for our own bad
  corpus would be the worst possible outcome.
"""
from __future__ import annotations


class DuelError(RuntimeError):
    """Base class for every duel-path failure."""

    #: Stable machine-readable code, surfaced in the verdict and the dashboard.
    reason = "duel_error"


class TransientDuelError(DuelError):
    """Something outside the model failed and may well work next time."""

    reason = "transient"


class CorpusIntegrityError(DuelError):
    """The corpus this box fetched does not match the pinned manifest.

    Fail-closed: a validator whose ground truth differs from everyone else's would
    produce distances nobody can reproduce. Refuse rather than diverge silently.
    """

    reason = "corpus_integrity"


class ConsensusConfigError(DuelError):
    """The consensus surface disagrees — pinned config vs. what we were handed.

    Raised when the request's ``consensus_digest`` doesn't match ours, when the
    manifest's decode parameters contradict the pinned generation parameters, or
    when a required field is missing. Never guess a default: a silently defaulted
    field is exactly how two honest validators end up with different verdicts.
    """

    reason = "consensus_config"


class ChallengerFault(DuelError):
    """The challenger's model is the problem."""

    reason = "challenger_fault"


class DegenerateGeneration(ChallengerFault):
    """The model emitted output that cannot be scored (too short, NaN, empty).

    A separate class because it is the *cheat* path, not the *broken* path: it is
    what a model trying to game a reference-distance metric produces.
    """

    reason = "challenger_degenerate"


__all__ = [
    "DuelError",
    "TransientDuelError",
    "CorpusIntegrityError",
    "ConsensusConfigError",
    "ChallengerFault",
    "DegenerateGeneration",
]
