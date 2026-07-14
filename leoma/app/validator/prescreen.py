"""Reject a bad model for ~5 seconds of config fetch, not hours of GPU.

The head-of-line fix (``continue`` instead of ``return``) stopped one bad challenger
*blocking* the others. It did not stop it being **expensive**: a 404 repo, or one
whose weights are the wrong architecture, still consumed a full dispatch — the eval
box downloads tens of gigabytes, loads two 14B pipelines, and only then discovers the
model was never loadable. Meanwhile the GPU is locked and every honest challenger
waits.

The prescreen is the actual mitigation. It fetches **configs only** (~200 KB, no
weights) and runs the architecture lock against them, on the *validator*, before the
duel is ever dispatched. A wrong-architecture model costs seconds; a missing repo
costs a 404.

It also finally consumes three hooks that have been dead since they were written:
``materialize_model(config_only=True)`` — whose docstring literally says *"use for the
validator's per-challenger arch/lock validation"* — plus ``EXTRA_LOCK_KEYS`` and
``ARCH_BASE_REPO``.

**Fail-open on infrastructure, fail-closed on architecture.** If the prescreen itself
cannot run — the Hub is down, our token is wrong — that is *our* problem, and it must
not be charged to the miner. Only a config we successfully read and found to be the
wrong architecture is a rejection.
"""
from __future__ import annotations

from typing import Optional

from leoma.bootstrap import emit_log as log
from leoma.eval.arch_lock import ArchMismatch, validate
from leoma.eval.errors import TransientDuelError
from leoma.infra.chain_config import ARCH_BASE_REPO, EXTRA_LOCK_KEYS, SPEC
from leoma.infra.model_store import ModelRef, materialize_model


def _base_ref() -> Optional[ModelRef]:
    """The pinned base architecture, as a fetchable ref.

    ``base_repo`` is an HF repo id with no digest — it is the *definition* of the
    architecture, not a competitor in the duel, so it is pinned by name and fetched at
    its default revision. The configs it yields are what every challenger is diffed
    against.
    """
    if not ARCH_BASE_REPO:
        return None
    return ModelRef(ARCH_BASE_REPO, "hf:" + "0" * 40)


def prescreen(repo: str, digest: str, *, base_dir: Optional[str] = None) -> dict:
    """Validate a challenger's architecture from its configs alone.

    Raises :class:`~leoma.eval.arch_lock.ArchMismatch` (a ``ChallengerFault`` ⇒
    PERMANENT ⇒ quarantined) when the model is not the pinned architecture, and
    :class:`~leoma.eval.errors.TransientDuelError` when *we* could not check.
    """
    ref = ModelRef(repo, digest)

    try:
        config_dir = materialize_model(ref, config_only=True)
    except Exception as e:  # noqa: BLE001
        # A 404 / invalid-repo error is the miner's fault and will be classified as
        # PERMANENT by the caller. Anything else (auth, network) is ours. We do not
        # try to tell them apart here — `failures.classify` already does, and doing it
        # in two places is how the two places drift apart.
        raise e

    if base_dir is None:
        base = _base_ref()
        if base is None:
            raise TransientDuelError(
                "chain.toml [arch].base_repo is not set, so there is nothing to lock the "
                "challenger's architecture against"
            )
        try:
            base_dir = materialize_model(base, config_only=True)
        except Exception as e:  # noqa: BLE001 — the BASE failing to fetch is OUR problem
            raise TransientDuelError(
                f"could not fetch the pinned base architecture's configs ({ARCH_BASE_REPO}): {e}"
            ) from e

    return validate(
        config_dir,
        base_dir,
        pipeline=SPEC.arch.pipeline,
        extra_keys=tuple(EXTRA_LOCK_KEYS),
        # Size is checked at download time, not here: a config-only fetch has no
        # weights in it, so there is nothing to measure yet.
        total_bytes=None,
    )


def prescreen_or_reason(repo: str, digest: str) -> Optional[Exception]:
    """Prescreen, returning the exception rather than raising. Never raises itself."""
    try:
        prescreen(repo, digest)
        return None
    except ArchMismatch as e:
        log(f"Prescreen REJECTED {repo}@{digest[:19]}: {e}", "warn")
        return e
    except Exception as e:  # noqa: BLE001
        return e


__all__ = ["prescreen", "prescreen_or_reason"]
