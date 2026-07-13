"""How often a hotkey may occupy the subnet's only GPU.

The seen-set (``hotkey|digest``) is an **idempotency** gate: it stops the same
artifact being dueled twice. It is not a **cost** gate, and the difference matters —
a hotkey can mint an unlimited number of fresh digests (change one byte, re-upload),
and each new digest is a brand-new key that buys a **free multi-hour duel** on the
only GPU in the subnet. Nothing stops one miner from starving everyone else, and it
costs them nothing but an upload.

The reference subnet's answer (burn the miner's slot) does not transfer: their eval
is cheap. Ours costs *hours*, so the limiter is tuned to the thing that is actually
scarce here — GPU time, not slots.

Three rules, all pure functions of state the validator already has:

* **Cooldown** — after any verdict, a hotkey waits ``COOLDOWN_BLOCKS`` before it can
  duel again. Bounds how fast one miner can consume the GPU.
* **Per-reign cap** — at most ``MAX_CHALLENGES_PER_REIGN`` duels against a given king.
  Keyed on ``reign_number``, which is already deterministic and chain-derived, so
  validators agree on it without coordinating.
* **Reign refresh** — the per-reign counter resets every ``REIGN_REFRESH_BLOCKS``.
  Without it a durable king would eventually lock every miner out forever, and the
  subnet would freeze with no way to challenge the incumbent.

**Strikes are only for gate rejections, never for losing.** Losing a duel fairly is
the system working; a miner who submits an honest model that isn't good enough has
done nothing wrong and must not be penalized for trying. Only rejections that mean
"you should never have been dispatched" (wrong architecture, a copy of the king,
degenerate output) count against a hotkey.
"""
from __future__ import annotations

import os
from typing import Optional

#: ~72 minutes at 12s blocks. A duel costs hours of GPU; this bounds how fast one
#: hotkey can queue the next one.
COOLDOWN_BLOCKS = int(os.environ.get("LEOMA_COOLDOWN_BLOCKS", "360"))

#: Duels a single hotkey may have against ONE king.
MAX_CHALLENGES_PER_REIGN = int(os.environ.get("LEOMA_MAX_CHALLENGES_PER_REIGN", "3"))

#: ~7 days at 12s blocks. The per-reign counter resets on this cadence so a long
#: reign never becomes a permanent lockout.
REIGN_REFRESH_BLOCKS = int(os.environ.get("LEOMA_REIGN_REFRESH_BLOCKS", "50400"))

#: Gate rejections tolerated before a hotkey is blacklisted at scan time.
MAX_STRIKES = int(os.environ.get("LEOMA_MAX_STRIKES", "5"))

#: Rejections that are the miner's fault and mean "this should never have been
#: dispatched". Losing a duel is NOT one of them.
STRIKEABLE = frozenset({
    "copy_of_king",
    "config_rejected",
    "arch_mismatch",
    "challenger_degenerate",
    "model_invalid",
})


def _row(duels: dict, hotkey: str) -> dict:
    return duels.setdefault(hotkey, {
        "last_verdict_block": 0,
        "reign_number": 0,
        "reign_count": 0,
        "reign_started_block": 0,
        "strikes": 0,
    })


def reign_of(king: Optional[dict]) -> int:
    """The reigning king's number — deterministic and chain-derived, so validators agree."""
    return int((king or {}).get("reign_number", 0) or 0)


def check(duels: dict, hotkey: str, *, king: Optional[dict], block: int) -> Optional[str]:
    """May this hotkey duel right now? Returns None if yes, else why not.

    A pure function: same state in, same answer out, so it is trivially testable and
    cannot drift between validators (though it does not need to — a rate limit is
    local liveness policy, not consensus).
    """
    row = duels.get(hotkey)
    if not row:
        return None

    since = block - int(row.get("last_verdict_block", 0) or 0)
    if since < COOLDOWN_BLOCKS:
        return f"cooldown: {COOLDOWN_BLOCKS - since} more blocks"

    reign = reign_of(king)
    if int(row.get("reign_number", 0)) == reign:
        started = int(row.get("reign_started_block", 0) or 0)
        # A long reign must not become a permanent lockout: refresh the allowance.
        if block - started < REIGN_REFRESH_BLOCKS:
            if int(row.get("reign_count", 0)) >= MAX_CHALLENGES_PER_REIGN:
                return (
                    f"reign cap: {MAX_CHALLENGES_PER_REIGN} challenges already spent against "
                    f"king #{reign} (resets in {REIGN_REFRESH_BLOCKS - (block - started)} blocks)"
                )

    return None


def record_verdict(duels: dict, hotkey: str, *, king: Optional[dict], block: int) -> dict:
    """Charge a completed duel to the hotkey's budget."""
    row = _row(duels, hotkey)
    reign = reign_of(king)

    started = int(row.get("reign_started_block", 0) or 0)
    refreshed = block - started >= REIGN_REFRESH_BLOCKS

    if int(row.get("reign_number", 0)) != reign or refreshed:
        row["reign_number"] = reign
        row["reign_count"] = 0
        row["reign_started_block"] = block

    row["reign_count"] = int(row.get("reign_count", 0)) + 1
    row["last_verdict_block"] = block
    return row


def record_strike(duels: dict, hotkey: str, reason: str) -> dict:
    """A gate rejection. Losing a duel fairly is NOT a strike — see the module docstring."""
    row = _row(duels, hotkey)
    if reason in STRIKEABLE:
        row["strikes"] = int(row.get("strikes", 0)) + 1
    return row


def struck_out(duels: dict) -> set[str]:
    """Hotkeys that have exhausted their strikes — dropped at scan time."""
    return {
        hotkey for hotkey, row in duels.items()
        if int(row.get("strikes", 0)) >= MAX_STRIKES
    }


__all__ = [
    "COOLDOWN_BLOCKS",
    "MAX_CHALLENGES_PER_REIGN",
    "MAX_STRIKES",
    "REIGN_REFRESH_BLOCKS",
    "STRIKEABLE",
    "check",
    "record_strike",
    "record_verdict",
    "reign_of",
    "struck_out",
]
