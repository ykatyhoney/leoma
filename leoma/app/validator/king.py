"""King-of-the-hill state transitions and weight targets (pure logic).

A single reigning **king** model holds the crown; a challenger that wins the
duel is crowned and the deposed king slides onto a bounded **king chain** of
recent prior champions. Emission is split *equally* across the current king plus
the prior kings still on the metagraph — else burned to UID 0.

These are pure functions over plain dicts so the consensus-critical logic is
fully unit-testable without a chain or GPU. Persistence lives in ``state_store``
and the chain calls live in ``validator.main``.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable, Optional

# Current king + up to (KING_CHAIN_SIZE - 1) prior kings share emission.
KING_CHAIN_SIZE = int(os.environ.get("LEOMA_KING_CHAIN_SIZE", "5"))
# UID that receives burned emission when there is no registered king.
BURN_UID = int(os.environ.get("LEOMA_BURN_UID", "0"))
# Blocks between weight refreshes (also forced on dethrone / startup).
WEIGHT_INTERVAL = int(os.environ.get("LEOMA_WEIGHT_INTERVAL", "300"))

SEED_CHALLENGE_ID = "seed"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def crown(
    king: Optional[dict],
    king_chain: list[dict],
    *,
    hotkey: str,
    model_repo: str,
    model_digest: str,
    block: int,
    challenge_id: str,
    crowned_at: Optional[str] = None,
    king_chain_size: int = KING_CHAIN_SIZE,
) -> tuple[dict, list[dict]]:
    """Crown a new king; return ``(new_king, new_king_chain)``.

    A genuine dethrone (``challenge_id != "seed"``) pushes the deposed king to
    the front of the chain and bumps the reign number; seeding the genesis king
    does neither. The input ``king``/``king_chain`` are not mutated.
    """
    prev_repo = (king or {}).get("model_repo", "") if king else ""
    is_seed = challenge_id == SEED_CHALLENGE_ID

    new_chain = list(king_chain or [])
    if king and not is_seed:
        new_chain.insert(0, dict(king))
        new_chain = new_chain[: max(0, king_chain_size - 1)]

    reign = (king or {}).get("reign_number", 0) if king else 0
    reign = reign + (0 if is_seed else 1)

    new_king = {
        "hotkey": hotkey,
        "model_repo": model_repo,
        "model_digest": model_digest,
        "reign_number": reign,
        "crowned_at": crowned_at or _now_iso(),
        "crowned_block": int(block),
        "challenge_id": challenge_id,
        "previous_repo": prev_repo,
    }
    return new_king, new_chain


def king_hotkeys(king: Optional[dict], king_chain: list[dict]) -> list[str]:
    """Distinct hotkeys sharing emission: current king first, then prior kings."""
    out: list[str] = []
    hk = (king or {}).get("hotkey", "") if king else ""
    if hk:
        out.append(hk)
    for entry in king_chain or []:
        h = entry.get("hotkey", "")
        if h and h not in out:
            out.append(h)
    return out


def weight_targets(
    king: Optional[dict],
    king_chain: list[dict],
    uid_map: dict[str, int],
    *,
    burn_uid: int = BURN_UID,
) -> tuple[list[int], list[float], str]:
    """Build ``(uids, weights, label)`` for ``set_weights``.

    Equal share across every king hotkey currently on the metagraph; if none are
    registered (or there is no king), burn 100% to ``burn_uid``.
    """
    hks = king_hotkeys(king, king_chain)
    target_uids = [int(uid_map[hk]) for hk in hks if hk in uid_map]

    if not target_uids:
        return [burn_uid], [1.0], f"burn:uid={burn_uid}"

    w = round(1.0 / len(target_uids), 9)
    label = (king or {}).get("hotkey", "") or "multi"
    return target_uids, [w] * len(target_uids), label
