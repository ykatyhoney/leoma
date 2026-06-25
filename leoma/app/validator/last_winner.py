"""Persisted 'last computed winner' for weight-setting continuity.

When the scoring window can't be fetched at an epoch because the owner-api is unreachable, the
validator repeats the last winner it successfully computed instead of burning alpha. Persisted to a
small JSON file (under ``LEOMA_STATE_DIR``, default ``~/.leoma``) so it survives a validator restart.

Determinism holds: the scoring window is a deterministic function of the epoch block + ledger, so
every validator's last successful winner is the same value — during an owner-api outage they all
repeat the same winner and stay aligned, rather than each burning the epoch.
"""
import json
import os
from typing import Optional, Tuple

from leoma.bootstrap import emit_log as log


def _path() -> str:
    state_dir = os.environ.get("LEOMA_STATE_DIR", os.path.expanduser("~/.leoma"))
    return os.path.join(state_dir, "last_winner.json")


def load_last_winner() -> Optional[Tuple[int, str]]:
    """Return ``(uid, hotkey)`` of the last persisted winner, or ``None`` if absent/unreadable."""
    try:
        with open(_path()) as f:
            d = json.load(f)
        uid = int(d["uid"])
        hotkey = d["hotkey"]
        if uid > 0 and hotkey:
            return uid, hotkey
    except (FileNotFoundError, KeyError, ValueError, TypeError, OSError):
        return None
    return None


def save_last_winner(uid: int, hotkey: str, epoch_block: Optional[int] = None) -> None:
    """Persist the winner atomically. Best-effort: a write failure is logged, never raised."""
    try:
        path = _path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump({"uid": int(uid), "hotkey": hotkey, "epoch_block": epoch_block}, f)
        os.replace(tmp, path)  # atomic swap
    except OSError as e:
        log(f"Could not persist last winner: {e}", "warn")
