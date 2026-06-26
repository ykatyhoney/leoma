"""Local scoring-window derivation (no owner-api ledger).

Instead of asking the owner-api ``GET /tasks/window``, each validator reconstructs the settled
window itself from the on-chain allowlist + the peer buckets it already reads:

  1. list every validator's bucket for the rotations it published a verdict for;
  2. resolve the canonical sampler per rotation (the primary, or the earliest failover backup that
     actually produced it) — so a validator can't stuff the window with rotations it wasn't
     scheduled for, and a duplicate (primary + backup) collapses to one deterministic owner;
  3. anchor to the shared epoch block and apply the same settle margin + window size as the server.

Every validator runs this over the same buckets at the same epoch block, so they all derive the
identical window — and therefore the identical winner — with no coordinator.
"""
from typing import Dict, List, Set, Tuple

from leoma.infra.scorer_constants import (
    SCORER_MAX_LOOKBACK_BLOCKS,
    SCORER_SETTLE_MARGIN,
    SCORER_TASK_WINDOW,
)


def resolve_canonical_samplers(
    produced_by_peer: Dict[str, Set[int]],
    validators: List[str],
) -> Dict[int, str]:
    """Map each produced rotation_id to its one legitimate sampler.

    For a rotation, the legitimate producer is the validator earliest in failover order
    (primary first, then backups) that actually published it. Producers not on any rotation
    position for that rotation_id are ignored (they weren't scheduled to sample it).
    """
    n = len(validators)
    if n == 0:
        return {}
    producers_by_rid: Dict[int, Set[str]] = {}
    for hotkey, rids in produced_by_peer.items():
        for r in rids:
            producers_by_rid.setdefault(int(r), set()).add(hotkey)

    canonical: Dict[int, str] = {}
    for rid, producers in producers_by_rid.items():
        for step in range(n):
            candidate = validators[(rid + step) % n]
            if candidate in producers:
                canonical[rid] = candidate
                break
    return canonical


def derive_window(
    canonical: Dict[int, str],
    epoch_block: int,
    interval: int,
    n: int = SCORER_TASK_WINDOW,
    margin: int = SCORER_SETTLE_MARGIN,
    max_lookback: int = SCORER_MAX_LOOKBACK_BLOCKS,
) -> Tuple[List[int], List[str]]:
    """The last ``n`` produced rotations at ``epoch_block``, dropping the newest ``margin``.

    Rotation ``r`` is anchored by its window's block ``r * interval``: included when
    ``r * interval <= epoch_block`` (and within ``max_lookback``). Returns ``(window_ids ascending,
    active_validators)`` — the distinct samplers in the window (denominator for the min-distinct gate).
    """
    step = max(1, interval)
    epoch_rid = epoch_block // step
    floor_rid = (epoch_block - max_lookback) // step if max_lookback and max_lookback > 0 else None

    rids = [
        r for r in canonical
        if r <= epoch_rid and (floor_rid is None or r >= floor_rid)
    ]
    rids.sort(reverse=True)
    selected = rids[max(0, margin): max(0, margin) + max(0, n)]
    selected.sort()
    active = sorted({canonical[r] for r in selected})
    return selected, active
