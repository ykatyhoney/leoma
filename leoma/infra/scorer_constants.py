"""Shared scoring-window settings (block-derived, so all validators agree by construction)."""

import math
import os
from typing import List, Optional

# Consecutive task_id window length (N IDs ending at the settled window end).
SCORER_TASK_WINDOW = int(os.environ.get("SCORER_TASK_WINDOW", "100"))

# Hard cap on how far back (in blocks) the production-based window may reach. Bounds the wall-clock
# span the "last N produced tasks" window can cover when the network samples slowly, so a long idle
# stretch can't drag ancient performance into scoring. 0 disables the cap. Default ~ 3x the nominal
# 100-task span at the 100-block interval.
SCORER_MAX_LOOKBACK_BLOCKS = int(os.environ.get("SCORER_MAX_LOOKBACK_BLOCKS", "30000"))

# Settle margin: exclude the most-recent M rotation indices from scoring. task_id = block // interval,
# and a sampler may still be writing the current (or, rarely, the previous) window's verdicts. Scoring
# only tasks up to `block // interval - M` guarantees every scored task is fully settled and immutable,
# so two validators reading at slightly different instants always see identical inputs. Default 2 covers
# a sampler that overruns its ~20-min turn.
SCORER_SETTLE_MARGIN = int(os.environ.get("SCORER_SETTLE_MARGIN", "2"))


def settled_window_end(block: int, interval: int, margin: Optional[int] = None) -> int:
    """The most-recent SETTLED rotation index at ``block``: ``floor(block / interval) - margin``.

    Derived purely from the consensus block + interval, so every validator that runs an epoch at
    the same block computes the same value (no dependence on a mutable announced ``latest_task_id``).
    """
    if margin is None:
        margin = SCORER_SETTLE_MARGIN
    if interval <= 0:
        interval = 1
    return (block // interval) - margin


def window_task_ids_ending_at(window_end: Optional[int]) -> Optional[List[int]]:
    """The ``SCORER_TASK_WINDOW`` consecutive task_ids ending at ``window_end`` (inclusive).

    Returns ``None`` when ``window_end`` is ``None`` or negative (subnet too young to have a settled
    window). The lower bound is clamped to 0 so an early subnet yields a shorter window rather than
    negative ids.
    """
    if window_end is None or window_end < 0:
        return None
    start = max(0, window_end - SCORER_TASK_WINDOW + 1)
    return list(range(start, window_end + 1))


def scoring_window_task_ids(max_task_id: Optional[int]) -> Optional[List[int]]:
    """Legacy helper: window ending at a known ``max_task_id`` (no settle margin).

    Prefer ``settled_window_end`` + ``window_task_ids_ending_at`` (block-derived). Kept for any
    caller that already has a concrete max task id.
    """
    if max_task_id is None:
        return None
    if max_task_id < SCORER_TASK_WINDOW:
        return list(range(1, max_task_id + 1))
    return list(range(max_task_id - SCORER_TASK_WINDOW + 1, max_task_id + 1))


# Fraction of the window's existing tasks a miner must have been evaluated on to rank (e.g. 0.8 = 80%).
COMPLETENESS_ELIGIBILITY_THRESHOLD = float(
    os.environ.get("SCORER_COMPLETENESS_THRESHOLD", "0.8")
)

# Minimum fraction of *active* validators (those that produced >=1 task in the window) that must have
# evaluated a miner for it to be eligible/win. Guards winner-take-all from being decided by too few
# validators in a skip-thinned window. 0.5 -> a majority of active validators.
MIN_DISTINCT_VALIDATORS_RATIO = float(
    os.environ.get("SCORER_MIN_DISTINCT_VALIDATORS_RATIO", "0.5")
)


def required_distinct_validators(active_count: int) -> int:
    """How many distinct validators must have evaluated a miner, given the active-validator count.

    ``ceil(active * ratio)``, floored at 1 so a window with a single live validator never deadlocks
    (and so the gate is a no-op until there are >=2 active validators).
    """
    if active_count <= 0:
        return 1
    return max(1, math.ceil(active_count * MIN_DISTINCT_VALIDATORS_RATIO))
