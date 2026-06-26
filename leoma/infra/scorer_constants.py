"""Shared scoring-window settings.

The window is the last N *produced* tasks (derived from peer-bucket producedness by validators, and
from the dual-reported ``validator_samples`` for the dashboard) — not a block-number range — so a
skipped rotation turn doesn't dilute it. These constants make every validator size + settle the
window identically.
"""

import math
import os

# Number of produced tasks in the scoring window (the last N produced/distinct task_ids).
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
