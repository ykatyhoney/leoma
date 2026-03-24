"""Shared stake-weighted scorer settings (API eligibility must match scorer logic)."""

import os
from typing import List, Optional

# Consecutive task_id window: [max_task_id - N + 1, max_task_id] inclusive (N IDs).
SCORER_TASK_WINDOW = int(os.environ.get("SCORER_TASK_WINDOW", "100"))


def scoring_window_task_ids(max_task_id: Optional[int]) -> Optional[List[int]]:
    """Task IDs in the current scoring window (aligned with stake-weighted scorer).

    When ``max_task_id >= SCORER_TASK_WINDOW``, returns the last ``N`` consecutive IDs:
    ``range(max_task_id - N + 1, max_task_id + 1)``.

    When fewer than ``N`` tasks exist, returns ``range(1, max_task_id + 1)`` (all tasks
    observed so far). Returns ``None`` if ``max_task_id`` is ``None``.
    """
    if max_task_id is None:
        return None
    if max_task_id < SCORER_TASK_WINDOW:
        return list(range(1, max_task_id + 1))
    return list(range(max_task_id - SCORER_TASK_WINDOW + 1, max_task_id + 1))

# Fraction of window tasks a miner must have been evaluated on to rank (e.g. 0.8 = 80%).
# Window = consecutive task_ids [max_id - N + 1, max_id]; completeness = evaluated_in_window / N.
COMPLETENESS_ELIGIBILITY_THRESHOLD = float(
    os.environ.get("SCORER_COMPLETENESS_THRESHOLD", "0.8")
)
