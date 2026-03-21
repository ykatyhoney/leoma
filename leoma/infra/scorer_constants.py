"""Shared stake-weighted scorer settings (API eligibility must match scorer logic)."""

import os

# Consecutive task_id window: [max_task_id - N + 1, max_task_id] inclusive (N IDs).
SCORER_TASK_WINDOW = int(os.environ.get("SCORER_TASK_WINDOW", "100"))

# Fraction of window tasks a miner must have been evaluated on to rank (e.g. 0.8 = 80%).
# Window = consecutive task_ids [max_id - N + 1, max_id]; completeness = evaluated_in_window / N.
COMPLETENESS_ELIGIBILITY_THRESHOLD = float(
    os.environ.get("SCORER_COMPLETENESS_THRESHOLD", "0.8")
)
