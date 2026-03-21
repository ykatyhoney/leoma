"""Shared helpers for stake-weighted task result aggregation."""

from typing import Any

from leoma.delivery.http.contracts import MinerTaskEntry, TaskDetailMinerEntry
from leoma.infra.stake_voting import stake_weighted_pass

def build_miner_task_entries(samples: list[Any], stake_map: dict[str, float]) -> list[MinerTaskEntry]:
    """Aggregate validator samples into per-task pass/fail entries for one miner."""
    by_task: dict[int, list[tuple[Any, float]]] = {}
    for sample in samples:
        by_task.setdefault(sample.task_id, []).append(
            (sample, stake_map.get(sample.validator_hotkey, 0.0))
        )

    entries = []
    for task_id, pairs in by_task.items():
        passed = stake_weighted_pass([(sample.passed, stake) for sample, stake in pairs])
        latency_ms = next(
            (getattr(sample, "latency_ms", None) for sample, _ in pairs if getattr(sample, "latency_ms", None) is not None),
            None,
        )
        updated = max((sample.evaluated_at for sample, _ in pairs if sample.evaluated_at), default=None)
        entries.append(
            MinerTaskEntry(
                task_id=task_id,
                passed=passed,
                latency_ms=latency_ms,
                updated=updated,
            )
        )

    entries.sort(key=lambda entry: (-entry.task_id,))
    return entries


def build_task_detail_entries(
    samples: list[Any],
    stake_map: dict[str, float],
) -> list[TaskDetailMinerEntry]:
    """Aggregate validator samples into per-miner task entries for one task."""
    by_miner: dict[str, list[tuple[Any, float]]] = {}
    for sample in samples:
        by_miner.setdefault(sample.miner_hotkey, []).append(
            (sample, stake_map.get(sample.validator_hotkey, 0.0))
        )

    entries = []
    for miner_hotkey, pairs in by_miner.items():
        passed = stake_weighted_pass([(sample.passed, stake) for sample, stake in pairs])
        latency_ms = next(
            (
                getattr(sample, "latency_ms", None)
                for sample, _ in pairs
                if getattr(sample, "latency_ms", None) is not None
            ),
            None,
        )
        updated = max((sample.evaluated_at for sample, _ in pairs if sample.evaluated_at), default=None)
        entries.append(
            TaskDetailMinerEntry(
                miner_hotkey=miner_hotkey,
                passed=passed,
                validator_count=len(pairs),
                latency_ms=latency_ms,
                updated=updated,
            )
        )

    entries.sort(key=lambda entry: (not entry.passed, entry.miner_hotkey))
    return entries
