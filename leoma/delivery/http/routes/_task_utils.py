"""Shared helpers for per-task results."""

import json
from typing import Any, Optional

from leoma.delivery.http.contracts import MinerTaskEntry, TaskDetailMinerEntry


def parse_aspect_and_overall(
    gen_art: Optional[str],
) -> tuple[Optional[dict[str, int]], Optional[int]]:
    """Extract ``(aspect_scores, overall_score)`` from a sample's ``generated_artifacts`` JSON.

    The validator's evaluator embeds the evaluation scores there (see app/evaluator/main.py).
    ``overall_score`` is the 0-100 total that drives pass/fail; returns ``None`` if unparseable.
    """
    if not gen_art:
        return None, None
    try:
        data = json.loads(gen_art)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, None
    raw_aspects = data.get("aspect_scores")
    aspect_scores = None
    if isinstance(raw_aspects, dict):
        aspect_scores = {k: int(v) for k, v in raw_aspects.items() if isinstance(v, (int, float))}
    overall = data.get("overall_score")
    overall_score = int(overall) if isinstance(overall, (int, float)) else None
    return aspect_scores, overall_score


def representative_sample(samples: list[Any]) -> Optional[Any]:
    """The sample row for a (task, miner); the most-recent if several rows exist."""
    if not samples:
        return None
    evaluated = [s for s in samples if s.evaluated_at is not None]
    return max(evaluated, key=lambda s: s.evaluated_at) if evaluated else samples[0]


def build_miner_task_entries(samples: list[Any]) -> list[MinerTaskEntry]:
    """Build a miner's per-task pass/fail entries from its samples."""
    by_task: dict[int, list[Any]] = {}
    for sample in samples:
        by_task.setdefault(sample.task_id, []).append(sample)

    entries = []
    for task_id, group in by_task.items():
        rep = representative_sample(group)
        latency_ms = next(
            (getattr(s, "latency_ms", None) for s in group if getattr(s, "latency_ms", None) is not None),
            None,
        )
        _, overall_score = parse_aspect_and_overall(getattr(rep, "generated_artifacts", None))
        entries.append(
            MinerTaskEntry(
                task_id=task_id,
                passed=bool(rep.passed),
                validator_hotkey=rep.validator_hotkey,
                overall_score=overall_score,
                latency_ms=latency_ms,
                updated=rep.evaluated_at,
            )
        )

    entries.sort(key=lambda entry: -entry.task_id)
    return entries


def build_task_detail_entries(samples: list[Any]) -> list[TaskDetailMinerEntry]:
    """Build one per-miner entry for a single task."""
    by_miner: dict[str, list[Any]] = {}
    for sample in samples:
        by_miner.setdefault(sample.miner_hotkey, []).append(sample)

    entries = []
    for miner_hotkey, group in by_miner.items():
        rep = representative_sample(group)
        latency_ms = next(
            (getattr(s, "latency_ms", None) for s in group if getattr(s, "latency_ms", None) is not None),
            None,
        )
        entries.append(
            TaskDetailMinerEntry(
                miner_hotkey=miner_hotkey,
                passed=bool(rep.passed),
                latency_ms=latency_ms,
                updated=rep.evaluated_at,
            )
        )

    entries.sort(key=lambda entry: (not entry.passed, entry.miner_hotkey))
    return entries
