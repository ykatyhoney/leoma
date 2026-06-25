"""
Pure dashboard aggregation helpers (no DB/IO) — easy to unit-test.

Self-evaluation model: each task in ``validator_samples`` was sampled AND evaluated by exactly
one validator (its sampler), so a sample's ``validator_hotkey`` is that task's sampler. These
helpers derive per-validator participation (did it sample on its rotation turns?) and per-miner
activity from a flat list of samples for a scoring window.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class ValidatorParticipation:
    validator_hotkey: str
    tasks_sampled: int          # distinct tasks this validator sampled+evaluated in the window
    evaluations: int            # total evaluation rows it produced
    expected_turns: int         # rotation turns assigned to it in the window
    participation_rate: float   # tasks_sampled / expected_turns (capped at 1.0)
    last_task_id: Optional[int]
    last_evaluated_at: Optional[Any]
    avg_latency_ms: Optional[int]


@dataclass
class MinerActivity:
    miner_hotkey: str
    tasks_evaluated: int
    validators_evaluating: int
    passed_tasks: int
    last_evaluated_at: Optional[Any]
    avg_latency_ms: Optional[int]


def expected_turns_for(
    validator_hotkey: str, ordered: List[str], window_task_ids: List[int]
) -> int:
    """How many window rotation indices map to this validator (``ordered[task_id % N]``)."""
    if not ordered:
        return 0
    n = len(ordered)
    return sum(1 for t in window_task_ids if ordered[t % n] == validator_hotkey)


def compute_validator_participation(
    samples: List[Any],
    ordered: List[str],
    window_task_ids: List[int],
) -> Dict[str, ValidatorParticipation]:
    """Per-validator participation over the window (includes ordered validators with 0 samples)."""
    by_v: Dict[str, dict] = {}
    for s in samples:
        v = s.validator_hotkey
        d = by_v.setdefault(
            v, {"tasks": set(), "evals": 0, "last_task": None, "last_at": None, "lat_sum": 0, "lat_n": 0}
        )
        d["tasks"].add(s.task_id)
        d["evals"] += 1
        tid = s.task_id if s.task_id is not None else -1
        if d["last_task"] is None or tid > d["last_task"]:
            d["last_task"] = s.task_id
        ev = getattr(s, "evaluated_at", None)
        if ev is not None and (d["last_at"] is None or ev > d["last_at"]):
            d["last_at"] = ev
        lat = getattr(s, "latency_ms", None)
        if lat is not None:
            d["lat_sum"] += lat
            d["lat_n"] += 1

    out: Dict[str, ValidatorParticipation] = {}
    for v in set(by_v.keys()) | set(ordered):
        d = by_v.get(v)
        expected = expected_turns_for(v, ordered, window_task_ids)
        tasks_sampled = len(d["tasks"]) if d else 0
        if expected:
            rate = min(1.0, tasks_sampled / expected)
        else:
            rate = 1.0 if tasks_sampled else 0.0
        out[v] = ValidatorParticipation(
            validator_hotkey=v,
            tasks_sampled=tasks_sampled,
            evaluations=d["evals"] if d else 0,
            expected_turns=expected,
            participation_rate=rate,
            last_task_id=d["last_task"] if d else None,
            last_evaluated_at=d["last_at"] if d else None,
            avg_latency_ms=int(d["lat_sum"] / d["lat_n"]) if d and d["lat_n"] else None,
        )
    return out


def compute_miner_activity(samples: List[Any]) -> Dict[str, MinerActivity]:
    """Per-miner activity (distinct tasks, distinct validators, passes, recency) over the window."""
    by_m: Dict[str, dict] = {}
    for s in samples:
        m = s.miner_hotkey
        d = by_m.setdefault(
            m, {"tasks": set(), "vals": set(), "passed": 0, "last_at": None, "lat_sum": 0, "lat_n": 0}
        )
        d["tasks"].add(s.task_id)
        d["vals"].add(s.validator_hotkey)
        if s.passed:
            d["passed"] += 1
        ev = getattr(s, "evaluated_at", None)
        if ev is not None and (d["last_at"] is None or ev > d["last_at"]):
            d["last_at"] = ev
        lat = getattr(s, "latency_ms", None)
        if lat is not None:
            d["lat_sum"] += lat
            d["lat_n"] += 1
    return {
        m: MinerActivity(
            miner_hotkey=m,
            tasks_evaluated=len(d["tasks"]),
            validators_evaluating=len(d["vals"]),
            passed_tasks=d["passed"],
            last_evaluated_at=d["last_at"],
            avg_latency_ms=int(d["lat_sum"] / d["lat_n"]) if d["lat_n"] else None,
        )
        for m, d in by_m.items()
    }
