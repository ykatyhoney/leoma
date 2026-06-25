"""
Per-validator-average score aggregation (decentralized consensus, self-evaluation model).

Each task is sampled AND evaluated by exactly one validator (no cross-validation), so there
is one verdict per (validator, task, miner). To give every validator equal weight regardless
of how many tasks it sampled:

  1. For each validator V and miner M, compute V's pass-rate for M over V's own tasks.
  2. A miner's score is the MEAN of those per-validator rates (each validator counts once).
  3. Rank eligible miners with the existing dominance algorithm on that mean rate.

Shared by the validator's local weight-setter (verdicts read from peer buckets) and the
owner-api dashboard scorer (verdicts read from the DB). Both use this same method, so the
dashboard mirrors the on-chain ranking; they can differ slightly when the dashboard's
dual-reported DB lags the authoritative buckets (the on-chain result is authoritative).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from leoma.infra.rank import compute_rank_from_miner_stats
from leoma.infra.scorer_constants import COMPLETENESS_ELIGIBILITY_THRESHOLD

# (validator_hotkey, task_id, miner_hotkey) -> passed
Verdicts = Dict[Tuple[str, int, str], bool]


@dataclass
class MinerAggregate:
    """Per-miner aggregate over the scoring window."""

    miner_hotkey: str
    avg_rate: float                       # mean of per-validator pass-rates (equal weight)
    total_passed: int                     # total passed tasks across all validators
    total_evaluated: int                  # total evaluated tasks across all validators
    completeness: float                   # distinct tasks evaluated / tasks that exist in window
    per_validator_rate: Dict[str, float] = field(default_factory=dict)
    block: Optional[int] = None


def compute_miner_aggregates(
    verdicts: Verdicts,
    window_task_ids: List[int],
    block_by_hotkey: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, MinerAggregate]:
    """Build per-miner aggregates from per-(validator, task, miner) verdicts in the window.

    No miner/eligibility filtering here — returns every miner seen, so callers can decide
    eligibility (and populate dashboard tables). Completeness uses the number of tasks that
    actually exist in the window (skipped rotation turns produce no task), not the nominal size.
    """
    block_by_hotkey = block_by_hotkey or {}
    window_set = set(window_task_ids)

    existing_tasks: Set[int] = set()
    pv_eval: Dict[Tuple[str, str], int] = {}   # (validator, miner) -> evaluated count
    pv_pass: Dict[Tuple[str, str], int] = {}   # (validator, miner) -> passed count
    miner_tasks: Dict[str, Set[int]] = {}      # miner -> distinct task_ids evaluated
    total_eval: Dict[str, int] = {}
    total_pass: Dict[str, int] = {}

    for (validator, task_id, miner), passed in verdicts.items():
        if task_id not in window_set:
            continue
        existing_tasks.add(task_id)
        pv_eval[(validator, miner)] = pv_eval.get((validator, miner), 0) + 1
        total_eval[miner] = total_eval.get(miner, 0) + 1
        miner_tasks.setdefault(miner, set()).add(task_id)
        if passed:
            pv_pass[(validator, miner)] = pv_pass.get((validator, miner), 0) + 1
            total_pass[miner] = total_pass.get(miner, 0) + 1

    n_existing = len(existing_tasks)

    rates_by_miner: Dict[str, Dict[str, float]] = {}
    for (validator, miner), evaluated in pv_eval.items():
        if evaluated <= 0:
            continue
        rate = pv_pass.get((validator, miner), 0) / evaluated
        rates_by_miner.setdefault(miner, {})[validator] = rate

    aggregates: Dict[str, MinerAggregate] = {}
    for miner, per_validator in rates_by_miner.items():
        rates = list(per_validator.values())
        avg_rate = sum(rates) / len(rates) if rates else 0.0
        completeness = (len(miner_tasks.get(miner, set())) / n_existing) if n_existing else 0.0
        aggregates[miner] = MinerAggregate(
            miner_hotkey=miner,
            avg_rate=avg_rate,
            total_passed=total_pass.get(miner, 0),
            total_evaluated=total_eval.get(miner, 0),
            completeness=completeness,
            per_validator_rate=per_validator,
            block=block_by_hotkey.get(miner),
        )
    return aggregates


def rank_from_aggregates(
    aggregates: Dict[str, MinerAggregate],
    valid_miners: Set[str],
    dominance_threshold: float,
    completeness_threshold: float = COMPLETENESS_ELIGIBILITY_THRESHOLD,
    min_distinct_validators: int = 1,
) -> Tuple[Optional[str], List[dict]]:
    """Rank eligible miners by the per-validator-average rate using the dominance algorithm.

    Eligible = valid (if a filter is given), completeness >= threshold, and evaluated by at least
    ``min_distinct_validators`` distinct validators (winner-take-all guard: a skip-thinned window
    can't crown #1 on too few validators' opinions). The dominance ranker sees
    ``pass_rate = avg_rate`` (per-validator mean) and ``passed_count = total_passed`` (magnitude
    tie-break). Returns ``(winner_hotkey, rank_entries)``.
    """
    miner_stats: List[Tuple[str, int, float, Optional[int]]] = []
    for miner, agg in aggregates.items():
        if valid_miners and miner not in valid_miners:
            continue
        if agg.completeness < completeness_threshold - 1e-12:
            continue
        if len(agg.per_validator_rate) < min_distinct_validators:
            continue
        miner_stats.append((miner, agg.total_passed, agg.avg_rate, agg.block))
    return compute_rank_from_miner_stats(miner_stats, dominance_threshold)


def aggregate_per_validator_average(
    verdicts: Verdicts,
    window_task_ids: List[int],
    valid_miners: Set[str],
    block_by_hotkey: Dict[str, Optional[int]],
    dominance_threshold: float,
    completeness_threshold: float = COMPLETENESS_ELIGIBILITY_THRESHOLD,
    min_distinct_validators: int = 1,
) -> Tuple[Optional[str], List[dict]]:
    """Convenience wrapper: build aggregates then rank. Returns ``(winner_hotkey, rank_entries)``."""
    aggregates = compute_miner_aggregates(verdicts, window_task_ids, block_by_hotkey)
    return rank_from_aggregates(
        aggregates,
        valid_miners,
        dominance_threshold,
        completeness_threshold,
        min_distinct_validators,
    )
