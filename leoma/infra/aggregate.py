"""
Pooled pass-rate score aggregation (decentralized consensus, self-evaluation model).

Each task is sampled AND evaluated by exactly one validator (no cross-validation), so there is one
verdict per (validator, task, miner). A miner's score is its POOLED pass-rate over the window:

    score(M) = total_passed(M) / total_evaluated(M)

A validator that evaluated M on more tasks therefore has proportionally more influence — the pooled
rate is self-normalizing by count, so a validator with only a handful of evaluations can't swing it.
Two gates protect winner-take-all:

  1. Completeness: M must be evaluated on >= COMPLETENESS_ELIGIBILITY_THRESHOLD of the tasks that
     actually exist in the window (skipped rotation turns don't count against it).
  2. Min-distinct validators: M must be evaluated by >= min_distinct_validators distinct validators
     that EACH covered >= per_validator_completeness_threshold of THEIR OWN slice (per-validator
     completeness), so a miner can't qualify on thin or one-validator-heavy coverage.

Eligible miners are ranked by the dominance algorithm on the pooled score. Shared by the validator's
local weight-setter (verdicts read from peer buckets) and the owner-api dashboard scorer (verdicts
read from the DB); both use this same method, so the dashboard mirrors the on-chain ranking.
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
    score: float                          # pooled pass-rate: total_passed / total_evaluated
    total_passed: int                     # total passed tasks across all validators
    total_evaluated: int                  # total evaluated tasks across all validators
    completeness: float                   # distinct tasks evaluated / tasks that exist in window
    per_validator_rate: Dict[str, float] = field(default_factory=dict)          # validator -> pass-rate (display)
    per_validator_completeness: Dict[str, float] = field(default_factory=dict)  # validator -> coverage of its slice (gate)
    block: Optional[int] = None


def compute_miner_aggregates(
    verdicts: Verdicts,
    window_task_ids: List[int],
    block_by_hotkey: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, MinerAggregate]:
    """Build per-miner aggregates from per-(validator, task, miner) verdicts in the window.

    No miner/eligibility filtering here — returns every miner seen, so callers decide eligibility
    (and populate dashboard tables). Global completeness uses the tasks that actually exist in the
    window, and per-validator completeness uses each validator's actual slice (the tasks it sampled),
    not the nominal sizes — so skipped turns and offline validators don't distort either.
    """
    block_by_hotkey = block_by_hotkey or {}
    window_set = set(window_task_ids)

    existing_tasks: Set[int] = set()
    validator_tasks: Dict[str, Set[int]] = {}    # validator -> distinct tasks it sampled (its slice)
    pv_eval: Dict[Tuple[str, str], int] = {}     # (validator, miner) -> tasks where it evaluated miner
    pv_pass: Dict[Tuple[str, str], int] = {}     # (validator, miner) -> tasks where it passed miner
    miner_tasks: Dict[str, Set[int]] = {}        # miner -> distinct tasks evaluated (any validator)
    total_eval: Dict[str, int] = {}
    total_pass: Dict[str, int] = {}

    for (validator, task_id, miner), passed in verdicts.items():
        if task_id not in window_set:
            continue
        existing_tasks.add(task_id)
        validator_tasks.setdefault(validator, set()).add(task_id)
        pv_eval[(validator, miner)] = pv_eval.get((validator, miner), 0) + 1
        total_eval[miner] = total_eval.get(miner, 0) + 1
        miner_tasks.setdefault(miner, set()).add(task_id)
        if passed:
            pv_pass[(validator, miner)] = pv_pass.get((validator, miner), 0) + 1
            total_pass[miner] = total_pass.get(miner, 0) + 1

    n_existing = len(existing_tasks)

    # Per-(validator, miner) pass-rate (dashboard display) and slice-coverage (the min-distinct gate).
    rates_by_miner: Dict[str, Dict[str, float]] = {}
    coverage_by_miner: Dict[str, Dict[str, float]] = {}
    for (validator, miner), evaluated in pv_eval.items():
        if evaluated <= 0:
            continue
        rates_by_miner.setdefault(miner, {})[validator] = pv_pass.get((validator, miner), 0) / evaluated
        slice_size = len(validator_tasks.get(validator, ()))
        if slice_size > 0:
            coverage_by_miner.setdefault(miner, {})[validator] = evaluated / slice_size

    aggregates: Dict[str, MinerAggregate] = {}
    for miner, evaluated in total_eval.items():
        score = total_pass.get(miner, 0) / evaluated if evaluated else 0.0
        completeness = (len(miner_tasks.get(miner, set())) / n_existing) if n_existing else 0.0
        aggregates[miner] = MinerAggregate(
            miner_hotkey=miner,
            score=score,
            total_passed=total_pass.get(miner, 0),
            total_evaluated=evaluated,
            completeness=completeness,
            per_validator_rate=rates_by_miner.get(miner, {}),
            per_validator_completeness=coverage_by_miner.get(miner, {}),
            block=block_by_hotkey.get(miner),
        )
    return aggregates


def rank_from_aggregates(
    aggregates: Dict[str, MinerAggregate],
    valid_miners: Set[str],
    dominance_threshold: float,
    completeness_threshold: float = COMPLETENESS_ELIGIBILITY_THRESHOLD,
    min_distinct_validators: int = 1,
    per_validator_completeness_threshold: float = COMPLETENESS_ELIGIBILITY_THRESHOLD,
) -> Tuple[Optional[str], List[dict]]:
    """Rank eligible miners by the pooled pass-rate using the dominance algorithm.

    Eligible = valid (if a filter is given), global completeness >= ``completeness_threshold``, and
    evaluated by at least ``min_distinct_validators`` distinct validators that EACH covered
    >= ``per_validator_completeness_threshold`` of their own slice (winner-take-all guard: a
    skip-thinned or one-validator-heavy window can't crown #1). The dominance ranker sees
    ``pass_rate = score`` (pooled) and ``passed_count = total_passed`` (magnitude tie-break).
    Returns ``(winner_hotkey, rank_entries)``.
    """
    miner_stats: List[Tuple[str, int, float, Optional[int]]] = []
    for miner, agg in aggregates.items():
        if valid_miners and miner not in valid_miners:
            continue
        if agg.completeness < completeness_threshold - 1e-12:
            continue
        qualifying = sum(
            1 for cov in agg.per_validator_completeness.values()
            if cov >= per_validator_completeness_threshold - 1e-12
        )
        if qualifying < min_distinct_validators:
            continue
        miner_stats.append((miner, agg.total_passed, agg.score, agg.block))
    return compute_rank_from_miner_stats(miner_stats, dominance_threshold)


def aggregate_scores(
    verdicts: Verdicts,
    window_task_ids: List[int],
    valid_miners: Set[str],
    block_by_hotkey: Dict[str, Optional[int]],
    dominance_threshold: float,
    completeness_threshold: float = COMPLETENESS_ELIGIBILITY_THRESHOLD,
    min_distinct_validators: int = 1,
    per_validator_completeness_threshold: float = COMPLETENESS_ELIGIBILITY_THRESHOLD,
) -> Tuple[Optional[str], List[dict]]:
    """Convenience wrapper: build aggregates then rank. Returns ``(winner_hotkey, rank_entries)``."""
    aggregates = compute_miner_aggregates(verdicts, window_task_ids, block_by_hotkey)
    return rank_from_aggregates(
        aggregates,
        valid_miners,
        dominance_threshold,
        completeness_threshold,
        min_distinct_validators,
        per_validator_completeness_threshold,
    )
