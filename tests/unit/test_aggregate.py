"""
Unit tests for per-validator-average aggregation (self-evaluation model).

One verdict per (validator, task, miner). Each validator's own pass-rate is computed, then a
miner's score is the MEAN across validators (equal weight), gated by completeness, ranked by
dominance. Covers: the per-validator-average property, completeness vs existing-tasks denominator,
validity filter, and ranking.
"""

from leoma.infra.aggregate import (
    compute_miner_aggregates,
    rank_from_aggregates,
    aggregate_per_validator_average,
)


def _verdicts(entries):
    """entries: list of (validator, task_id, miner, passed) -> dict keyed by (v, t, m)."""
    return {(v, t, m): p for (v, t, m, p) in entries}


class TestComputeMinerAggregates:
    def test_single_validator_rate_and_completeness(self):
        window = [1, 2, 3, 4, 5]
        v = _verdicts([("A", t, "m", t != 5) for t in window])  # passes 1-4, fails 5
        aggs = compute_miner_aggregates(v, window)
        m = aggs["m"]
        assert m.avg_rate == 0.8
        assert m.total_passed == 4
        assert m.total_evaluated == 5
        assert m.completeness == 1.0
        assert m.per_validator_rate == {"A": 0.8}

    def test_per_validator_average_not_task_pooled(self):
        # A samples 10 tasks (5 pass -> rate 0.5); B samples 2 tasks (2 pass -> rate 1.0).
        # Equal weight => (0.5 + 1.0)/2 = 0.75, NOT task-pooled 7/12 = 0.583.
        window = list(range(1, 13))
        entries = [("A", t, "m", t <= 5) for t in range(1, 11)]
        entries += [("B", t, "m", True) for t in range(11, 13)]
        aggs = compute_miner_aggregates(_verdicts(entries), window)
        m = aggs["m"]
        assert m.avg_rate == 0.75
        assert m.total_passed == 7
        assert m.total_evaluated == 12
        assert round(7 / 12, 3) == 0.583  # task-pooled value we are deliberately NOT using

    def test_completeness_uses_existing_tasks_not_window_size(self):
        # Window of 10, but only tasks 1-6 ever produced a verdict (turns 7-10 skipped).
        window = list(range(1, 11))
        entries = [("A", t, "m", True) for t in range(1, 7)]
        aggs = compute_miner_aggregates(_verdicts(entries), window)
        # Evaluated on all 6 existing tasks => completeness 1.0 (not 6/10).
        assert aggs["m"].completeness == 1.0

    def test_verdicts_outside_window_ignored(self):
        window = [1, 2, 3]
        entries = [("A", t, "m", True) for t in window] + [("A", 999, "m", False)]
        aggs = compute_miner_aggregates(_verdicts(entries), window)
        assert aggs["m"].total_evaluated == 3
        assert aggs["m"].avg_rate == 1.0


class TestRankFromAggregates:
    def test_empty_no_winner(self):
        winner, entries = rank_from_aggregates({}, set(), 0.05)
        assert winner is None and entries == []

    def test_completeness_threshold_excludes(self):
        # 10 tasks exist (another miner is in all of them); "sparse" is in only 5 -> 50% < 80%.
        window = list(range(1, 11))
        entries = [("A", t, "other", True) for t in window]
        entries += [("A", t, "sparse", True) for t in range(1, 6)]
        aggs = compute_miner_aggregates(_verdicts(entries), window)
        assert aggs["sparse"].completeness == 0.5
        winner, ranked = rank_from_aggregates(aggs, {"sparse"}, 0.05)
        assert winner is None and ranked == []

    def test_valid_filter_excludes_unknown_miner(self):
        window = [1, 2, 3, 4, 5]
        aggs = compute_miner_aggregates(_verdicts([("A", t, "m", True) for t in window]), window)
        winner, ranked = rank_from_aggregates(aggs, {"someone_else"}, 0.05)
        assert winner is None

    def test_higher_avg_rate_wins(self):
        window = list(range(1, 11))
        entries = [("A", t, "hi", t <= 9) for t in window]   # 0.9
        entries += [("A", t, "lo", t <= 5) for t in window]  # 0.5
        aggs = compute_miner_aggregates(_verdicts(entries), window, {"hi": 1, "lo": 2})
        winner, ranked = rank_from_aggregates(aggs, {"hi", "lo"}, 0.05)
        assert winner == "hi"
        assert ranked[0]["miner_hotkey"] == "hi"
        assert ranked[0]["pass_rate"] == 0.9


class TestMinDistinctValidatorsGate:
    def _two_validator_window(self):
        # 6 tasks: A samples 1-3, B samples 4-6; miner "m" evaluated by both -> 2 distinct validators.
        window = list(range(1, 7))
        entries = [("A", t, "m", True) for t in range(1, 4)]
        entries += [("B", t, "m", True) for t in range(4, 7)]
        return compute_miner_aggregates(_verdicts(entries), window)

    def test_passes_when_distinct_validators_meets_min(self):
        aggs = self._two_validator_window()
        assert len(aggs["m"].per_validator_rate) == 2
        winner, ranked = rank_from_aggregates(aggs, set(), 0.05, min_distinct_validators=2)
        assert winner == "m" and ranked[0]["miner_hotkey"] == "m"

    def test_excluded_when_below_min(self):
        # Same miner, but require 3 distinct validators -> gated out (only 2 evaluated it).
        aggs = self._two_validator_window()
        winner, ranked = rank_from_aggregates(aggs, set(), 0.05, min_distinct_validators=3)
        assert winner is None and ranked == []

    def test_single_supporter_miner_excluded(self):
        # "broad" seen by A and B; "solo" seen only by A. With min=2, solo is dropped, broad wins.
        window = list(range(1, 5))
        entries = [("A", t, "broad", True) for t in window]
        entries += [("B", t, "broad", True) for t in window]
        entries += [("A", t, "solo", True) for t in window]
        aggs = compute_miner_aggregates(_verdicts(entries), window, {"broad": 1, "solo": 2})
        winner, ranked = rank_from_aggregates(aggs, set(), 0.05, min_distinct_validators=2)
        ranked_miners = {e["miner_hotkey"] for e in ranked}
        assert "solo" not in ranked_miners
        assert winner == "broad"

    def test_default_min_is_noop(self):
        # Default min_distinct_validators=1 keeps prior behavior (a single validator can rank).
        window = [1, 2, 3, 4, 5]
        aggs = compute_miner_aggregates(_verdicts([("A", t, "m", True) for t in window]), window)
        winner, ranked = rank_from_aggregates(aggs, set(), 0.05)
        assert winner == "m"


class TestAggregateConvenienceWrapper:
    def test_end_to_end(self):
        window = list(range(1, 11))
        # Two validators, equal weight; miner passes more often under A than B.
        entries = [("A", t, "m", True) for t in range(1, 6)]          # A: 5/5 = 1.0
        entries += [("B", t, "m", t % 2 == 0) for t in range(6, 11)]  # B: tasks 6,8,10 -> 3/5 = 0.6
        aggs_winner, ranked = aggregate_per_validator_average(
            _verdicts(entries), window, {"m"}, {"m": 1}, 0.05
        )
        # avg = (1.0 + 0.6)/2 = 0.8
        assert ranked[0]["pass_rate"] == 0.8
        assert aggs_winner == "m"
