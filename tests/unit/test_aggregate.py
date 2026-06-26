"""
Unit tests for pooled pass-rate aggregation (self-evaluation model).

One verdict per (validator, task, miner). A miner's score is the POOLED pass-rate
(total_passed / total_evaluated), gated by global completeness AND per-validator completeness
(>= K distinct validators that each covered >= 80% of their own slice), ranked by dominance.
"""

from leoma.infra.aggregate import (
    compute_miner_aggregates,
    rank_from_aggregates,
    aggregate_scores,
)


def _verdicts(entries):
    """entries: list of (validator, task_id, miner, passed) -> dict keyed by (v, t, m)."""
    return {(v, t, m): p for (v, t, m, p) in entries}


class TestComputeMinerAggregates:
    def test_single_validator_score_and_completeness(self):
        window = [1, 2, 3, 4, 5]
        v = _verdicts([("A", t, "m", t != 5) for t in window])  # passes 1-4, fails 5
        aggs = compute_miner_aggregates(v, window)
        m = aggs["m"]
        assert m.score == 0.8
        assert m.total_passed == 4
        assert m.total_evaluated == 5
        assert m.completeness == 1.0
        assert m.per_validator_rate == {"A": 0.8}
        assert m.per_validator_completeness == {"A": 1.0}  # A sampled all 5, evaluated m on all 5

    def test_pooled_passrate_not_per_validator_average(self):
        # A samples 10 tasks (5 pass); B samples 2 tasks (2 pass).
        # Pooled = (5+2)/(10+2) = 7/12 ~ 0.583. (The old per-validator mean would be 0.75 — NOT used.)
        window = list(range(1, 13))
        entries = [("A", t, "m", t <= 5) for t in range(1, 11)]
        entries += [("B", t, "m", True) for t in range(11, 13)]
        aggs = compute_miner_aggregates(_verdicts(entries), window)
        m = aggs["m"]
        assert round(m.score, 4) == round(7 / 12, 4)
        assert m.total_passed == 7 and m.total_evaluated == 12
        assert round((0.5 + 1.0) / 2, 3) == 0.75  # the per-validator mean we are deliberately NOT using

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
        assert aggs["m"].score == 1.0


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

    def test_higher_score_wins(self):
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
        # 6 tasks: A samples 1-3, B samples 4-6; miner "m" evaluated by both on every task they
        # sampled -> per-validator completeness 1.0 for both.
        window = list(range(1, 7))
        entries = [("A", t, "m", True) for t in range(1, 4)]
        entries += [("B", t, "m", True) for t in range(4, 7)]
        return compute_miner_aggregates(_verdicts(entries), window)

    def test_passes_when_distinct_validators_meets_min(self):
        aggs = self._two_validator_window()
        assert aggs["m"].per_validator_completeness == {"A": 1.0, "B": 1.0}
        winner, ranked = rank_from_aggregates(aggs, set(), 0.05, min_distinct_validators=2)
        assert winner == "m" and ranked[0]["miner_hotkey"] == "m"

    def test_excluded_when_below_min(self):
        # Same miner, but require 3 distinct validators -> gated out (only 2 evaluated it).
        aggs = self._two_validator_window()
        winner, ranked = rank_from_aggregates(aggs, set(), 0.05, min_distinct_validators=3)
        assert winner is None and ranked == []

    def test_low_coverage_validator_does_not_count(self):
        # V1 covers m on its whole slice (1.0); V2 only covers m on 40% of its slice (0.4).
        # Two validators evaluated m, but only ONE cleared 80% coverage -> fails min_distinct=2.
        window = list(range(1, 11))
        entries = [("V1", t, "m", True) for t in range(1, 6)]         # V1 slice {1..5}, m on all 5
        entries += [("V2", t, "other", True) for t in range(6, 11)]  # V2 slice {6..10} (via 'other')
        entries += [("V2", t, "m", True) for t in range(6, 8)]       # m on only 2 of V2's 5
        aggs = compute_miner_aggregates(_verdicts(entries), window)
        assert aggs["m"].per_validator_completeness == {"V1": 1.0, "V2": 0.4}
        # Isolate this gate (turn off the global completeness gate):
        winner, ranked = rank_from_aggregates(
            aggs, set(), 0.05, completeness_threshold=0.0, min_distinct_validators=2
        )
        assert winner is None and ranked == []
        # With min=1 the single well-covered validator suffices.
        winner2, _ = rank_from_aggregates(
            aggs, set(), 0.05, completeness_threshold=0.0, min_distinct_validators=1
        )
        assert winner2 == "m"

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

    def test_default_min_is_one_well_covered_validator(self):
        # Default min_distinct_validators=1: a single validator that fully covered the miner ranks.
        window = [1, 2, 3, 4, 5]
        aggs = compute_miner_aggregates(_verdicts([("A", t, "m", True) for t in window]), window)
        winner, ranked = rank_from_aggregates(aggs, set(), 0.05)
        assert winner == "m"


class TestAggregateScoresWrapper:
    def test_end_to_end(self):
        window = list(range(1, 11))
        # Two validators; pooled pass-rate over all evaluations.
        entries = [("A", t, "m", True) for t in range(1, 6)]          # A: 5/5 passed
        entries += [("B", t, "m", t % 2 == 0) for t in range(6, 11)]  # B: tasks 6,8,10 -> 3/5 passed
        winner, ranked = aggregate_scores(
            _verdicts(entries), window, {"m"}, {"m": 1}, 0.05
        )
        # pooled = (5 + 3) / (5 + 5) = 0.8
        assert ranked[0]["pass_rate"] == 0.8
        assert winner == "m"
