"""
Unit tests for the pure dashboard helpers (validator participation, miner activity).

Self-eval model: each sample's validator_hotkey is that task's sampler. Participation measures
whether a validator sampled on its rotation turns.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

from leoma.delivery.http.dashboard_service import (
    expected_turns_for,
    compute_validator_participation,
    compute_miner_activity,
)


def _s(validator, task_id, miner, passed=True, latency=None, when=None):
    return SimpleNamespace(
        validator_hotkey=validator,
        task_id=task_id,
        miner_hotkey=miner,
        passed=passed,
        latency_ms=latency,
        evaluated_at=when,
    )


class TestExpectedTurns:
    def test_round_robin_distribution(self):
        ordered = ["A", "B", "C"]
        window = [0, 1, 2, 3, 4, 5]
        assert expected_turns_for("A", ordered, window) == 2  # indices 0, 3
        assert expected_turns_for("B", ordered, window) == 2  # 1, 4
        assert expected_turns_for("C", ordered, window) == 2  # 2, 5

    def test_empty_ordered(self):
        assert expected_turns_for("A", [], [1, 2, 3]) == 0

    def test_validator_not_in_ring(self):
        assert expected_turns_for("Z", ["A", "B"], [0, 1, 2, 3]) == 0


class TestValidatorParticipation:
    def test_full_and_partial_and_absent(self):
        ordered = ["A", "B", "C"]
        window = [0, 1, 2, 3, 4, 5]
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # A sampled its turns 0 and 3 (2 miners each); B sampled only turn 1 (skipped 4); C nothing.
        samples = [
            _s("A", 0, "m1", latency=100, when=t0), _s("A", 0, "m2", latency=200, when=t0),
            _s("A", 3, "m1", latency=300, when=t0),
            _s("B", 1, "m1", latency=None, when=t0),
        ]
        part = compute_validator_participation(samples, ordered, window)

        assert part["A"].tasks_sampled == 2
        assert part["A"].evaluations == 3
        assert part["A"].expected_turns == 2
        assert part["A"].participation_rate == 1.0
        assert part["A"].last_task_id == 3
        assert part["A"].avg_latency_ms == 200  # (100+200+300)/3

        assert part["B"].tasks_sampled == 1
        assert part["B"].expected_turns == 2
        assert part["B"].participation_rate == 0.5
        assert part["B"].avg_latency_ms is None

        # C is in the ring but never sampled -> included with zeros.
        assert "C" in part
        assert part["C"].tasks_sampled == 0
        assert part["C"].participation_rate == 0.0

    def test_rate_capped_at_one(self):
        # A produced more distinct tasks than its expected turns (e.g. set changed) -> cap 1.0.
        ordered = ["A", "B"]
        window = [0, 2]  # A's expected turns in window = {0, 2} -> 2
        samples = [_s("A", 0, "m"), _s("A", 2, "m"), _s("A", 4, "m")]  # 3 distinct, but 4 not in window
        part = compute_validator_participation(samples, ordered, window)
        # task 4 is outside the window list, but compute counts distinct tasks present in samples;
        # expected=2, tasks_sampled counts all distinct in samples (0,2,4)=3 -> capped at 1.0
        assert part["A"].participation_rate == 1.0


class TestMinerActivity:
    def test_activity_aggregation(self):
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        samples = [
            _s("A", 0, "m", passed=True, latency=100, when=t0),
            _s("B", 1, "m", passed=False, latency=200, when=t0),
            _s("A", 3, "m", passed=True, latency=None, when=t0),
        ]
        act = compute_miner_activity(samples)
        m = act["m"]
        assert m.tasks_evaluated == 3
        assert m.validators_evaluating == 2  # A and B
        assert m.passed_tasks == 2
        assert m.avg_latency_ms == 150  # (100+200)/2, None ignored
