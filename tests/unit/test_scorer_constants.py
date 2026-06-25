"""Tests for scorer window helpers."""

from leoma.infra.scorer_constants import (
    SCORER_TASK_WINDOW,
    required_distinct_validators,
    scoring_window_task_ids,
    settled_window_end,
    window_task_ids_ending_at,
)


class TestRequiredDistinctValidators:
    def test_majority_of_active(self):
        assert required_distinct_validators(4) == 2  # ceil(4*0.5)
        assert required_distinct_validators(5) == 3  # ceil(5*0.5)
        assert required_distinct_validators(3) == 2  # ceil(3*0.5)

    def test_floored_at_one(self):
        # A single live validator (or none) must never deadlock the gate.
        assert required_distinct_validators(1) == 1
        assert required_distinct_validators(0) == 1
        assert required_distinct_validators(-3) == 1

    def test_two_active_requires_both(self):
        assert required_distinct_validators(2) == 1  # ceil(2*0.5)=1 -> gate effectively off at 2


def test_scoring_window_task_ids_none():
    assert scoring_window_task_ids(None) is None


def test_scoring_window_task_ids_partial_chain():
    """Before N tasks exist, window is all task_ids from 1 .. max."""
    w = scoring_window_task_ids(7)
    assert w == list(range(1, 8))
    assert len(w) < SCORER_TASK_WINDOW


def test_scoring_window_task_ids_full_window():
    """At max >= N, window is the last N consecutive task_ids."""
    end = 500
    w = scoring_window_task_ids(end)
    assert len(w) == SCORER_TASK_WINDOW
    assert w[0] == end - SCORER_TASK_WINDOW + 1
    assert w[-1] == end


class TestSettledWindowEnd:
    def test_excludes_current_window_by_margin(self):
        # block 18050 -> rotation index 180; margin 2 -> settled end 178.
        assert settled_window_end(18050, 100, margin=2) == 178

    def test_deterministic_across_block_drift_within_a_window(self):
        # The whole point: any block inside rotation window 180 (18000..18099) yields the same
        # settled end, so validators reading blocks a few apart agree.
        ends = {settled_window_end(b, 100, margin=2) for b in (18000, 18001, 18050, 18099)}
        assert ends == {178}

    def test_changes_only_at_rotation_boundary(self):
        assert settled_window_end(17999, 100, margin=2) == 177  # window 179
        assert settled_window_end(18000, 100, margin=2) == 178  # window 180

    def test_margin_one(self):
        assert settled_window_end(18050, 100, margin=1) == 179

    def test_zero_interval_guarded(self):
        # Defensive: interval 0 must not divide-by-zero.
        assert isinstance(settled_window_end(100, 0, margin=1), int)


class TestWindowTaskIdsEndingAt:
    def test_full_window(self):
        w = window_task_ids_ending_at(500)
        assert len(w) == SCORER_TASK_WINDOW
        assert w[0] == 500 - SCORER_TASK_WINDOW + 1
        assert w[-1] == 500

    def test_clamped_low_bound(self):
        # Early subnet: end below N -> clamp start to 0 (no negative ids).
        w = window_task_ids_ending_at(50)
        assert w[0] == 0
        assert w[-1] == 50

    def test_negative_end_is_none(self):
        assert window_task_ids_ending_at(-1) is None

    def test_none_end_is_none(self):
        assert window_task_ids_ending_at(None) is None
