"""Tests for scorer window helpers."""

from leoma.infra.scorer_constants import SCORER_TASK_WINDOW, scoring_window_task_ids


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
