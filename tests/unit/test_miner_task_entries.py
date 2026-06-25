"""
Tests for build_miner_task_entries (miner detail "Recent tasks").

A task is sampled and self-evaluated by exactly one validator (the sampler for that window), and the
DB enforces one sample per (validator, task, miner) — so (task, miner) is a 1:1 mapping. Each entry
names that validator and uses its verdict directly: no consensus, no stake weighting.
"""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from leoma.delivery.http.routes._task_utils import build_miner_task_entries, representative_sample

NOW = datetime(2026, 6, 25, 12, 0, 0, tzinfo=timezone.utc)


def _sample(task_id, validator_hotkey, passed, *, latency_ms=1000, minutes_ago=0, overall_score=None):
    gen_art = None
    if overall_score is not None:
        gen_art = json.dumps({"overall_score": overall_score, "aspect_scores": {"a": overall_score}})
    return SimpleNamespace(
        task_id=task_id,
        validator_hotkey=validator_hotkey,
        passed=passed,
        latency_ms=latency_ms,
        evaluated_at=NOW - timedelta(minutes=minutes_ago),
        generated_artifacts=gen_art,
    )


def test_entry_names_the_evaluating_validator():
    entries = build_miner_task_entries([_sample(10, "5VAlpha", True)])
    assert len(entries) == 1
    assert entries[0].task_id == 10
    assert entries[0].validator_hotkey == "5VAlpha"
    assert entries[0].passed is True


def test_single_validator_verdict_is_used_regardless_of_stake():
    # Regression: the old stake-weighted path returned False when total stake was 0,
    # so a 0-stake sampler's PASS verdict was silently flipped to "Failed".
    entries = build_miner_task_entries([_sample(7, "5VZeroStake", True)])
    assert entries[0].passed is True


def test_failing_verdict_preserved():
    entries = build_miner_task_entries([_sample(7, "5VAlpha", False)])
    assert entries[0].passed is False


def test_duplicate_rows_latest_evaluation_wins():
    # Defensive only: 1:1 means this shouldn't occur, but if duplicate rows exist the most-recent
    # evaluation is reported verbatim — NOT a vote/majority across validators.
    samples = [
        _sample(5, "5VAlpha", True, minutes_ago=30),
        _sample(5, "5VBeta", True, minutes_ago=20),
        _sample(5, "5VGamma", False, minutes_ago=5),  # newest
    ]
    entries = build_miner_task_entries(samples)
    assert len(entries) == 1
    assert entries[0].passed is False  # the latest evaluation's verdict, not a 2-of-3 majority
    assert entries[0].validator_hotkey == "5VGamma"
    assert entries[0].updated == NOW - timedelta(minutes=5)


def test_overall_score_parsed_from_generated_artifacts():
    entries = build_miner_task_entries([_sample(10, "5VAlpha", False, overall_score=62)])
    assert entries[0].overall_score == 62


def test_overall_score_none_when_absent():
    entries = build_miner_task_entries([_sample(10, "5VAlpha", True)])  # no generated_artifacts
    assert entries[0].overall_score is None


def test_overall_score_taken_from_representative_evaluator():
    # If duplicate rows exist, the most-recent evaluation's score is the one reported.
    samples = [
        _sample(5, "5VAlpha", True, minutes_ago=30, overall_score=90),
        _sample(5, "5VGamma", True, minutes_ago=5, overall_score=77),  # newest -> representative
    ]
    entries = build_miner_task_entries(samples)
    assert entries[0].validator_hotkey == "5VGamma"
    assert entries[0].overall_score == 77


def test_representative_sample_picks_latest_and_handles_empty():
    assert representative_sample([]) is None
    rep = representative_sample([
        _sample(5, "5VAlpha", True, minutes_ago=30),
        _sample(5, "5VGamma", False, minutes_ago=5),
    ])
    assert rep.validator_hotkey == "5VGamma"


def test_entries_sorted_by_task_desc():
    entries = build_miner_task_entries(
        [_sample(3, "5VAlpha", True), _sample(9, "5VBeta", True), _sample(5, "5VGamma", False)]
    )
    assert [e.task_id for e in entries] == [9, 5, 3]
