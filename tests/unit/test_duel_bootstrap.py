"""
Unit tests for the paired-bootstrap duel verdict.

The verdict crowns a challenger only when its per-clip advantage over the king
(king_distance - challenger_distance, lower distance = better) clears the delta
threshold at the bootstrap lower-confidence bound.
"""

import pytest

from leoma.eval.bootstrap import paired_bootstrap_verdict, can_still_win

DELTA = 0.0025
ALPHA = 0.001
NB = 2000


def _verdict(king, chall, seed=7):
    return paired_bootstrap_verdict(
        king, chall, delta_threshold=DELTA, alpha=ALPHA, n_bootstrap=NB, seed=seed
    )


class TestVerdict:
    def test_clear_challenger_win_crowns(self):
        king = [0.50, 0.52, 0.48, 0.55, 0.51] * 6
        chall = [0.40, 0.41, 0.39, 0.42, 0.40] * 6
        v = _verdict(king, chall)
        assert v["accepted"] is True
        assert v["verdict"] == "challenger"
        assert v["lcb"] > DELTA
        assert v["mu_hat"] > 0

    def test_king_better_keeps_crown(self):
        king = [0.40, 0.41, 0.39] * 8
        chall = [0.50, 0.52, 0.48] * 8
        v = _verdict(king, chall)
        assert v["accepted"] is False
        assert v["verdict"] == "king"
        assert v["mu_hat"] < 0

    def test_tie_keeps_king(self):
        scores = [0.5, 0.4, 0.6, 0.45] * 5
        v = _verdict(scores, list(scores))
        assert v["accepted"] is False
        assert v["lcb"] <= DELTA

    def test_tiny_win_below_threshold_keeps_king(self):
        # Challenger better by ~0.001 mean, under the 0.0025 delta -> not crowned.
        king = [0.5000] * 40
        chall = [0.4990] * 40
        v = _verdict(king, chall)
        assert v["accepted"] is False

    def test_deterministic_same_seed(self):
        king = [0.5, 0.52, 0.48, 0.55] * 5
        chall = [0.45, 0.47, 0.44, 0.5] * 5
        # Same seed -> identical scoring, every field (consensus requires this).
        # (timestamp is wall-clock and excluded.)
        a = _verdict(king, chall, seed=99)
        b = _verdict(king, chall, seed=99)
        a.pop("timestamp"); b.pop("timestamp")
        assert a == b

    def test_reports_averages_and_counts(self):
        king = [0.6, 0.6, 0.6, 0.6]
        chall = [0.4, 0.4, 0.4, 0.4]
        v = _verdict(king, chall)
        assert v["n_clips"] == 4
        assert v["avg_king_distance"] == 0.6
        assert v["avg_challenger_distance"] == 0.4

    @pytest.mark.parametrize("king,chall", [([], []), ([0.1], []), ([0.1, 0.2], [0.1])])
    def test_bad_inputs_raise(self, king, chall):
        with pytest.raises(ValueError):
            _verdict(king, chall)


class TestEarlyStop:
    def test_hopeless_cannot_win(self):
        # King far better so far; even zero further advantage can't reach delta.
        assert not can_still_win(
            [0.4, 0.4], [0.6, 0.6], remaining=2,
            delta_threshold=DELTA, best_possible_advantage=0.0,
        )

    def test_still_possible(self):
        assert can_still_win(
            [0.5], [0.5], remaining=9,
            delta_threshold=DELTA, best_possible_advantage=1.0,
        )
