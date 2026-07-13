"""The freeze cheat, and the gate that stops it inheriting the crown.

Leoma scores *closeness to the real continuation*, which creates a cheat the
reference subnet structurally cannot have: a model that simply **holds the
conditioning frame** scores well on any clip that doesn't move much, having learned
nothing at all.

The 1-frame version of this is already dead (`metrics.require_frames`). What remains
is a *full-length* freeze — a perfectly legal generation that can genuinely beat a
mediocre king. The gate makes the cheat a **third duelist**, run through the same
`run_duel` loop, the same metric, the same clips and the same bootstrap, so a
challenger must beat the king *and* the cheat.
"""

import numpy as np
import pytest

from leoma.eval.baselines import evaluate_freeze_gates, freeze_frames, freeze_scores
from leoma.eval.metrics import mse
from leoma.eval.video_runner import Clip, GenParams, run_duel

PARAMS = GenParams(num_frames=6, fps=2, width=8, height=8)


def _moving_clips(n=16, seed=0):
    """Clips with real motion: frame t is the first frame plus t*drift."""
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(n):
        first = rng.integers(0, 200, size=(8, 8, 3)).astype("uint8")
        frames = [np.clip(first.astype(int) + t * 10, 0, 255).astype("uint8") for t in range(6)]
        truth = np.stack(frames)
        clips.append(Clip(clip_index=i, clip_id=f"clip-{i:04d}", first_frame=truth[0],
                          prompt="p", truth_frames=truth, params=PARAMS))
    return clips


def _static_clips(n=16, seed=1):
    """Clips that barely move. This is the terrain the freeze cheat is built for."""
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(n):
        first = rng.integers(0, 200, size=(8, 8, 3)).astype("uint8")
        truth = np.repeat(first[None, ...], 6, axis=0)
        clips.append(Clip(clip_index=i, clip_id=f"clip-{i:04d}", first_frame=truth[0],
                          prompt="p", truth_frames=truth, params=PARAMS))
    return clips


def _freeze(clip, seed):
    """A cheating "model": it just holds the conditioning frame."""
    return freeze_frames(clip, seed)


def _bad(clip, seed):
    """A genuinely mediocre model: it moves, but wrongly."""
    return np.clip(clip.truth_frames.astype(int) + 60, 0, 255).astype("uint8")


def _good(clip, seed):
    """A model that actually tracks the real continuation."""
    return np.clip(clip.truth_frames.astype(int) + 2, 0, 255).astype("uint8")


def _duel(clips, king, challenger, *, margin=0.0):
    return run_duel(
        clips, generate_king=king, generate_challenger=challenger, distance_fn=mse,
        master_seed=7, delta_threshold=0.0025, alpha=0.05, n_bootstrap=400,
        freeze_margin_fraction=margin,
    )


class TestTheCheatIsReal:
    def test_freezing_beats_a_mediocre_king_on_static_clips(self):
        """The premise. If this ever stops being true the gate is unnecessary — but on
        low-motion clips a frozen frame really does score better than a model that
        moves in the wrong direction."""
        clips = _static_clips()
        cheat = [mse(_freeze(c, 0), c.truth_frames) for c in clips]
        mediocre = [mse(_bad(c, 0), c.truth_frames) for c in clips]
        assert np.mean(cheat) < np.mean(mediocre)

    def test_freeze_frames_is_full_length_and_therefore_legal(self):
        """It is NOT the 1-frame exploit — the metrics already reject that. This is a
        perfectly well-formed generation that happens to be a cheat."""
        clip = _moving_clips(1)[0]
        frames = _freeze(clip, 0)
        assert frames.shape == clip.truth_frames.shape
        assert mse(frames, clip.truth_frames) > 0  # it is scored, not rejected


class TestTheGate:
    def test_a_freeze_cheater_that_BEAT_the_king_is_still_rejected(self):
        """THE test. On static clips the cheat beats a mediocre king — so without the
        gate it would take the crown having learned nothing."""
        clips = _static_clips()

        ungated = run_duel(
            clips, generate_king=_bad, generate_challenger=_freeze, distance_fn=mse,
            master_seed=7, delta_threshold=0.0025, alpha=0.05, n_bootstrap=400,
        )
        assert ungated["accepted"] is True, "premise broken: the cheat did not beat the king"

        gated = _duel(clips, king=_bad, challenger=_freeze)
        assert gated["accepted"] is False
        assert gated["rejected_by"] == "freeze_gate"
        assert "has not learned anything" in gated["reason"]

    def test_a_genuinely_good_model_still_wins(self):
        """The gate must not become a wall. A model that tracks reality beats both the
        king and the cheat."""
        clips = _moving_clips()
        v = _duel(clips, king=_bad, challenger=_good)
        assert v["accepted"] is True
        assert v["gates"]["challenger_passed"] is True

    def test_the_gate_can_only_ever_take_the_crown_away(self):
        """It runs after the verdict and can only flip accepted True -> False. A model
        that lost to the king cannot be *promoted* by beating the cheat."""
        clips = _moving_clips()
        v = _duel(clips, king=_good, challenger=_bad)
        assert v["accepted"] is False
        assert "rejected_by" not in v          # it lost on the merits, not on the gate

    def test_the_verdict_carries_the_freeze_baseline_for_the_dashboard(self):
        """avg_freeze_distance is the number that lets the margin be MEASURED later
        rather than invented. It is the cheat floor the whole subnet is judged against."""
        v = _duel(_moving_clips(), king=_bad, challenger=_good)
        assert v["gates"]["avg_freeze_distance"] > 0
        assert v["gates"]["challenger"]["margin_fraction"] == 0.0

    def test_a_failing_KING_raises_an_alarm_but_is_NOT_deposed(self):
        """Auto-dethroning on a gate failure would be an attack vector: shift the corpus
        static, the incumbent 'fails', and the crown falls to a marginal challenger."""
        events = []
        clips = _static_clips()

        def _also_bad(clip, seed):   # a DIFFERENT mediocre model, so this isn't a copy
            return np.clip(clip.truth_frames.astype(int) + 55, 0, 255).astype("uint8")

        v = run_duel(
            clips, generate_king=_bad, generate_challenger=_also_bad, distance_fn=mse,
            master_seed=7, delta_threshold=0.0025, alpha=0.05, n_bootstrap=400,
            freeze_margin_fraction=0.0, on_phase=events.append,
        )
        alarms = [e for e in events if e.get("phase") == "king_alarm"]
        assert alarms, "a king no better than a frozen frame must raise an alarm"
        assert v["gates"]["king_failed"] is True
        # ...but nothing in the verdict deposes it. The operator re-seeds via chain.toml.
        assert v.get("rejected_by") != "king_failed"

    def test_the_gate_is_off_when_the_margin_is_not_configured(self):
        """None means 'do not run the gate' — distinct from 0.0, which means 'run it
        with no extra headroom'."""
        v = run_duel(
            _static_clips(), generate_king=_bad, generate_challenger=_freeze, distance_fn=mse,
            master_seed=7, delta_threshold=0.0025, alpha=0.05, n_bootstrap=400,
        )
        assert "gates" not in v


class TestTheMarginIsScaleFree:
    def test_the_margin_is_a_fraction_of_the_cheats_own_score(self):
        """LPIPS, MSE and flow live on wildly different numeric scales. An ABSOLUTE
        margin would silently mean something completely different on each; a fraction
        of the baseline's own mean survives any metric recalibration."""
        clips = _moving_clips()
        baseline = freeze_scores(clips, mse)
        king = [mse(_bad(c, 0), c.truth_frames) for c in clips]
        chall = [mse(_good(c, 0), c.truth_frames) for c in clips]

        gates = evaluate_freeze_gates(
            clips, king, chall, mse,
            margin_fraction=0.1, alpha=0.05, n_bootstrap=400, seed=1,
        )
        assert gates["challenger"]["margin"] == pytest.approx(0.1 * float(np.mean(baseline)), rel=1e-3)

    def test_a_demanding_margin_rejects_a_merely_ok_model(self):
        """Turning the margin up is how the subnet tightens later — one number, and it
        means the same thing under every metric."""
        clips = _moving_clips()
        lenient = _duel(clips, king=_bad, challenger=_good, margin=0.0)
        assert lenient["accepted"] is True

        # Demand the challenger beat the cheat by 10x the cheat's own average distance.
        strict = _duel(clips, king=_bad, challenger=_good, margin=10.0)
        assert strict["accepted"] is False
        assert strict["rejected_by"] == "freeze_gate"


class TestItIsTheSameStatisticalPrimitive:
    def test_the_gate_reuses_paired_bootstrap_verdict(self):
        """One statistical path, two opponents. A second, bespoke significance test is a
        second thing to get subtly wrong."""
        clips = _moving_clips()
        baseline = freeze_scores(clips, mse)
        chall = [mse(_good(c, 0), c.truth_frames) for c in clips]

        from leoma.eval.baselines import freeze_gate
        from leoma.eval.bootstrap import paired_bootstrap_verdict

        gate = freeze_gate(chall, baseline, margin_fraction=0.0, alpha=0.05, n_bootstrap=400, seed=3)
        direct = paired_bootstrap_verdict(
            baseline, chall, delta_threshold=0.0, alpha=0.05, n_bootstrap=400, seed=3
        )
        assert gate["lcb"] == direct["lcb"]
        assert gate["passed"] == direct["accepted"]

    def test_the_baseline_costs_no_gpu(self):
        """The cheat needs no generation — it is the conditioning frame, repeated. The
        gate is a rounding error next to the hours the real generations take."""
        clip = _moving_clips(1)[0]
        frames = freeze_frames(clip, 999)
        assert np.array_equal(frames[0], clip.first_frame)
        assert np.array_equal(frames[-1], clip.first_frame)
        assert frames.shape[0] == clip.truth_frames.shape[0]
