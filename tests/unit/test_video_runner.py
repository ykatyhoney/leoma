"""
Unit tests for the duel runner orchestration (numpy fakes, no torch/diffusers).
"""

import numpy as np
import pytest

from leoma.eval.metrics import (
    mse,
    ssim_distance,
    temporal_distance,
    make_composite,
    get_metric,
)
from leoma.eval.video_runner import Clip, GenParams, run_duel


def _clips(n=10, t=4, h=8, w=8, seed=0):
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(n):
        truth = rng.integers(0, 255, size=(t, h, w, 3)).astype("uint8")
        clips.append(Clip(clip_index=i, first_frame=truth[0], prompt="p",
                          truth_frames=truth, params=GenParams(num_frames=t, width=w, height=h)))
    return clips


def _near(clip, seed):  # generation close to ground truth
    return np.clip(clip.truth_frames.astype(float) + 3, 0, 255).astype("uint8")


def _far(clip, seed):   # generation far from ground truth
    return np.clip(clip.truth_frames.astype(float) + 90, 0, 255).astype("uint8")


class TestRunDuel:
    def test_challenger_closer_wins(self):
        clips = _clips()
        v = run_duel(clips, generate_king=_far, generate_challenger=_near, distance_fn=mse,
                     master_seed=42, delta_threshold=0.0025, alpha=0.001, n_bootstrap=500)
        assert v["verdict"] == "challenger"
        assert v["avg_king_distance"] > v["avg_challenger_distance"]
        assert len(v["per_clip"]) == len(clips)
        assert v["early_stopped"] is False

    def test_king_closer_keeps(self):
        clips = _clips()
        v = run_duel(clips, generate_king=_near, generate_challenger=_far, distance_fn=mse,
                     master_seed=42, delta_threshold=0.0025, alpha=0.001, n_bootstrap=500)
        assert v["verdict"] == "king"

    def test_works_with_ssim_metric(self):
        clips = _clips()
        v = run_duel(clips, generate_king=_far, generate_challenger=_near, distance_fn=ssim_distance,
                     master_seed=1, delta_threshold=0.0025, alpha=0.001, n_bootstrap=500)
        assert v["verdict"] == "challenger"

    def test_same_seed_reproducible(self):
        clips = _clips()
        v1 = run_duel(clips, _far, _near, mse, master_seed=7, delta_threshold=0.0025, alpha=0.001, n_bootstrap=500)
        v2 = run_duel(clips, _far, _near, mse, master_seed=7, delta_threshold=0.0025, alpha=0.001, n_bootstrap=500)
        assert v1["lcb"] == v2["lcb"]

    def test_king_and_challenger_get_same_gen_seed_per_clip(self):
        clips = _clips(n=3)
        seen = {"king": [], "chall": []}

        def gk(clip, seed):
            seen["king"].append((clip.clip_index, seed))
            return _far(clip, seed)

        def gc(clip, seed):
            seen["chall"].append((clip.clip_index, seed))
            return _near(clip, seed)

        run_duel(clips, gk, gc, mse, master_seed=5, delta_threshold=0.0025, alpha=0.001, n_bootstrap=100)
        assert seen["king"] == seen["chall"]  # identical (clip_index, seed) pairs

    def test_early_stop_hopeless(self):
        clips = _clips(n=20)
        v = run_duel(clips, generate_king=_near, generate_challenger=_far, distance_fn=mse,
                     master_seed=3, delta_threshold=0.0025, alpha=0.001, n_bootstrap=200,
                     early_stop_max_advantage=0.0)
        assert v["verdict"] == "king"
        assert v["early_stopped"] is True
        assert len(v["per_clip"]) < 20  # bailed early

    def test_disabled_early_stop_preserves_a_late_comeback(self):
        """An assumed bound can be exceeded; None must always score the full exam."""
        clips = _clips(n=20)

        def king(clip, _seed):
            if clip.clip_index < 2:
                return clip.truth_frames
            return np.clip(clip.truth_frames.astype(float) + 90, 0, 255).astype("uint8")

        def challenger(clip, _seed):
            if clip.clip_index < 2:
                return np.clip(clip.truth_frames.astype(float) + 90, 0, 255).astype("uint8")
            return clip.truth_frames

        complete = run_duel(
            clips, king, challenger, mse, master_seed=3,
            delta_threshold=0.0025, alpha=0.001, n_bootstrap=500,
            early_stop_max_advantage=None,
        )
        unsafe = run_duel(
            clips, king, challenger, mse, master_seed=3,
            delta_threshold=0.0025, alpha=0.001, n_bootstrap=500,
            early_stop_max_advantage=0.05,
        )

        assert complete["verdict"] == "challenger"
        assert complete["early_stopped"] is False
        assert len(complete["per_clip"]) == len(clips)
        assert unsafe["verdict"] == "king"
        assert unsafe["early_stopped"] is True

    def test_no_clips_raises(self):
        with pytest.raises(ValueError):
            run_duel([], _far, _near, mse, master_seed=1, delta_threshold=0.0025, alpha=0.001, n_bootstrap=10)


class TestMetrics:
    def test_mse_zero_for_identical(self):
        a = np.zeros((3, 4, 4, 3), dtype="uint8")
        assert mse(a, a) == 0.0

    def test_ssim_distance_zero_for_identical(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, size=(3, 6, 6, 3)).astype("uint8")
        assert ssim_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_longer_generation_is_truncated_to_the_truth(self):
        """A LONGER generation is fine — truncate it. Only a SHORTER one is a cheat."""
        gen = np.zeros((5, 4, 4, 3), dtype="uint8")
        truth = np.zeros((2, 4, 4, 3), dtype="uint8")
        assert mse(gen, truth) == 0.0

    def test_shorter_generation_raises(self):
        gen = np.zeros((2, 4, 4, 3), dtype="uint8")
        truth = np.zeros((5, 4, 4, 3), dtype="uint8")
        with pytest.raises(ValueError, match="too short"):
            mse(gen, truth)

    def test_size_mismatch_raises(self):
        a = np.zeros((2, 4, 4, 3), dtype="uint8")
        b = np.zeros((2, 8, 8, 3), dtype="uint8")
        with pytest.raises(ValueError):
            mse(a, b)

    def test_get_metric_unknown_raises(self):
        with pytest.raises(ValueError):
            get_metric("nope")

    def test_get_metric_default_is_lpips(self):
        # lpips is the default; resolves to a callable (torch loaded lazily on call).
        assert callable(get_metric(None))


class TestTemporalMetric:
    def test_identical_motion_is_zero(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, size=(5, 6, 6, 3)).astype("uint8")
        assert temporal_distance(a, a) == 0.0

    def test_frozen_generation_penalized(self):
        # Real clip moves; a frozen generation (all frames equal) has no motion,
        # so its temporal distance to the moving truth is > 0.
        rng = np.random.default_rng(1)
        truth = rng.integers(0, 255, size=(5, 6, 6, 3)).astype("uint8")
        frozen = np.repeat(truth[:1], 5, axis=0)
        assert temporal_distance(frozen, truth) > 0.0

    def test_single_frame_generation_raises(self):
        """This test used to assert the EXPLOIT: a 1-frame generation scored 0.0
        (perfect). A degenerate generation must never be rewarded."""
        truth = np.zeros((8, 4, 4, 3), dtype="uint8")
        one_frame = np.zeros((1, 4, 4, 3), dtype="uint8")
        with pytest.raises(ValueError, match="too short"):
            temporal_distance(one_frame, truth)

    def test_motion_axis_is_distinct_from_spatial(self):
        # Two generations with the SAME per-frame content set but different motion:
        # one matches the truth's frame order, the other reverses it. MSE (spatial,
        # orderless per frame here) ties, temporal distinguishes.
        rng = np.random.default_rng(2)
        truth = rng.integers(0, 255, size=(4, 5, 5, 3)).astype("uint8")
        reversed_gen = truth[::-1].copy()
        # spatial content identical set; temporal (motion direction) differs
        assert temporal_distance(reversed_gen, truth) > 0.0


class TestComposite:
    def test_blend_sums_weighted_parts(self):
        rng = np.random.default_rng(3)
        gen = rng.integers(0, 255, size=(4, 5, 5, 3)).astype("uint8")
        truth = rng.integers(0, 255, size=(4, 5, 5, 3)).astype("uint8")
        blended = make_composite({"mse": 1.0, "temporal": 0.5})
        expected = mse(gen, truth) + 0.5 * temporal_distance(gen, truth)
        assert blended(gen, truth) == pytest.approx(expected)

    def test_composite_via_get_metric_spec(self):
        m = get_metric("composite:mse=1.0,temporal=0.5")
        rng = np.random.default_rng(4)
        gen = rng.integers(0, 255, size=(3, 4, 4, 3)).astype("uint8")
        truth = rng.integers(0, 255, size=(3, 4, 4, 3)).astype("uint8")
        expected = mse(gen, truth) + 0.5 * temporal_distance(gen, truth)
        assert m(gen, truth) == pytest.approx(expected)

    def test_composite_default_weight_is_one(self):
        m = get_metric("composite:mse,ssim")
        rng = np.random.default_rng(5)
        gen = rng.integers(0, 255, size=(3, 4, 4, 3)).astype("uint8")
        truth = rng.integers(0, 255, size=(3, 4, 4, 3)).astype("uint8")
        assert m(gen, truth) == pytest.approx(mse(gen, truth) + ssim_distance(gen, truth))

    def test_empty_composite_raises(self):
        with pytest.raises(ValueError):
            make_composite({})

    def test_temporal_selectable_by_name(self):
        assert get_metric("temporal") is temporal_distance
