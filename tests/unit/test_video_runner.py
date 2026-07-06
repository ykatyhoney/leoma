"""
Unit tests for the duel runner orchestration (numpy fakes, no torch/diffusers).
"""

import numpy as np
import pytest

from leoma.eval.metrics import mse, ssim_distance, get_metric
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

    def test_align_truncates_to_shortest(self):
        a = np.zeros((5, 4, 4, 3), dtype="uint8")
        b = np.zeros((2, 4, 4, 3), dtype="uint8")
        assert mse(a, b) == 0.0  # compares first 2 frames

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
