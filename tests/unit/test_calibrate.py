"""The cross-GPU noise-measurement analysis.

The number this produces is the one thing standing between Leoma's consensus
converging and forking: is ``delta_threshold`` wider than the hardware noise, or not?
Each test pins a property of the measurement, using synthetic per-box records so the
pure analysis is exercised without a GPU.

The experiment it analyzes: the SAME model, generated on N different GPU types, scored
on the SAME clips with the SAME seed. Any per-clip distance difference between two
boxes is pure hardware noise, because everything else is held identical. A self-duel
between two boxes has a true mean advantage of zero, so whatever the bootstrap finds is
the bias a marginal verdict could suffer.
"""

import numpy as np
import pytest

from leoma.eval.calibrate import analyze


def _record(gpu, distances, *, model="u/leoma-k@sha256:aaa", corpus="v1", digests=None):
    clips = []
    for i, d in enumerate(distances):
        clip = {"clip_id": f"clip-{i:04d}", "distance": float(d)}
        if digests is not None:
            clip["frames_digest"] = digests[i]
        clips.append(clip)
    return {"gpu": gpu, "model": model, "corpus_id": corpus, "clips": clips}


class TestValidation:
    def test_one_box_is_not_enough(self):
        with pytest.raises(ValueError, match="at least 2"):
            analyze([_record("H100", [0.1, 0.2])])

    def test_different_models_are_rejected(self):
        a = _record("H100", [0.1, 0.2], model="u/a@sha256:aaa")
        b = _record("A100", [0.1, 0.2], model="u/b@sha256:bbb")
        with pytest.raises(ValueError, match="different models"):
            analyze([a, b])

    def test_different_corpora_are_rejected(self):
        a = _record("H100", [0.1, 0.2], corpus="v1")
        b = _record("A100", [0.1, 0.2], corpus="v2")
        with pytest.raises(ValueError, match="different corpora"):
            analyze([a, b])


class TestNoiseMeasurement:
    def test_two_identical_boxes_have_zero_noise(self):
        """If both boxes produced the same distances (perfect reproducibility), the
        measured floor is exactly zero and any positive delta clears it."""
        dists = [0.30, 0.42, 0.28, 0.35, 0.31]
        r = analyze([_record("H100", dists), _record("A100", dists)],
                    current_delta_threshold=0.0025)
        assert r.max_abs_mu_hat == 0.0
        assert r.max_abs_delta == 0.0
        assert r.recommended_delta_threshold == 0.0
        assert r.verdict.startswith("PASS")

    def test_noise_shows_up_as_a_nonzero_floor(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(0.2, 0.5, size=40)
        noise = rng.normal(0, 0.01, size=40)   # 1e-2 per-clip hardware jitter
        r = analyze([_record("H100", base), _record("A100", base + noise)],
                    current_delta_threshold=0.0025)
        assert r.max_abs_mu_hat > 0
        assert r.max_abs_delta > 0
        assert r.recommended_delta_threshold > 0

    def test_the_recommendation_is_the_safety_multiple_of_the_floor(self):
        rng = np.random.default_rng(1)
        base = rng.uniform(0.2, 0.5, size=50)
        r3 = analyze([_record("H100", base), _record("A100", base + rng.normal(0, 0.01, 50))],
                     safety_factor=3.0)
        # Doubling the safety factor doubles the recommendation (same underlying floor).
        rng = np.random.default_rng(1)
        base = rng.uniform(0.2, 0.5, size=50)
        r6 = analyze([_record("H100", base), _record("A100", base + rng.normal(0, 0.01, 50))],
                     safety_factor=6.0)
        # Both are independently rounded to 6 decimals, so allow one rounding unit.
        assert r6.recommended_delta_threshold == pytest.approx(2 * r3.recommended_delta_threshold, abs=2e-6)

    def test_a_delta_below_the_floor_FAILS_loudly(self):
        """The whole point of the tool: catch a delta_threshold that would let two
        honest validators fork."""
        rng = np.random.default_rng(2)
        base = rng.uniform(0.2, 0.5, size=60)
        big_noise = base + rng.normal(0, 0.05, size=60)   # noise >> a tiny delta
        r = analyze([_record("H100", base), _record("A100", big_noise)],
                    current_delta_threshold=0.0001)   # absurdly tight
        assert r.verdict.startswith("FAIL")
        assert "fork" in r.verdict

    def test_a_delta_above_the_floor_PASSES(self):
        rng = np.random.default_rng(3)
        base = rng.uniform(0.2, 0.5, size=60)
        tiny_noise = base + rng.normal(0, 1e-5, size=60)   # noise << delta
        r = analyze([_record("H100", base), _record("A100", tiny_noise)],
                    current_delta_threshold=0.0025)
        assert r.verdict.startswith("PASS")


class TestManyBoxes:
    def test_all_pairs_are_compared(self):
        rng = np.random.default_rng(4)
        base = rng.uniform(0.2, 0.5, size=30)
        boxes = [_record(f"gpu{i}", base + rng.normal(0, 0.01, 30)) for i in range(4)]
        r = analyze(boxes)
        assert r.n_boxes == 4
        assert r.n_pairs == 6      # 4 choose 2

    def test_the_floor_is_the_WORST_pair_not_the_average(self):
        """One noisy box pair must drive the recommendation — consensus fails on the
        worst case, not the mean."""
        base = np.linspace(0.2, 0.5, 40)
        quiet = base + 1e-6
        loud = base + 0.03
        r = analyze([_record("H100", base), _record("A100", quiet), _record("MI300", loud)])
        # The H100-vs-MI300 (and A100-vs-MI300) pair dominates.
        worst = max(p.abs_delta_max for p in r.pairs)
        assert r.max_abs_delta == worst
        assert worst > 0.02


class TestBitReproducibility:
    def test_identical_frame_digests_are_reported_as_fully_reproducible(self):
        """If two boxes generated byte-identical frames, they are fully reproducible for
        this model — the ideal, and worth surfacing distinctly from 'close distances'."""
        dists = [0.3, 0.4, 0.35]
        digs = ["sha256:d0", "sha256:d1", "sha256:d2"]
        r = analyze([_record("H100", dists, digests=digs),
                     _record("A100", dists, digests=digs)])
        assert r.fully_reproducible_pairs == 1
        assert r.pairs[0].bit_identical_clips == 3

    def test_differing_frames_are_not_counted_reproducible(self):
        r = analyze([_record("H100", [0.3, 0.4], digests=["sha256:a", "sha256:b"]),
                     _record("A100", [0.3, 0.4], digests=["sha256:a", "sha256:X"])])
        assert r.fully_reproducible_pairs == 0
        assert r.pairs[0].bit_identical_clips == 1


class TestDeterminism:
    def test_the_analysis_is_reproducible(self):
        rng = np.random.default_rng(5)
        base = rng.uniform(0.2, 0.5, size=40)
        boxes = [_record("H100", base), _record("A100", base + rng.normal(0, 0.01, 40))]
        a = analyze(boxes, seed=7)
        b = analyze(boxes, seed=7)
        assert a.recommended_delta_threshold == b.recommended_delta_threshold
        assert a.max_abs_mu_hat == b.max_abs_mu_hat
