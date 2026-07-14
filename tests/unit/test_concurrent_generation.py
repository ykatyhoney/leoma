"""Concurrent king/challenger generation.

The only thing ``concurrent=True`` may change is *when* the two generations happen,
never *what* they compute. The single most important test in this file is
``test_concurrent_and_sequential_produce_byte_identical_verdicts`` — if that one ever
fails, the throughput change has become a consensus bug.
"""

import threading
import time

import numpy as np
import pytest

from leoma.eval.metrics import mse
from leoma.eval.video_runner import Clip, GenParams, run_duel

PARAMS = GenParams(num_frames=4, fps=2, width=8, height=8)
SLEEP = 0.06  # generous relative to scheduling jitter, small enough for a fast test


def _clips(n=6, seed=0):
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(n):
        truth = rng.integers(0, 255, size=(4, 8, 8, 3)).astype("uint8")
        clips.append(Clip(clip_index=i, clip_id=f"clip-{i:04d}", first_frame=truth[0],
                          prompt="p", truth_frames=truth, params=PARAMS))
    return clips


def _slow_generate(offset: int):
    """A deterministic generator that also takes real wall-clock time, so the two
    duelists' generations can genuinely overlap in a thread pool."""
    def gen(clip: Clip, seed: int) -> np.ndarray:
        time.sleep(SLEEP)
        return np.clip(clip.truth_frames.astype(int) + offset + (seed % 7), 0, 255).astype("uint8")
    return gen


def _run(clips, concurrent, seed=42):
    return run_duel(
        clips, generate_king=_slow_generate(3), generate_challenger=_slow_generate(5),
        distance_fn=mse, master_seed=seed, delta_threshold=0.0025, alpha=0.05,
        n_bootstrap=200, concurrent=concurrent,
    )


class TestByteIdenticalRegardlessOfConcurrency:
    """THE property this feature must never violate."""

    def test_concurrent_and_sequential_produce_byte_identical_verdicts(self):
        clips = _clips()
        sequential = _run(clips, concurrent=False)
        concurrent = _run(clips, concurrent=True)

        assert sequential["accepted"] == concurrent["accepted"]
        assert sequential["mu_hat"] == concurrent["mu_hat"]
        assert sequential["lcb"] == concurrent["lcb"]
        assert sequential["avg_king_distance"] == concurrent["avg_king_distance"]
        assert sequential["avg_challenger_distance"] == concurrent["avg_challenger_distance"]

        for a, b in zip(sequential["per_clip"], concurrent["per_clip"]):
            assert a["king_distance"] == b["king_distance"]
            assert a["challenger_distance"] == b["challenger_distance"]
            assert a["king_frames_digest"] == b["king_frames_digest"]
            assert a["challenger_frames_digest"] == b["challenger_frames_digest"]
            assert a["gen_seed"] == b["gen_seed"]

    def test_the_per_clip_seed_pairing_is_unaffected_by_concurrency(self):
        seen = {"king": [], "chall": []}

        def gk(clip, seed):
            seen["king"].append((clip.clip_index, seed))
            return np.clip(clip.truth_frames.astype(int) + 1, 0, 255).astype("uint8")

        def gc(clip, seed):
            seen["chall"].append((clip.clip_index, seed))
            return np.clip(clip.truth_frames.astype(int) + 2, 0, 255).astype("uint8")

        run_duel(_clips(n=4), gk, gc, mse, master_seed=7, delta_threshold=0.0025,
                 alpha=0.05, n_bootstrap=100, concurrent=True)
        assert seen["king"] == seen["chall"]


class TestActualOverlap:
    def test_concurrent_is_meaningfully_faster_than_sequential(self):
        """time.sleep() releases the GIL, so two threads sleeping in parallel genuinely
        overlap — this is a reliable way to prove concurrency without a real GPU."""
        clips = _clips(n=8)

        t0 = time.monotonic()
        _run(clips, concurrent=False)
        sequential_elapsed = time.monotonic() - t0

        t0 = time.monotonic()
        _run(clips, concurrent=True)
        concurrent_elapsed = time.monotonic() - t0

        # Sequential: ~2 * n * SLEEP. Concurrent: ~1 * n * SLEEP. A 1.5x margin is
        # comfortably below the theoretical 2x speedup, avoiding CI flakiness.
        assert concurrent_elapsed < sequential_elapsed / 1.5, (
            f"concurrent ({concurrent_elapsed:.3f}s) was not meaningfully faster than "
            f"sequential ({sequential_elapsed:.3f}s)"
        )


class TestFailureModes:
    def test_a_king_side_exception_propagates(self):
        def boom(clip, seed):
            raise RuntimeError("king pipeline OOM")

        with pytest.raises(RuntimeError, match="king pipeline OOM"):
            run_duel(_clips(n=2), boom, _slow_generate(0), mse, master_seed=1,
                     delta_threshold=0.0025, alpha=0.05, n_bootstrap=50, concurrent=True)

    def test_a_challenger_side_exception_propagates(self):
        def boom(clip, seed):
            raise RuntimeError("challenger pipeline OOM")

        with pytest.raises(RuntimeError, match="challenger pipeline OOM"):
            run_duel(_clips(n=2), _slow_generate(0), boom, mse, master_seed=1,
                     delta_threshold=0.0025, alpha=0.05, n_bootstrap=50, concurrent=True)

    def test_cancellation_between_clips_still_works_when_concurrent(self):
        from leoma.eval.errors import DuelCancelled

        calls = {"n": 0}

        def cancel_after_one():
            calls["n"] += 1
            return calls["n"] > 1

        with pytest.raises(DuelCancelled):
            run_duel(_clips(n=5), _slow_generate(0), _slow_generate(1), mse, master_seed=1,
                     delta_threshold=0.0025, alpha=0.05, n_bootstrap=50,
                     concurrent=True, should_cancel=cancel_after_one)

    def test_a_degenerate_challenger_is_still_caught_after_concurrent_generation(self):
        from leoma.eval.errors import DegenerateGeneration

        def one_frame(clip, seed):
            return np.zeros((1, 8, 8, 3), dtype="uint8")

        with pytest.raises(DegenerateGeneration):
            run_duel(_clips(n=2), _slow_generate(0), one_frame, mse, master_seed=1,
                     delta_threshold=0.0025, alpha=0.05, n_bootstrap=50, concurrent=True)


class TestNoThreadLeak:
    def test_the_executor_is_torn_down_after_every_run(self):
        baseline = threading.active_count()
        for _ in range(5):
            _run(_clips(n=2), concurrent=True)
        # Give any lingering worker threads a moment to actually exit.
        for _ in range(20):
            if threading.active_count() <= baseline:
                break
            time.sleep(0.02)
        assert threading.active_count() <= baseline

    def test_torn_down_even_when_the_duel_raises(self):
        baseline = threading.active_count()

        def boom(clip, seed):
            raise RuntimeError("x")

        for _ in range(5):
            with pytest.raises(RuntimeError):
                run_duel(_clips(n=2), boom, _slow_generate(0), mse, master_seed=1,
                         delta_threshold=0.0025, alpha=0.05, n_bootstrap=50, concurrent=True)

        for _ in range(20):
            if threading.active_count() <= baseline:
                break
            time.sleep(0.02)
        assert threading.active_count() <= baseline


class TestSequentialIsUnaffected:
    """concurrent defaults to False — every existing caller's behavior is untouched."""

    def test_default_is_sequential(self):
        clips = _clips(n=2)
        v = run_duel(clips, _slow_generate(0), _slow_generate(1), mse, master_seed=1,
                     delta_threshold=0.0025, alpha=0.05, n_bootstrap=50)
        assert v["per_clip"][0]["king_distance"] is not None  # ran to completion normally
