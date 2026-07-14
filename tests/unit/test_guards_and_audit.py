"""Degenerate-output rejection, full-width seeds, and the verdict's audit trail."""

import numpy as np
import pytest

from leoma.app.validator.seeds import TORCH_SEED_MASK, clip_generation_seed, torch_seed
from leoma.eval.errors import DegenerateGeneration
from leoma.eval.guards import validate_generation
from leoma.eval.metrics import mse
from leoma.eval.video_runner import Clip, GenParams, run_duel

PARAMS = GenParams(num_frames=4, fps=2, width=8, height=8)


def _clips(n=6):
    rng = np.random.default_rng(0)
    clips = []
    for i in range(n):
        truth = rng.integers(0, 255, size=(4, 8, 8, 3)).astype("uint8")
        clips.append(Clip(clip_index=i, clip_id=f"clip-{i:04d}", first_frame=truth[0],
                          prompt="p", truth_frames=truth, params=PARAMS))
    return clips


def _near(clip, seed):
    return np.clip(clip.truth_frames.astype(float) + 3, 0, 255).astype("uint8")


def _far(clip, seed):
    return np.clip(clip.truth_frames.astype(float) + 90, 0, 255).astype("uint8")


class TestValidateGeneration:
    """Turns "unscoreable" into a typed, deterministic rejection.

    Without this, a degenerate generation surfaces as a bare ValueError from deep
    inside a metric — which the validator classifies as a generic duel error and
    RETRIES four times, for output that will be exactly as broken every time.
    """

    def test_a_good_generation_passes_through_as_uint8(self):
        clip = _clips(1)[0]
        out = validate_generation(_near(clip, 0), expected_frames=4, width=8, height=8)
        assert out.shape == (4, 8, 8, 3)
        assert out.dtype == np.uint8

    def test_a_short_generation_is_rejected(self):
        with pytest.raises(DegenerateGeneration, match="fewer than"):
            validate_generation(
                np.zeros((1, 8, 8, 3), dtype="uint8"), expected_frames=4, width=8, height=8
            )

    def test_the_rejection_explains_the_cheat(self):
        """The message is miner-facing: it must say WHY, or the miner assumes a bug."""
        with pytest.raises(DegenerateGeneration, match="conditioning frame"):
            validate_generation(
                np.zeros((2, 8, 8, 3), dtype="uint8"), expected_frames=4, width=8, height=8
            )

    def test_wrong_resolution_is_rejected(self):
        with pytest.raises(DegenerateGeneration, match="pinned to 8x8"):
            validate_generation(
                np.zeros((4, 16, 16, 3), dtype="uint8"), expected_frames=4, width=8, height=8
            )

    def test_wrong_rank_is_rejected(self):
        with pytest.raises(DegenerateGeneration, match="shape"):
            validate_generation(np.zeros((4, 8, 8), dtype="uint8"),
                                expected_frames=4, width=8, height=8)

    def test_nan_output_is_rejected(self):
        """A NaN distance poisons the bootstrap rather than losing the duel — the
        challenger would not lose, the whole verdict would be garbage."""
        frames = np.zeros((4, 8, 8, 3), dtype="float32")
        frames[0, 0, 0, 0] = np.nan
        with pytest.raises(DegenerateGeneration, match="non-finite"):
            validate_generation(frames, expected_frames=4, width=8, height=8)

    def test_a_longer_generation_is_truncated_not_rejected(self):
        out = validate_generation(
            np.zeros((9, 8, 8, 3), dtype="uint8"), expected_frames=4, width=8, height=8
        )
        assert out.shape[0] == 4

    def test_a_degenerate_challenger_stops_the_duel(self):
        """Rejection is deterministic and identical on every validator, so it is part
        of consensus rather than a local accident."""
        with pytest.raises(DegenerateGeneration):
            run_duel(
                _clips(), generate_king=_near,
                generate_challenger=lambda c, s: np.zeros((1, 8, 8, 3), dtype="uint8"),
                distance_fn=mse, master_seed=1, delta_threshold=0.0025, alpha=0.001,
                n_bootstrap=50,
            )


class TestTorchSeed:
    def test_keeps_63_bits_not_31(self):
        """The generation path masked with 0x7FFFFFFF, throwing away half of the 64
        bits blake2b produced and shrinking the noise space to 2^31."""
        big = (1 << 62) | 12345
        assert torch_seed(big) == big
        assert torch_seed(big) > 0x7FFFFFFF

    def test_stays_inside_torchs_accepted_range(self):
        for i in range(64):
            assert 0 <= torch_seed(clip_generation_seed(i, i)) <= TORCH_SEED_MASK

    def test_is_deterministic(self):
        assert torch_seed(clip_generation_seed(7, 3)) == torch_seed(clip_generation_seed(7, 3))


class TestPerClipAudit:
    """What a validator needs to localize a disagreement instead of arguing about it."""

    def test_every_clip_records_its_id_seed_and_frame_digests(self):
        v = run_duel(_clips(), _far, _near, mse, master_seed=42, delta_threshold=0.0025,
                     alpha=0.001, n_bootstrap=100)
        row = v["per_clip"][0]
        assert row["clip_id"] == "clip-0000"
        assert row["gen_seed"] == clip_generation_seed(42, 0)
        assert row["king_frames_digest"].startswith("sha256:")
        assert row["challenger_frames_digest"] != row["king_frames_digest"]

    def test_frame_digests_separate_generation_noise_from_a_broken_scorer(self):
        """If two validators' distances differ but their frame digests MATCH, the
        generations were identical and the scoring diverged — a broken box, not GPU
        noise. That distinction is the difference between a bug hunt and a shrug."""
        a = run_duel(_clips(), _far, _near, mse, master_seed=42, delta_threshold=0.0025,
                     alpha=0.001, n_bootstrap=100)
        b = run_duel(_clips(), _far, _near, mse, master_seed=42, delta_threshold=0.0025,
                     alpha=0.001, n_bootstrap=100)
        assert [c["king_frames_digest"] for c in a["per_clip"]] == \
               [c["king_frames_digest"] for c in b["per_clip"]]

    def test_the_verdict_carries_no_wall_clock(self):
        """A timestamp inside the verdict would make two agreeing validators produce
        different verdict bytes — and so different verdict_digests, forever."""
        v = run_duel(_clips(), _far, _near, mse, master_seed=1, delta_threshold=0.0025,
                     alpha=0.001, n_bootstrap=50)
        assert "timestamp" not in v
