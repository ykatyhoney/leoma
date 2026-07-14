"""The raw-RGB decode path and the canonical digests it feeds.

These run the REAL ffmpeg (it is a hard dependency of an eval box, and stubbing it
would test nothing about the thing that actually went wrong). Two bugs are pinned:

* **frame order broke at 100 frames.** The old path wrote ``frame_%02d.jpg`` and
  then `sorted()` the filenames, so ``frame_100`` sorted before ``frame_11``. Any
  duel longer than 99 frames — and the pinned config is **81**, uncomfortably close
  — silently scored against a shuffled ground truth.
* **the truth was lossy twice** (x264 re-encode, then JPEG), so the "real
  continuation" every distance is measured against was a compressed approximation
  of itself.
"""

import subprocess

import numpy as np
import pytest

from leoma.eval.digests import (
    canonical_float,
    canonical_json,
    digest_frames,
    digest_obj,
    sha256_bytes,
)
from leoma.infra.video_utils import FFmpegError, decode_frames_rgb, motion_energy


@pytest.fixture(scope="module")
def moving_video(tmp_path_factory):
    """A 150-frame test clip whose frame N is *identifiable* from its pixels.

    Each frame is a solid grey ramp: frame N has luma ≈ N. That makes frame ORDER
    directly observable in the decoded array — which is what the 100-frame sort bug
    silently destroyed.
    """
    path = tmp_path_factory.mktemp("video") / "ramp.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:r=25:d=6",
            "-vf", "geq=lum='clip(N,0,255)':cb=128:cr=128",
            "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True, capture_output=True,
    )
    return str(path)


class TestDecodeFramesRgb:
    def test_returns_exactly_the_requested_shape(self, moving_video):
        frames = decode_frames_rgb(
            moving_video, start_seconds=0.0, duration_seconds=2.0,
            fps=25, num_frames=40, width=32, height=16,
        )
        assert frames.shape == (40, 16, 32, 3)
        assert frames.dtype == np.uint8

    def test_frame_order_survives_past_100_frames(self, moving_video):
        """THE bug. frame_%02d.jpg + sorted() put frame_100 before frame_11, so any
        duel longer than 99 frames scored against a shuffled truth. The pinned
        config is 81 frames — one config bump away from silent corruption."""
        frames = decode_frames_rgb(
            moving_video, start_seconds=0.0, duration_seconds=6.0,
            fps=25, num_frames=120, width=32, height=32,
        )
        assert frames.shape[0] == 120

        # Luma rises monotonically with frame index, so the decoded order must too.
        luma = frames.reshape(120, -1).mean(axis=1)
        assert np.all(np.diff(luma) >= -1.0), "frames are not in temporal order"
        assert luma[110] > luma[10], "frame 110 should be brighter than frame 10"

    def test_decoding_is_reproducible_on_this_box(self, moving_video):
        """Run-to-run determinism is what makes truth_sha256 a usable pin."""
        kw = dict(start_seconds=1.0, duration_seconds=1.0, fps=16, num_frames=16,
                  width=32, height=32)
        a = decode_frames_rgb(moving_video, **kw)
        b = decode_frames_rgb(moving_video, **kw)
        assert digest_frames(a) == digest_frames(b)

    def test_a_window_past_the_end_of_the_video_raises(self, moving_video):
        """It must NOT quietly return a short array — that is a short ground truth,
        which is the freeze cheat's other door."""
        with pytest.raises(FFmpegError, match="decoded .* frames, need"):
            decode_frames_rgb(
                moving_video, start_seconds=5.5, duration_seconds=5.0,
                fps=25, num_frames=125, width=32, height=32,
            )

    def test_motion_energy_separates_moving_from_static(self, moving_video):
        moving = decode_frames_rgb(
            moving_video, start_seconds=0.0, duration_seconds=1.0,
            fps=16, num_frames=16, width=32, height=32,
        )
        frozen = np.repeat(moving[:1], 16, axis=0)
        assert motion_energy(moving) > motion_energy(frozen)
        assert motion_energy(frozen) == 0.0


class TestCanonicalDigests:
    def test_key_order_does_not_change_the_digest(self):
        assert digest_obj({"a": 1, "b": 2}) == digest_obj({"b": 2, "a": 1})

    def test_float_noise_below_the_precision_floor_does_not_change_the_digest(self):
        """The spec crosses TOML -> JSON -> HTTP -> JSON. A last-bit difference in a
        float must not flip a consensus digest and refuse every duel on the subnet."""
        assert digest_obj({"delta": 0.0025}) == digest_obj({"delta": 0.0025 + 1e-15})

    def test_a_real_difference_does_change_the_digest(self):
        assert digest_obj({"delta": 0.0025}) != digest_obj({"delta": 0.0026})

    def test_negative_zero_normalizes(self):
        assert canonical_float(-0.0) == canonical_float(0.0)
        assert digest_obj({"x": -0.0}) == digest_obj({"x": 0.0})

    def test_nan_is_refused_rather_than_hashed(self):
        # A NaN in a consensus digest would hash fine and compare unequal to itself
        # everywhere downstream. Refuse it at the door.
        with pytest.raises(ValueError):
            canonical_json({"x": float("nan")})

    def test_frames_of_the_same_bytes_but_a_different_shape_do_not_collide(self):
        a = np.arange(24, dtype="uint8").reshape(2, 2, 2, 3)
        b = np.arange(24, dtype="uint8").reshape(1, 4, 2, 3)
        assert a.tobytes() == b.tobytes()
        assert digest_frames(a) != digest_frames(b)

    def test_empty_bytes_digest_is_stable(self):
        from leoma.eval.digests import EMPTY

        assert sha256_bytes(b"") == EMPTY
