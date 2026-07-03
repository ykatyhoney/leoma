"""Unit tests for sampler-core resolution gating policy."""


class TestIsResolutionAcceptable:
    """Tests for the tolerant 480p resolution check used to gate miner uploads."""

    def test_canonical_832x480_accepted(self):
        from leoma.app.sampler.core import _is_resolution_acceptable
        assert _is_resolution_acceptable(832, 480) is True

    def test_codec_rounded_832x464_accepted(self):
        """User's motivating example: macroblock-rounded height is still 480p."""
        from leoma.app.sampler.core import _is_resolution_acceptable
        assert _is_resolution_acceptable(832, 464) is True

    def test_standard_16x9_854x480_accepted(self):
        from leoma.app.sampler.core import _is_resolution_acceptable
        assert _is_resolution_acceptable(854, 480) is True

    def test_boundary_800x448_accepted(self):
        """Exactly tolerance away on both axes is accepted."""
        from leoma.app.sampler.core import _is_resolution_acceptable
        assert _is_resolution_acceptable(800, 448) is True

    def test_just_past_boundary_rejected(self):
        from leoma.app.sampler.core import _is_resolution_acceptable
        assert _is_resolution_acceptable(799, 448) is False
        assert _is_resolution_acceptable(800, 447) is False

    def test_4x3_640x480_rejected(self):
        from leoma.app.sampler.core import _is_resolution_acceptable
        assert _is_resolution_acceptable(640, 480) is False

    def test_1080p_rejected(self):
        from leoma.app.sampler.core import _is_resolution_acceptable
        assert _is_resolution_acceptable(1920, 1080) is False

    def test_probe_failure_zero_rejected(self):
        """get_video_resolution returns (0, 0) on probe failure — must be rejected."""
        from leoma.app.sampler.core import _is_resolution_acceptable
        assert _is_resolution_acceptable(0, 0) is False


class TestDeterministicOrder:
    """Block-hash-seeded source ordering is deterministic + a full permutation."""

    KEYS = ["a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4"]

    def test_reproducible_and_permutation(self):
        from leoma.app.sampler.core import _deterministic_order
        h = "0x" + "de" * 24
        o1 = _deterministic_order(self.KEYS, h)
        o2 = _deterministic_order(self.KEYS, h)
        assert o1 == o2                          # same (keys, hash) -> same order
        assert sorted(o1) == sorted(self.KEYS)   # every key exactly once (a permutation)

    def test_starts_at_hash_index_and_walks_forward(self):
        from leoma.app.sampler.core import _deterministic_order
        h = "0x" + "ab" * 24
        start = int(h, 16) % len(self.KEYS)
        expected = [self.KEYS[(start + i) % len(self.KEYS)] for i in range(len(self.KEYS))]
        order = _deterministic_order(self.KEYS, h)
        assert order == expected
        assert order[0] == self.KEYS[start]      # the mandated primary source

    def test_empty(self):
        from leoma.app.sampler.core import _deterministic_order
        assert _deterministic_order([], "0x1") == []
