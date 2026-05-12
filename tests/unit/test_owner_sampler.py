"""Unit tests for owner-sampler resolution gating policy."""


class TestIsResolutionAcceptable:
    """Tests for the tolerant 480p resolution check used to gate miner uploads."""

    def test_canonical_832x480_accepted(self):
        from leoma.app.owner_sampler.main import _is_resolution_acceptable
        assert _is_resolution_acceptable(832, 480) is True

    def test_codec_rounded_832x464_accepted(self):
        """User's motivating example: macroblock-rounded height is still 480p."""
        from leoma.app.owner_sampler.main import _is_resolution_acceptable
        assert _is_resolution_acceptable(832, 464) is True

    def test_standard_16x9_854x480_accepted(self):
        from leoma.app.owner_sampler.main import _is_resolution_acceptable
        assert _is_resolution_acceptable(854, 480) is True

    def test_boundary_800x448_accepted(self):
        """Exactly tolerance away on both axes is accepted."""
        from leoma.app.owner_sampler.main import _is_resolution_acceptable
        assert _is_resolution_acceptable(800, 448) is True

    def test_just_past_boundary_rejected(self):
        from leoma.app.owner_sampler.main import _is_resolution_acceptable
        assert _is_resolution_acceptable(799, 448) is False
        assert _is_resolution_acceptable(800, 447) is False

    def test_4x3_640x480_rejected(self):
        from leoma.app.owner_sampler.main import _is_resolution_acceptable
        assert _is_resolution_acceptable(640, 480) is False

    def test_1080p_rejected(self):
        from leoma.app.owner_sampler.main import _is_resolution_acceptable
        assert _is_resolution_acceptable(1920, 1080) is False

    def test_probe_failure_zero_rejected(self):
        """get_video_resolution returns (0, 0) on probe failure — must be rejected."""
        from leoma.app.owner_sampler.main import _is_resolution_acceptable
        assert _is_resolution_acceptable(0, 0) is False
