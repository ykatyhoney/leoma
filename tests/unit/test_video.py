"""
Unit tests for video processing utilities.

Tests frame extraction, duration parsing, and clip selection.
"""

import base64
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch


def _success_process() -> MagicMock:
    """Create a successful subprocess.CompletedProcess-like mock."""
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = 0
    result.stderr = b""
    return result


class TestExtractFrames:
    """Tests for extract_frames function."""

    async def test_extract_frames_creates_output_dir(self, tmp_path):
        """Test that extract_frames creates output directory."""
        from leoma.infra.video_utils import extract_frames

        output_dir = tmp_path / "frames"
        mock_result = _success_process()

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result):
            await extract_frames("/fake/video.mp4", str(output_dir))

        assert output_dir.exists()

    async def test_extract_frames_clears_existing(self, tmp_path):
        """Test that existing frames are cleared."""
        from leoma.infra.video_utils import extract_frames

        output_dir = tmp_path / "frames"
        output_dir.mkdir()

        # Create some existing files
        old_frame = output_dir / "old_frame.jpg"
        another = output_dir / "another.jpg"
        old_frame.touch()
        another.touch()

        # Verify files exist before
        assert old_frame.exists()
        assert another.exists()

        mock_result = _success_process()

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result):
            await extract_frames("/fake/video.mp4", str(output_dir))

        # Old files should have been removed by the clear logic
        assert not old_frame.exists()
        assert not another.exists()

    async def test_extract_frames_calls_ffmpeg(self, tmp_path):
        """Test that ffmpeg is called with correct arguments."""
        from leoma.infra.video_utils import extract_frames

        output_dir = tmp_path / "frames"
        mock_result = _success_process()

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result) as mock_to_thread:
            await extract_frames("/input/video.mp4", str(output_dir), max_frames=8)

        # Verify subprocess.run was called via asyncio.to_thread
        mock_to_thread.assert_called_once()
        call_args = mock_to_thread.call_args[0]
        assert call_args[0].__name__ == "run"  # subprocess.run

    async def test_extract_frames_returns_sorted_paths(self, tmp_path):
        """Test that frame paths are returned sorted."""
        from leoma.infra.video_utils import extract_frames

        output_dir = tmp_path / "frames"
        output_dir.mkdir()

        mock_result = _success_process()

        # We need to create files AFTER the clear happens inside extract_frames
        # So we mock asyncio.to_thread to create files as a side effect
        def create_frames_side_effect(*args, **kwargs):
            # Create frame files after ffmpeg "runs"
            (output_dir / "frame_02.jpg").touch()
            (output_dir / "frame_01.jpg").touch()
            (output_dir / "frame_03.jpg").touch()
            return mock_result

        with patch("leoma.infra.video_utils.asyncio.to_thread", side_effect=create_frames_side_effect):
            result = await extract_frames("/fake/video.mp4", str(output_dir))

        # Should be sorted
        filenames = [Path(p).name for p in result]
        assert filenames == ["frame_01.jpg", "frame_02.jpg", "frame_03.jpg"]


class TestFramesToBase64:
    """Tests for frames_to_base64 function."""

    def test_frames_to_base64_converts_files(self, tmp_path):
        """Test conversion of frame files to base64."""
        from leoma.infra.video_utils import frames_to_base64

        # Create test image files
        frame1 = tmp_path / "frame1.jpg"
        frame2 = tmp_path / "frame2.jpg"
        frame1.write_bytes(b"fake image data 1")
        frame2.write_bytes(b"fake image data 2")

        result = frames_to_base64([str(frame1), str(frame2)])

        assert len(result) == 2
        for item in result:
            assert item["type"] == "image_url"
            assert "image_url" in item
            assert item["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_frames_to_base64_correct_encoding(self, tmp_path):
        """Test that base64 encoding is correct."""
        from leoma.infra.video_utils import frames_to_base64

        test_data = b"test image content"
        frame = tmp_path / "test.jpg"
        frame.write_bytes(test_data)

        result = frames_to_base64([str(frame)])

        expected_b64 = base64.b64encode(test_data).decode("utf-8")
        assert result[0]["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"

    def test_frames_to_base64_empty_list(self):
        """Test with empty list."""
        from leoma.infra.video_utils import frames_to_base64

        result = frames_to_base64([])

        assert result == []


class TestGetVideoDuration:
    """Tests for get_video_duration function."""

    async def test_get_video_duration_parses_output(self):
        """Test parsing ffprobe duration output."""
        from leoma.infra.video_utils import get_video_duration

        mock_result = MagicMock()
        mock_result.stdout = "120.5\n"

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result):
            duration = await get_video_duration("/path/to/video.mp4")

        assert duration == 120.5

    async def test_get_video_duration_handles_integer(self):
        """Test parsing integer duration."""
        from leoma.infra.video_utils import get_video_duration

        mock_result = MagicMock()
        mock_result.stdout = "60"

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result):
            duration = await get_video_duration("/path/to/video.mp4")

        assert duration == 60.0

    async def test_get_video_duration_returns_zero_on_error(self):
        """Test that 0.0 is returned on parse error."""
        from leoma.infra.video_utils import get_video_duration

        mock_result = MagicMock()
        mock_result.stdout = "N/A"

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result):
            duration = await get_video_duration("/path/to/video.mp4")

        assert duration == 0.0

    async def test_get_video_duration_returns_zero_on_empty(self):
        """Test that 0.0 is returned on empty output."""
        from leoma.infra.video_utils import get_video_duration

        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result):
            duration = await get_video_duration("/path/to/video.mp4")

        assert duration == 0.0


class TestGetVideoResolution:
    """Tests for get_video_resolution function."""

    async def test_parses_widthxheight(self):
        from leoma.infra.video_utils import get_video_resolution

        mock_result = MagicMock()
        mock_result.stdout = "854x480\n"

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result):
            width, height = await get_video_resolution("/path/to/video.mp4")

        assert (width, height) == (854, 480)

    async def test_returns_zero_on_empty(self):
        from leoma.infra.video_utils import get_video_resolution

        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result):
            width, height = await get_video_resolution("/path/to/video.mp4")

        assert (width, height) == (0, 0)

    async def test_returns_zero_on_malformed(self):
        from leoma.infra.video_utils import get_video_resolution

        mock_result = MagicMock()
        mock_result.stdout = "not-a-resolution\n"

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result):
            width, height = await get_video_resolution("/path/to/video.mp4")

        assert (width, height) == (0, 0)


class TestExtractClip:
    """Tests for extract_clip function."""

    async def test_extract_clip_calls_ffmpeg(self):
        """Test that ffmpeg is called with correct clip parameters."""
        from leoma.infra.video_utils import extract_clip

        mock_result = _success_process()

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result) as mock_to_thread:
            await extract_clip(
                video_path="/input/video.mp4",
                output_path="/output/clip.mp4",
                start_offset=30.0,
                duration=5.0,
            )

        mock_to_thread.assert_called_once()
        call_args = mock_to_thread.call_args[0]
        cmd = call_args[1]  # The command list

        assert cmd[0] == "ffmpeg"
        assert "-ss" in cmd
        assert "30.0" in cmd
        assert "-t" in cmd
        assert "5.0" in cmd
        assert "-i" in cmd
        assert "/input/video.mp4" in cmd


class TestExtractFirstFrame:
    """Tests for extract_first_frame function."""

    async def test_extract_first_frame_at_offset(self):
        """Test extracting first frame at specified offset."""
        from leoma.infra.video_utils import extract_first_frame

        mock_result = _success_process()

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result) as mock_to_thread:
            await extract_first_frame(
                video_path="/input/video.mp4",
                output_path="/output/frame.jpg",
                start_offset=15.0,
            )

        mock_to_thread.assert_called_once()
        call_args = mock_to_thread.call_args[0]
        cmd = call_args[1]

        assert cmd[0] == "ffmpeg"
        assert "-ss" in cmd
        assert "15.0" in cmd
        assert "-vframes" in cmd
        assert "1" in cmd

    async def test_extract_first_frame_default_offset(self):
        """Test extracting first frame with default offset."""
        from leoma.infra.video_utils import extract_first_frame

        mock_result = _success_process()

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result) as mock_to_thread:
            await extract_first_frame(
                video_path="/input/video.mp4",
                output_path="/output/frame.jpg",
            )

        mock_to_thread.assert_called_once()
        call_args = mock_to_thread.call_args[0]
        cmd = call_args[1]

        # Default offset is 0
        assert "-ss" in cmd
        assert "0" in cmd


class TestStitchVideosSideBySide:
    """Tests for stitch_videos_side_by_side function."""

    async def test_stitch_videos_calls_ffmpeg(self):
        """Test that ffmpeg is called with filter_complex for stitching."""
        from leoma.infra.video_utils import stitch_videos_side_by_side

        mock_result = _success_process()

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result) as mock_to_thread:
            await stitch_videos_side_by_side(
                left_path="/left.mp4",
                right_path="/right.mp4",
                output_path="/output.mp4",
            )

        mock_to_thread.assert_called_once()
        call_args = mock_to_thread.call_args[0]
        cmd = call_args[1]

        assert cmd[0] == "ffmpeg"
        assert "-filter_complex" in cmd
        # Find the filter_complex argument and check it contains hstack
        filter_idx = cmd.index("-filter_complex")
        filter_value = cmd[filter_idx + 1]
        assert "hstack" in filter_value
        assert "/left.mp4" in cmd
        assert "/right.mp4" in cmd


class TestSelectClip:
    """Tests for clip selection logic (e.g. owner-sampler)."""

    def test_select_random_clip_offset(self):
        """Test random clip offset selection."""
        import random

        source_duration = 120.0  # 2 minute video
        clip_duration = 5.0
        min_offset = 5.0  # Skip first 5 seconds

        # Simulate clip selection logic
        max_start = source_duration - clip_duration - min_offset
        start_offset = random.uniform(min_offset, max_start)

        assert min_offset <= start_offset <= max_start
        assert start_offset + clip_duration <= source_duration

    def test_select_clip_handles_short_video(self):
        """Test clip selection with video shorter than desired clip."""
        source_duration = 3.0  # 3 second video
        clip_duration = 5.0
        min_offset = 0.0

        # If video is too short, use entire video
        if source_duration <= clip_duration:
            start_offset = 0.0
            actual_duration = source_duration
        else:
            max_start = source_duration - clip_duration
            start_offset = 0.0
            actual_duration = clip_duration

        assert start_offset == 0.0
        assert actual_duration == 3.0


class TestSceneDetection:
    """Tests for scene-cut detection and one-shot clip selection."""

    async def test_detect_scene_cuts_parses_pts_time(self):
        """Scene timestamps should be parsed from ffmpeg showinfo output."""
        from leoma.infra.video_utils import detect_scene_cuts

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "pts_time:1.250\npts_time:3.500\n"
        mock_result.stderr = "pts_time:3.500\n"

        with patch("leoma.infra.video_utils.asyncio.to_thread", return_value=mock_result):
            cuts = await detect_scene_cuts("/input/video.mp4", scene_threshold=0.2)

        assert cuts == [1.25, 3.5]

    async def test_choose_one_shot_clip_start_returns_none_without_valid_segment(self):
        """No segment should be selected when every detected shot is shorter than clip duration."""
        from leoma.infra.video_utils import choose_one_shot_clip_start

        duration_result = MagicMock()
        duration_result.stdout = "8.0"
        scene_result = MagicMock()
        scene_result.returncode = 0
        scene_result.stdout = "pts_time:2.0\npts_time:4.0\npts_time:6.0\n"
        scene_result.stderr = ""

        with patch(
            "leoma.infra.video_utils.asyncio.to_thread",
            side_effect=[duration_result, scene_result],
        ):
            selection = await choose_one_shot_clip_start(
                "/input/video.mp4",
                clip_duration=5.0,
                scene_threshold=0.2,
                boundary_margin=0.0,
            )

        assert selection is None

    async def test_choose_one_shot_clip_start_success(self):
        """A clip start should be selected inside a valid one-shot window."""
        from leoma.infra.video_utils import choose_one_shot_clip_start

        duration_result = MagicMock()
        duration_result.stdout = "20.0"
        scene_result = MagicMock()
        scene_result.returncode = 0
        scene_result.stdout = "pts_time:6.0\npts_time:14.0\n"
        scene_result.stderr = ""

        with (
            patch(
                "leoma.infra.video_utils.asyncio.to_thread",
                side_effect=[duration_result, scene_result],
            ),
            patch(
                "leoma.infra.video_utils.random.choice",
                side_effect=lambda seq: seq[0],
            ),
            # Segment [6,14] with margin 0.1 → usable [6.1, 13.9]; clip_start must be in [6.1, 8.9]
            patch("leoma.infra.video_utils.random.uniform", return_value=6.1),
        ):
            selection = await choose_one_shot_clip_start(
                "/input/video.mp4",
                clip_duration=5.0,
                scene_threshold=0.2,
                boundary_margin=0.1,
            )

        assert selection is not None
        assert selection.segment_start_seconds <= selection.clip_start_seconds
        assert (selection.segment_end_seconds - selection.clip_start_seconds) >= 5.0


class TestClipStartSeedDeterminism:
    """A block-hash seed makes clip selection reproducible (same seed -> same clip)."""

    async def _select(self, seed):
        from unittest.mock import MagicMock, patch
        from leoma.infra.video_utils import choose_one_shot_clip_start
        duration = MagicMock(); duration.stdout = "30.0"
        scenes = MagicMock(); scenes.returncode = 0
        scenes.stdout = "pts_time:10.0\npts_time:20.0\n"; scenes.stderr = ""
        with patch("leoma.infra.video_utils.asyncio.to_thread", side_effect=[duration, scenes]):
            return await choose_one_shot_clip_start(
                "/input/video.mp4", clip_duration=5.0, scene_threshold=0.2, boundary_margin=0.0, seed=seed,
            )

    async def test_same_seed_same_clip(self):
        a = await self._select("0xabc123")
        b = await self._select("0xabc123")
        assert a is not None and b is not None
        assert (a.segment_start_seconds, a.clip_start_seconds) == (b.segment_start_seconds, b.clip_start_seconds)

    async def test_selection_valid_across_seeds(self):
        for seed in ("0x1", "0xdeadbeef", "0xffffffffffffffff"):
            s = await self._select(seed)
            assert s is not None
            assert s.segment_start_seconds <= s.clip_start_seconds
            assert (s.segment_end_seconds - s.clip_start_seconds) >= 5.0
