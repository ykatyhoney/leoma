"""
Unit tests for evaluation utilities.

Tests description generation (Gemini full-clip, sampler side) and
generated-video scoring (Gemini, validator side) prompt building and response parsing.
"""

import base64
import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_video(tmp_path):
    """A real on-disk file so _video_to_gemini_part can read it (Gemini is mocked)."""
    path = tmp_path / "gen.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    return str(path)

# A tiny valid JPEG-shaped byte string, base64-encoded so _frame_dict_to_gemini_part
# can decode it without raising.
_FAKE_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")
_FAKE_FRAME = {
    "type": "image_url",
    "image_url": {"url": f"data:image/jpeg;base64,{_FAKE_JPEG_B64}"},
}


def _pass_rate(passed_count: int, total: int) -> float:
    """Compute pass rate with zero-total guard."""
    return passed_count / total if total > 0 else 0.0


class TestGetDescriptionAsync:
    """Tests for get_description_async function (Gemini, full-clip video)."""

    async def test_get_description_returns_text(self, mock_gemini_client, fake_video):
        """Test that get_description_async returns description text."""
        from leoma.infra.judge import get_description_async

        mock_gemini_client.aio.models.generate_content.return_value = MagicMock(
            text="A skateboarder crossing a quiet plaza."
        )

        result = await get_description_async(mock_gemini_client, fake_video)

        assert result == "A skateboarder crossing a quiet plaza."

    async def test_get_description_strips_whitespace(self, mock_gemini_client, fake_video):
        """Test that description is stripped of whitespace."""
        from leoma.infra.judge import get_description_async

        mock_gemini_client.aio.models.generate_content.return_value = MagicMock(
            text="  Trimmed text  \n"
        )

        result = await get_description_async(mock_gemini_client, fake_video)

        assert result == "Trimmed text"

    async def test_get_description_uses_full_clip_video(self, mock_gemini_client, fake_video):
        """The full clip is sent as a video part to the configured Gemini model."""
        from leoma.infra import judge
        from leoma.infra.judge import get_description_async

        mock_gemini_client.aio.models.generate_content.return_value = MagicMock(
            text="Description"
        )

        await get_description_async(mock_gemini_client, fake_video)

        mock_gemini_client.aio.models.generate_content.assert_called_once()
        call_kwargs = mock_gemini_client.aio.models.generate_content.call_args.kwargs

        assert call_kwargs["model"] == judge.GEMINI_DESCRIPTION_MODEL

        # Contents = prompt text + label + a single video part (the full clip).
        contents = call_kwargs["contents"]
        assert judge.DESCRIPTION_PROMPT in contents
        video_parts = [c for c in contents if hasattr(c, "inline_data") and c.inline_data]
        assert len(video_parts) == 1
        assert video_parts[0].inline_data.mime_type == "video/mp4"

    async def test_get_description_requires_gemini_client(self):
        """Calling without a gemini_client should raise ValueError."""
        from leoma.infra.judge import get_description_async

        with pytest.raises(ValueError):
            await get_description_async(None, "clip.mp4")


class TestCalculatePassRate:
    """Tests for pass rate calculation logic."""

    def test_calculate_pass_rate_basic(self):
        """Test basic pass rate calculation."""
        # This is inline in the DAO, but we test the logic here
        total = 20
        passed_count = 15
        pass_rate = _pass_rate(passed_count, total)

        assert pass_rate == 0.75

    def test_calculate_pass_rate_zero_total(self):
        """Test pass rate with zero total samples."""
        total = 0
        passed_count = 0
        pass_rate = _pass_rate(passed_count, total)

        assert pass_rate == 0.0

    def test_calculate_pass_rate_all_passes(self):
        """Test pass rate with all passes."""
        total = 10
        passed_count = 10
        pass_rate = _pass_rate(passed_count, total)

        assert pass_rate == 1.0

    def test_calculate_pass_rate_no_passes(self):
        """Test pass rate with no passes."""
        total = 10
        passed_count = 0
        pass_rate = _pass_rate(passed_count, total)

        assert pass_rate == 0.0


class TestEvaluateGeneratedVideoAsync:
    """Tests for single-video benchmark evaluation (Gemini only, full video input)."""

    async def test_evaluate_generated_video_passes(self, mock_gemini_client, fake_video):
        """High aspect scores should pass benchmark threshold (Gemini path)."""
        from leoma.infra.judge import evaluate_generated_video_async

        response_json = {
            "overall_score": 86,
            "confidence": 89,
            "aspect_scores": {
                "first_frame_fidelity": 88,
                "prompt_adherence": 84,
                "motion_quality": 82,
                "temporal_consistency": 85,
                "visual_quality": 87,
                "camera_composition": 80,
            },
            "major_issues": ["minor blur"],
            "strengths": ["good temporal coherence"],
            "reasoning": "Strong adherence to prompt and stable motion.",
        }
        mock_gemini_client.aio.models.generate_content.return_value = MagicMock(
            text=json.dumps(response_json)
        )

        result = await evaluate_generated_video_async(
            first_frame=[_FAKE_FRAME],
            generated_video_path=fake_video,
            prompt="A person jogging in a city park at sunrise",
            gemini_client=mock_gemini_client,
        )

        assert result["passed"] is True
        assert result["overall_score"] == 86
        assert result["confidence"] == 89

    async def test_evaluate_generated_video_fails_on_critical_floor(
        self, mock_gemini_client, fake_video
    ):
        """Low critical aspect should fail even with decent overall score."""
        from leoma.infra.judge import evaluate_generated_video_async

        response_json = {
            "overall_score": 78,
            "confidence": 80,
            "aspect_scores": {
                "first_frame_fidelity": 42,
                "prompt_adherence": 84,
                "motion_quality": 82,
                "temporal_consistency": 85,
                "visual_quality": 87,
                "camera_composition": 80,
            },
            "major_issues": ["identity drift from conditioning frame"],
            "strengths": [],
            "reasoning": "Good quality overall but misses conditioning constraints.",
        }
        mock_gemini_client.aio.models.generate_content.return_value = MagicMock(
            text=json.dumps(response_json)
        )

        result = await evaluate_generated_video_async(
            first_frame=[],
            generated_video_path=fake_video,
            prompt="Test prompt",
            gemini_client=mock_gemini_client,
        )

        assert result["passed"] is False
        assert result["overall_score"] == 78

    async def test_evaluate_generated_video_handles_parse_error(
        self, mock_gemini_client, fake_video
    ):
        """Invalid JSON should return fail-safe output."""
        from leoma.infra.judge import evaluate_generated_video_async

        mock_gemini_client.aio.models.generate_content.return_value = MagicMock(
            text="not-json"
        )

        result = await evaluate_generated_video_async(
            first_frame=[],
            generated_video_path=fake_video,
            prompt="Test prompt",
            gemini_client=mock_gemini_client,
        )

        assert result["passed"] is False
        assert result["overall_score"] == 0
        assert "Parse error" in result["reasoning"]

    async def test_evaluate_generated_video_retries_then_succeeds_on_gemini(
        self, mocker, mock_gemini_client, fake_video
    ):
        """A transient Gemini failure on attempt 1 should succeed on attempt 2."""
        from unittest.mock import AsyncMock

        from leoma.infra import judge
        from leoma.infra.judge import evaluate_generated_video_async

        sleep_mock = mocker.patch.object(judge.asyncio, "sleep", new_callable=AsyncMock)
        success_payload = MagicMock(text=json.dumps({"overall_score": 90, "confidence": 80}))
        mock_gemini_client.aio.models.generate_content.side_effect = [
            RuntimeError("transient"),
            success_payload,
        ]

        result = await evaluate_generated_video_async(
            first_frame=[_FAKE_FRAME],
            generated_video_path=fake_video,
            prompt="Test prompt",
            gemini_client=mock_gemini_client,
        )

        assert result["overall_score"] == 90
        assert mock_gemini_client.aio.models.generate_content.call_count == 2
        assert sleep_mock.await_count == 1

    async def test_evaluate_generated_video_propagates_after_retries(
        self, mocker, mock_gemini_client, fake_video
    ):
        """All 3 Gemini failures should propagate the last error (no fallback)."""
        from unittest.mock import AsyncMock

        from leoma.infra import judge
        from leoma.infra.judge import evaluate_generated_video_async

        sleep_mock = mocker.patch.object(judge.asyncio, "sleep", new_callable=AsyncMock)
        mock_gemini_client.aio.models.generate_content.side_effect = RuntimeError("Gemini down")

        with pytest.raises(RuntimeError, match="Gemini down"):
            await evaluate_generated_video_async(
                first_frame=[_FAKE_FRAME],
                generated_video_path=fake_video,
                prompt="Test prompt",
                gemini_client=mock_gemini_client,
            )
        assert mock_gemini_client.aio.models.generate_content.call_count == 3
        assert sleep_mock.await_count == 2
        sleep_mock.assert_awaited_with(300)

    async def test_evaluate_generated_video_requires_gemini_client(self):
        """Calling without a gemini_client should raise ValueError."""
        from leoma.infra.judge import evaluate_generated_video_async

        with pytest.raises(ValueError):
            await evaluate_generated_video_async(
                first_frame=[],
                generated_video_path="/nonexistent.mp4",
                prompt="Test prompt",
            )
