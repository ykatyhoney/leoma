"""
Unit tests for evaluation utilities.

Tests description generation (GPT-4o, sampler side) and
generated-video scoring (Gemini, validator side) prompt building and response parsing.
"""

import base64
import json
from unittest.mock import MagicMock

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
    """Tests for get_description_async function."""

    async def test_get_description_returns_text(self, mock_openai_client):
        """Test that get_description_async returns description text."""
        from leoma.infra.judge import get_description_async

        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="A skateboarder crossing a quiet plaza."))]
        )

        frames = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}}]
        result = await get_description_async(mock_openai_client, frames)

        assert result == "A skateboarder crossing a quiet plaza."

    async def test_get_description_strips_whitespace(self, mock_openai_client):
        """Test that description is stripped of whitespace."""
        from leoma.infra.judge import get_description_async

        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="  Trimmed text  \n"))]
        )

        frames = []
        result = await get_description_async(mock_openai_client, frames)

        assert result == "Trimmed text"

    async def test_get_description_builds_correct_prompt(self, mock_openai_client):
        """Test that the prompt is correctly constructed."""
        from leoma.infra.judge import get_description_async

        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Description"))]
        )

        frames = [
            {"type": "image_url", "image_url": {"url": "frame1"}},
            {"type": "image_url", "image_url": {"url": "frame2"}},
        ]

        await get_description_async(mock_openai_client, frames)

        # Verify the API was called
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs

        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["max_tokens"] == 220

        # Content should include text prompt + frames
        content = call_kwargs["messages"][0]["content"]
        assert len(content) == 3  # 1 text + 2 frames


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
    """Tests for single-video benchmark evaluation (Gemini primary, GPT-4o fallback)."""

    async def test_evaluate_generated_video_passes(self, mock_gemini_client):
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
            generated_frames=[_FAKE_FRAME],
            prompt="A person jogging in a city park at sunrise",
            gemini_client=mock_gemini_client,
        )

        assert result["passed"] is True
        assert result["overall_score"] == 86
        assert result["confidence"] == 89

    async def test_evaluate_generated_video_fails_on_critical_floor(self, mock_gemini_client):
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
            generated_frames=[],
            prompt="Test prompt",
            gemini_client=mock_gemini_client,
        )

        assert result["passed"] is False
        assert result["overall_score"] == 78

    async def test_evaluate_generated_video_handles_parse_error(self, mock_gemini_client):
        """Invalid JSON should return fail-safe output."""
        from leoma.infra.judge import evaluate_generated_video_async

        mock_gemini_client.aio.models.generate_content.return_value = MagicMock(
            text="not-json"
        )

        result = await evaluate_generated_video_async(
            first_frame=[],
            generated_frames=[],
            prompt="Test prompt",
            gemini_client=mock_gemini_client,
        )

        assert result["passed"] is False
        assert result["overall_score"] == 0
        assert "Parse error" in result["reasoning"]

    async def test_evaluate_generated_video_openai_only(self, mock_openai_client):
        """When no Gemini client is provided, evaluation runs via GPT-4o."""
        from leoma.infra.judge import evaluate_generated_video_async

        response_json = {
            "overall_score": 81,
            "confidence": 77,
            "aspect_scores": {
                "first_frame_fidelity": 80,
                "prompt_adherence": 82,
                "motion_quality": 79,
                "temporal_consistency": 83,
                "visual_quality": 80,
                "camera_composition": 78,
            },
            "major_issues": [],
            "strengths": ["coherent motion"],
            "reasoning": "Solid overall quality.",
        }
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(response_json)))]
        )

        result = await evaluate_generated_video_async(
            first_frame=[_FAKE_FRAME],
            generated_frames=[_FAKE_FRAME],
            prompt="Test prompt",
            openai_client=mock_openai_client,
        )

        assert result["passed"] is True
        assert result["overall_score"] == 81
        mock_openai_client.chat.completions.create.assert_called_once()

    async def test_evaluate_generated_video_falls_back_to_openai_on_gemini_error(
        self, mocker, mock_gemini_client, mock_openai_client
    ):
        """Gemini is retried 3 times with 5-minute sleeps before GPT-4o fallback kicks in."""
        from unittest.mock import AsyncMock

        from leoma.infra import judge
        from leoma.infra.judge import evaluate_generated_video_async

        sleep_mock = mocker.patch.object(judge.asyncio, "sleep", new_callable=AsyncMock)
        mock_gemini_client.aio.models.generate_content.side_effect = RuntimeError("Gemini down")

        response_json = {
            "overall_score": 74,
            "confidence": 70,
            "aspect_scores": {
                "first_frame_fidelity": 72,
                "prompt_adherence": 76,
                "motion_quality": 70,
                "temporal_consistency": 78,
                "visual_quality": 74,
                "camera_composition": 70,
            },
            "major_issues": [],
            "strengths": [],
            "reasoning": "Fallback evaluation via GPT-4o.",
        }
        mock_openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(response_json)))]
        )

        result = await evaluate_generated_video_async(
            first_frame=[_FAKE_FRAME],
            generated_frames=[_FAKE_FRAME],
            prompt="Test prompt",
            gemini_client=mock_gemini_client,
            openai_client=mock_openai_client,
        )

        assert result["passed"] is True
        assert result["overall_score"] == 74
        assert mock_gemini_client.aio.models.generate_content.call_count == 3
        assert sleep_mock.await_count == 2
        sleep_mock.assert_awaited_with(300)
        mock_openai_client.chat.completions.create.assert_called_once()

    async def test_evaluate_generated_video_retries_then_succeeds_on_gemini(
        self, mocker, mock_gemini_client
    ):
        """A transient Gemini failure on attempt 1 should succeed on attempt 2 without fallback."""
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
            generated_frames=[_FAKE_FRAME],
            prompt="Test prompt",
            gemini_client=mock_gemini_client,
        )

        assert result["overall_score"] == 90
        assert mock_gemini_client.aio.models.generate_content.call_count == 2
        assert sleep_mock.await_count == 1

    async def test_evaluate_generated_video_reraises_when_no_fallback(
        self, mocker, mock_gemini_client
    ):
        """Without an openai_client, all 3 Gemini failures should propagate the last error."""
        import pytest
        from unittest.mock import AsyncMock

        from leoma.infra import judge
        from leoma.infra.judge import evaluate_generated_video_async

        mocker.patch.object(judge.asyncio, "sleep", new_callable=AsyncMock)
        mock_gemini_client.aio.models.generate_content.side_effect = RuntimeError("Gemini down")

        with pytest.raises(RuntimeError, match="Gemini down"):
            await evaluate_generated_video_async(
                first_frame=[_FAKE_FRAME],
                generated_frames=[_FAKE_FRAME],
                prompt="Test prompt",
                gemini_client=mock_gemini_client,
            )
        assert mock_gemini_client.aio.models.generate_content.call_count == 3

    async def test_evaluate_generated_video_requires_a_client(self):
        """Calling without either client should raise ValueError."""
        import pytest
        from leoma.infra.judge import evaluate_generated_video_async

        with pytest.raises(ValueError):
            await evaluate_generated_video_async(
                first_frame=[],
                generated_frames=[],
                prompt="Test prompt",
            )
