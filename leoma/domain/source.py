"""
Source and generation metadata.

VideoSource, prompt/generation params and info.
"""

from pydantic import BaseModel

DEFAULT_PROMPT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GENERATION_FPS = 16
DEFAULT_GENERATION_FRAMES = 81
DEFAULT_GENERATION_RESOLUTION = "480p"


class VideoSource(BaseModel):
    """Source video metadata: bucket, key, durations."""

    bucket: str
    key: str
    full_duration_seconds: float
    clip_start_seconds: float
    clip_duration_seconds: float


class PromptInfo(BaseModel):
    """Generated prompt metadata: model, text."""

    model: str = DEFAULT_PROMPT_MODEL
    text: str


class GenerationParams(BaseModel):
    """Video generation parameters: fps, frames, resolution, fast."""

    fps: int = DEFAULT_GENERATION_FPS
    frames: int = DEFAULT_GENERATION_FRAMES
    resolution: str = DEFAULT_GENERATION_RESOLUTION
    fast: bool = True


class GenerationInfo(BaseModel):
    """Generation run metadata: model, endpoint, parameters."""

    model: str
    endpoint: str
    parameters: GenerationParams
