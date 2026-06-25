"""Source and generation metadata."""

from pydantic import BaseModel

DEFAULT_PROMPT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GENERATION_FPS = 16
DEFAULT_GENERATION_FRAMES = 81
DEFAULT_GENERATION_RESOLUTION = "480p"


class VideoSource(BaseModel):
    """Source video metadata."""

    bucket: str
    key: str
    full_duration_seconds: float
    clip_start_seconds: float
    clip_duration_seconds: float


class PromptInfo(BaseModel):
    """Generated prompt metadata."""

    model: str = DEFAULT_PROMPT_MODEL
    text: str


class GenerationParams(BaseModel):
    """Video generation parameters."""

    fps: int = DEFAULT_GENERATION_FPS
    frames: int = DEFAULT_GENERATION_FRAMES
    resolution: str = DEFAULT_GENERATION_RESOLUTION
    fast: bool = True


class GenerationInfo(BaseModel):
    """Generation run metadata."""

    model: str
    endpoint: str
    parameters: GenerationParams
