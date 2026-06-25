"""Sample and submission metadata."""

from typing import Dict, List

from pydantic import BaseModel

from leoma.domain.participant import MinerResult
from leoma.domain.source import GenerationInfo, PromptInfo, VideoSource


class SampleMetadata(BaseModel):
    """Full sample metadata."""

    task_id: int
    created_at: str
    source: VideoSource
    prompt: PromptInfo
    generation: GenerationInfo
    miners: Dict[str, MinerResult]
    files: List[str]
