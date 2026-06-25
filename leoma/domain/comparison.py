"""
Evaluation and comparison result types.
"""

from typing import List

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Evaluation result for a single comparison."""

    passed: bool
    confidence: int = Field(ge=0, le=100)
    reasoning: str
    original_artifacts: List[str] = Field(default_factory=list)
    generated_artifacts: List[str] = Field(default_factory=list)
    presentation_order: str
