"""Pydantic models for structured outputs."""

from pydantic import BaseModel, Field


class BrainstormResult(BaseModel):
    """Structured output from a brainstorm turn."""

    response: str = Field(description="The conversational response to show the user")
    ideas: list[str] = Field(default_factory=list, description="Key ideas generated")
    themes: list[str] | None = Field(default=None, description="Emerging themes")
    questions: list[str] | None = Field(default=None, description="Questions to explore further")
    next_steps: list[str] | None = Field(default=None, description="Actionable next steps")
