"""Serializable Pine configuration; API credentials stay in the environment."""

from pydantic import BaseModel, ConfigDict, Field


class PineConfig(BaseModel):
    """Public model IDs and reproducible instructions for a custom Pine agent.

    Backend instructions precede the benchmark policy, which is always retained.
    No task solution, caller goal, or hidden environment state is injected.
    """

    model_config = ConfigDict(extra="forbid")

    frontend_model: str = Field(min_length=1)
    backend_model: str = Field(min_length=1)
    frontend_instructions: str = Field(min_length=1)
    backend_instructions: str = Field(min_length=1)
    voice_id: str = Field(min_length=1)
    tool_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_duration_seconds: int = Field(default=1800, ge=30, le=7200)

    def render_backend_instructions(self, system_prompt: str) -> str:
        """Retain the complete benchmark policy after the supplied role prompt."""
        return self.backend_instructions + "\n\n" + system_prompt
