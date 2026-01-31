"""Pydantic AI agent for brainstorming."""

import os
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent, ModelSettings

from .creativity import CreativityPreset, get_preset, get_temperature_for_provider
from .tools import web_fetch, web_search

load_dotenv()

# Map user-friendly key names to what Pydantic AI expects
if os.environ.get("CLAUDE_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]

# Use GEMINI_API_KEY directly (Pydantic AI supports it), unset GOOGLE_API_KEY to avoid conflicts
if os.environ.get("GEMINI_API_KEY"):
    os.environ.pop("GOOGLE_API_KEY", None)

SYSTEM_PROMPT = """You are a creative brainstorming partner. Your role is to:

1. Generate and explore ideas freely with the user
2. Build on their thoughts with "yes, and..." thinking
3. Use web_search when current information would help the discussion
4. Challenge assumptions constructively
5. Help organize and synthesize ideas when asked

Brainstorming principles:
- Quantity breeds quality - generate many ideas before evaluating
- Defer judgment - explore before critiquing
- Wild ideas welcome - they often lead to practical innovations
- Combine and build - merge concepts together

{mode_suffix}

Respond conversationally. When you generate notable ideas, capture them in your response.
If the user asks you to search for something or you need current information, use the web_search tool.
"""


def create_agent(mode: str = "balanced", model: str | None = None) -> Agent[None, str]:
    """Create a brainstorming agent with the specified creativity mode.

    Args:
        mode: Creativity mode (practical, balanced, creative, wild)
        model: LLM model to use (defaults to ANTHROPIC_API_KEY model)

    Returns:
        Configured Pydantic AI agent
    """
    preset = get_preset(mode)
    model_name = model or os.environ.get("DEFAULT_MODEL", "anthropic:claude-sonnet-4")

    agent: Agent[None, str] = Agent(
        model_name,
        system_prompt=SYSTEM_PROMPT.format(mode_suffix=preset.prompt_suffix),
    )

    # Register tools
    agent.tool_plain(web_search)
    agent.tool_plain(web_fetch)

    return agent


def get_model_settings(preset: CreativityPreset, model: str) -> ModelSettings:
    """Get model settings for a creativity preset, mapped to the provider's range.

    Args:
        preset: The creativity preset
        model: Model string (e.g., "anthropic:claude-sonnet-4-20250514")

    Returns:
        ModelSettings with provider-appropriate temperature
    """
    temperature = get_temperature_for_provider(preset.creativity, model)
    return ModelSettings(
        temperature=temperature,
        top_p=preset.top_p,
    )


async def run_brainstorm(
    agent: Agent[None, str],
    prompt: str,
    preset: CreativityPreset,
    model: str,
    message_history: list[Any] | None = None,
) -> tuple[str, list[Any]]:
    """Run a brainstorm turn.

    Args:
        agent: The brainstorming agent
        prompt: User's prompt
        preset: Creativity preset for model settings
        model: Model string for provider-specific temperature mapping
        message_history: Previous conversation messages

    Returns:
        Tuple of (response text, updated message history)
    """
    result = await agent.run(
        prompt,
        message_history=message_history,
        model_settings=get_model_settings(preset, model),
    )

    return result.output, result.all_messages()
