"""Creativity presets mapping modes to LLM parameters."""

from dataclasses import dataclass


@dataclass
class CreativityPreset:
    """Configuration for a creativity mode.

    The `creativity` value is a normalized 0-1 scale representing relative creativity.
    This gets mapped to provider-specific temperature ranges at runtime.
    """

    creativity: float  # 0.0 = minimal, 1.0 = maximum creativity
    top_p: float
    prompt_suffix: str


# Provider-specific temperature ranges
PROVIDER_TEMP_RANGES: dict[str, tuple[float, float]] = {
    "anthropic": (0.0, 1.0),
    "openai": (0.0, 2.0),
    "google": (0.0, 2.0),  # Gemini
    "gemini": (0.0, 2.0),
    "ollama": (0.0, 2.0),
    "groq": (0.0, 2.0),
    "mistral": (0.0, 1.0),
}

DEFAULT_TEMP_RANGE = (0.0, 1.0)


def get_temperature_for_provider(creativity: float, model: str) -> float:
    """Map normalized creativity (0-1) to provider-specific temperature.

    Args:
        creativity: Normalized creativity level (0.0-1.0)
        model: Model string like "anthropic:claude-sonnet-4-20250514"

    Returns:
        Temperature value within the provider's valid range
    """
    # Extract provider from model string (e.g., "anthropic:claude-..." -> "anthropic")
    provider = model.split(":")[0].lower() if ":" in model else model.lower()

    min_temp, max_temp = PROVIDER_TEMP_RANGES.get(provider, DEFAULT_TEMP_RANGE)

    # Map creativity 0-1 to provider's temperature range
    # We use a curve that emphasizes the upper range for more expressive "wild" mode
    temperature = min_temp + (creativity * (max_temp - min_temp))

    return round(temperature, 2)


PRESETS: dict[str, CreativityPreset] = {
    "practical": CreativityPreset(
        creativity=0.3,
        top_p=0.85,
        prompt_suffix="Be concise and actionable. Focus on feasibility and implementation.",
    ),
    "balanced": CreativityPreset(
        creativity=0.5,
        top_p=0.9,
        prompt_suffix="Balance creative exploration with practical considerations.",
    ),
    "creative": CreativityPreset(
        creativity=0.75,
        top_p=0.95,
        prompt_suffix="Think divergently. Explore unconventional angles and possibilities.",
    ),
    "wild": CreativityPreset(
        creativity=1.0,
        top_p=1.0,
        prompt_suffix="No constraints. Make unexpected connections. Surprise me.",
    ),
}


def get_preset(mode: str) -> CreativityPreset:
    """Get a creativity preset by mode name."""
    if mode not in PRESETS:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(PRESETS.keys())}")
    return PRESETS[mode]
