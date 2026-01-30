"""Creativity presets mapping modes to LLM parameters."""

from dataclasses import dataclass


@dataclass
class CreativityPreset:
    """Configuration for a creativity mode."""

    temperature: float
    top_p: float
    prompt_suffix: str


PRESETS: dict[str, CreativityPreset] = {
    "practical": CreativityPreset(
        temperature=0.3,
        top_p=0.85,
        prompt_suffix="Be concise and actionable. Focus on feasibility and implementation.",
    ),
    "balanced": CreativityPreset(
        temperature=0.7,
        top_p=0.9,
        prompt_suffix="Balance creative exploration with practical considerations.",
    ),
    "creative": CreativityPreset(
        temperature=1.0,
        top_p=0.95,
        prompt_suffix="Think divergently. Explore unconventional angles and possibilities.",
    ),
    "wild": CreativityPreset(
        temperature=1.4,
        top_p=1.0,
        prompt_suffix="No constraints. Make unexpected connections. Surprise me.",
    ),
}


def get_preset(mode: str) -> CreativityPreset:
    """Get a creativity preset by mode name."""
    if mode not in PRESETS:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(PRESETS.keys())}")
    return PRESETS[mode]
