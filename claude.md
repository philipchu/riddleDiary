# RiddleDiary - Creative Brainstorming Tool

## Overview

A CLI brainstorming tool powered by LLMs with adjustable creativity levels, internet research capabilities, and extensible data analysis features. Built on [Pydantic AI](https://ai.pydantic.dev/).

## Why Pydantic AI?

- **Per-call temperature control** - Switch between creative and practical modes mid-conversation
- **Type-safe outputs** - Structured results via Pydantic models
- **Model-agnostic** - Swap between Claude, GPT, Gemini, local models
- **Excellent DX** - FastAPI-like ergonomics from the Pydantic team
- **MCP support** - Connect external tools via Model Context Protocol
- **Active development** - Strong backing, growing ecosystem

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Entry Point                         │
│                 riddle brainstorm --mode X                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Creativity Controller                      │
│         Maps modes to temperature/top_p settings             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │
│  │practical │ │ balanced │ │ creative │ │    wild      │    │
│  │ t=0.3    │ │  t=0.7   │ │  t=1.0   │ │   t=1.4      │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pydantic AI Agent                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                      Tools                            │  │
│  │  @agent.tool web_search    @agent.tool web_fetch     │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 Structured Outputs                    │  │
│  │  BrainstormResult { ideas, themes, next_steps }      │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               Conversation History                    │  │
│  │  Passed via message_history for multi-turn           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLM Providers                            │
│       Claude | GPT | Gemini | Ollama | Any OpenAI-compat    │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
riddleDiary/
├── .venv/                      # Virtual environment
├── pyproject.toml              # Project config & dependencies
├── .env                        # API keys (git-ignored)
├── src/
│   └── riddle/
│       ├── __init__.py
│       ├── cli.py              # Typer CLI entry point
│       ├── agent.py            # Pydantic AI agent definition
│       ├── creativity.py       # Mode presets (temp, top_p, prompts)
│       ├── tools.py            # Web search, web fetch tools
│       ├── models.py           # Pydantic output models
│       └── session.py          # Session persistence (future)
├── tests/
├── claude.md
└── README.md
```

## Core Components

### 1. Creativity Presets

```python
# src/riddle/creativity.py
from dataclasses import dataclass

@dataclass
class CreativityPreset:
    temperature: float
    top_p: float
    prompt_suffix: str

PRESETS = {
    "practical": CreativityPreset(
        temperature=0.3,
        top_p=0.85,
        prompt_suffix="Be concise and actionable. Focus on feasibility.",
    ),
    "balanced": CreativityPreset(
        temperature=0.7,
        top_p=0.9,
        prompt_suffix="Balance creative exploration with practical considerations.",
    ),
    "creative": CreativityPreset(
        temperature=1.0,
        top_p=0.95,
        prompt_suffix="Think divergently. Explore unconventional angles.",
    ),
    "wild": CreativityPreset(
        temperature=1.4,
        top_p=1.0,
        prompt_suffix="No constraints. Make unexpected connections. Surprise me.",
    ),
}
```

### 2. Agent Definition

```python
# src/riddle/agent.py
from pydantic_ai import Agent, RunContext
from .models import BrainstormResult
from .tools import web_search, web_fetch

SYSTEM_PROMPT = """You are a creative brainstorming partner. Your role is to:

1. Generate and explore ideas freely
2. Build on thoughts with "yes, and..." thinking
3. Use web search when current information would help
4. Challenge assumptions constructively
5. Organize ideas when asked

{mode_suffix}
"""

def create_agent(mode_suffix: str = "") -> Agent[None, BrainstormResult]:
    agent = Agent(
        'anthropic:claude-sonnet-4-20250514',
        output_type=BrainstormResult,
        system_prompt=SYSTEM_PROMPT.format(mode_suffix=mode_suffix),
    )

    # Register tools
    agent.tool(web_search)
    agent.tool(web_fetch)

    return agent
```

### 3. Output Models

```python
# src/riddle/models.py
from pydantic import BaseModel

class BrainstormResult(BaseModel):
    """Structured output from a brainstorm turn."""
    ideas: list[str]
    themes: list[str] | None = None
    questions: list[str] | None = None
    next_steps: list[str] | None = None
```

### 4. Tools

```python
# src/riddle/tools.py
from pydantic_ai import RunContext
import httpx

async def web_search(ctx: RunContext, query: str) -> str:
    """Search the web for current information on a topic."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"query": query, "api_key": os.environ["TAVILY_API_KEY"]},
        )
        return resp.json()

async def web_fetch(ctx: RunContext, url: str) -> str:
    """Fetch and extract content from a webpage."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        # Use trafilatura for content extraction
        from trafilatura import extract
        return extract(resp.text) or resp.text[:5000]
```

## Dependencies

```toml
[project]
name = "riddle-diary"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "pydantic-ai>=0.1",
    "typer>=0.12",
    "rich>=13.0",
    "httpx>=0.27",
    "trafilatura>=1.12",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
tavily = ["tavily-python>=0.5"]
analysis = ["pandas>=2.0", "matplotlib>=3.8"]
dev = ["pytest>=8.0", "ruff>=0.4"]

[project.scripts]
riddle = "riddle.cli:app"
```

## Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...           # For web search
OPENAI_API_KEY=sk-...             # Optional, for GPT models
DEFAULT_MODEL=anthropic:claude-sonnet-4-20250514
```

## Usage

```bash
# Creative brainstorm
riddle brainstorm "How might we redesign public libraries?" --mode creative

# Practical problem-solving
riddle brainstorm "Reduce CI/CD build times" --mode practical

# Wild ideation
riddle brainstorm "Future of human-AI collaboration" --mode wild

# Interactive session (default)
riddle brainstorm "Sustainable packaging ideas"
> Focus on the mushroom-based materials
> [switches to practical] Make it actionable
> /export ideas.md
```

## Next Steps

1. [x] Define architecture
2. [ ] Set up .venv and install dependencies
3. [ ] Implement core agent with creativity presets
4. [ ] Add web search/fetch tools
5. [ ] Build CLI with interactive mode
6. [ ] Add session persistence
7. [ ] Add data analysis tools (future)
