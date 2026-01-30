# RiddleDiary

A creative brainstorming tool powered by LLMs with adjustable creativity levels.

## Installation

```bash
# Clone and install
git clone <repo>
cd riddleDiary
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```bash
# Start a brainstorming session
riddle brainstorm "How might we redesign public transit?"

# Use different creativity modes
riddle brainstorm "Reduce build times" --mode practical
riddle brainstorm "Future of AI" --mode wild

# One-shot mode (no interactive follow-up)
riddle brainstorm "Quick ideas for a logo" --one-shot

# List available modes
riddle modes
```

## Modes

- **practical** - Focused, actionable, grounded (temp: 0.3)
- **balanced** - Mix of creative and practical (temp: 0.7)
- **creative** - Divergent thinking, unconventional (temp: 1.0)
- **wild** - No constraints, unexpected connections (temp: 1.4)

## Interactive Commands

During a session:
- `/mode <name>` - Switch creativity mode
- `/quit` - Exit session
