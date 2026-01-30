"""CLI entry point for RiddleDiary."""

import asyncio
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .agent import create_agent, get_model_settings, run_brainstorm
from .creativity import PRESETS, get_preset

app = typer.Typer(
    name="riddle",
    help="Creative brainstorming tool powered by LLMs",
    no_args_is_help=True,
)
console = Console()


def print_response(response: str) -> None:
    """Print a formatted response."""
    console.print()
    console.print(Markdown(response))
    console.print()


def print_mode_info(mode: str) -> None:
    """Print information about the current mode."""
    preset = get_preset(mode)
    console.print(
        f"[dim]Mode: {mode} | Temperature: {preset.temperature} | Top-P: {preset.top_p}[/dim]"
    )


@app.command()
def brainstorm(
    topic: str = typer.Argument(..., help="Topic or question to brainstorm about"),
    mode: str = typer.Option(
        "balanced",
        "--mode",
        "-m",
        help="Creativity mode: practical, balanced, creative, wild",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="LLM model to use (e.g., anthropic:claude-sonnet-4-20250514)",
    ),
    one_shot: bool = typer.Option(
        False,
        "--one-shot",
        "-1",
        help="Get a single response without entering interactive mode",
    ),
) -> None:
    """Start a brainstorming session on a topic."""
    if mode not in PRESETS:
        console.print(f"[red]Unknown mode: {mode}[/red]")
        console.print(f"Available modes: {', '.join(PRESETS.keys())}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]{topic}[/bold]",
            title="Brainstorming",
            border_style="blue",
        )
    )
    print_mode_info(mode)

    asyncio.run(_brainstorm_loop(topic, mode, model, one_shot))


async def _brainstorm_loop(
    topic: str,
    mode: str,
    model: str | None,
    one_shot: bool,
) -> None:
    """Async brainstorming loop."""
    agent = create_agent(mode=mode, model=model)
    preset = get_preset(mode)
    message_history: list[Any] = []

    # Initial brainstorm
    console.print("[dim]Thinking...[/dim]")
    response, message_history = await run_brainstorm(
        agent,
        f"Let's brainstorm about: {topic}",
        preset,
        message_history=None,
    )
    print_response(response)

    if one_shot:
        return

    # Interactive loop
    console.print("[dim]Enter your thoughts, or:[/dim]")
    console.print("[dim]  /mode <name> - switch creativity mode[/dim]")
    console.print("[dim]  /quit - exit[/dim]")
    console.print()

    current_mode = mode

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input.strip():
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd_parts = user_input[1:].split(maxsplit=1)
            cmd = cmd_parts[0].lower()

            if cmd in ("quit", "q", "exit"):
                console.print("[dim]Session ended.[/dim]")
                break

            elif cmd == "mode":
                if len(cmd_parts) < 2:
                    console.print(f"[dim]Current mode: {current_mode}[/dim]")
                    console.print(f"[dim]Available: {', '.join(PRESETS.keys())}[/dim]")
                else:
                    new_mode = cmd_parts[1].lower()
                    if new_mode in PRESETS:
                        current_mode = new_mode
                        preset = get_preset(current_mode)
                        print_mode_info(current_mode)
                    else:
                        console.print(f"[red]Unknown mode: {new_mode}[/red]")
                continue

            elif cmd == "help":
                console.print("[dim]Commands:[/dim]")
                console.print("[dim]  /mode <name> - switch mode (practical/balanced/creative/wild)[/dim]")
                console.print("[dim]  /quit - exit session[/dim]")
                continue

            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                continue

        # Run brainstorm turn
        console.print("[dim]Thinking...[/dim]")
        response, message_history = await run_brainstorm(
            agent,
            user_input,
            preset,
            message_history=message_history,
        )
        print_response(response)


@app.command()
def modes() -> None:
    """List available creativity modes."""
    console.print("\n[bold]Creativity Modes[/bold]\n")
    for name, preset in PRESETS.items():
        console.print(f"[bold cyan]{name}[/bold cyan]")
        console.print(f"  Temperature: {preset.temperature}, Top-P: {preset.top_p}")
        console.print(f"  [dim]{preset.prompt_suffix}[/dim]")
        console.print()


if __name__ == "__main__":
    app()
