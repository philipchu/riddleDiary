"""CLI entry point for RiddleDiary."""

import asyncio
import os
from typing import Any

import typer
from pydantic_ai.messages import ModelRequest, ModelResponse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .agent import create_agent, run_brainstorm
from .creativity import PRESETS, get_preset, get_temperature_for_provider
from .session import (
    delete_session,
    generate_session_filename,
    list_sessions,
    load_session,
    save_session,
)

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


def print_mode_info(mode: str, model: str) -> None:
    """Print information about the current mode."""
    preset = get_preset(mode)
    temp = get_temperature_for_provider(preset.creativity, model)
    console.print(
        f"[dim]Mode: {mode} | Temperature: {temp} | Top-P: {preset.top_p}[/dim]"
    )


def get_default_model() -> str:
    """Get the default model from environment or fallback."""
    return os.environ.get("DEFAULT_MODEL", "anthropic:claude-sonnet-4-20250514")


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
    resume: str = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume from a saved session file",
    ),
) -> None:
    """Start a brainstorming session on a topic."""
    model = model or get_default_model()

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
    print_mode_info(mode, model)

    asyncio.run(_brainstorm_loop(topic, mode, model, one_shot, resume))


def _reconstruct_message_history(serialized: list[dict[str, Any]]) -> list[Any]:
    """Reconstruct Pydantic AI message objects from serialized data."""
    messages: list[Any] = []
    for msg_data in serialized:
        kind = msg_data.get("kind")
        if kind == "request":
            messages.append(ModelRequest.model_validate(msg_data))
        elif kind == "response":
            messages.append(ModelResponse.model_validate(msg_data))
        # Skip unknown message types
    return messages


def _auto_save(
    topic: str,
    mode: str,
    model: str,
    message_history: list[Any],
    session_filename: str,
) -> None:
    """Auto-save session after each turn."""
    try:
        save_session(
            topic=topic,
            mode=mode,
            model=model,
            message_history=message_history,
            filename=session_filename,
        )
    except Exception:
        # Silently fail auto-save to not interrupt the session
        pass


async def _brainstorm_loop(
    topic: str,
    mode: str,
    model: str,
    one_shot: bool,
    resume: str | None = None,
) -> None:
    """Async brainstorming loop."""
    current_mode = mode
    message_history: list[Any] = []

    # Resume from saved session if specified
    if resume:
        try:
            session = load_session(resume)
            topic = session.topic
            current_mode = session.mode
            model = session.model
            message_history = _reconstruct_message_history(session.message_history)
            session_filename = resume if resume.endswith(".json") else f"{resume}.json"
            console.print(f"[green]Resumed session: {topic}[/green]")
            console.print(f"[dim]Messages loaded: {len(message_history)}[/dim]")
        except FileNotFoundError:
            console.print(f"[red]Session not found: {resume}[/red]")
            console.print("[dim]Use /sessions to list available sessions[/dim]")
            return
    else:
        # Generate a session filename for new sessions
        session_filename = generate_session_filename(topic)

    agent = create_agent(mode=current_mode, model=model)
    preset = get_preset(current_mode)

    # Initial brainstorm (skip if resuming)
    if not resume:
        console.print("[dim]Thinking...[/dim]")
        response, message_history = await run_brainstorm(
            agent,
            f"Let's brainstorm about: {topic}",
            preset,
            model,
            message_history=None,
        )
        print_response(response)
        # Auto-save after initial response
        _auto_save(topic, current_mode, model, message_history, session_filename)

    if one_shot:
        return

    # Interactive loop
    console.print("[dim]Enter your thoughts, or:[/dim]")
    console.print("[dim]  /mode <name> - switch creativity mode[/dim]")
    console.print("[dim]  /save [name] - rename session file[/dim]")
    console.print("[dim]  /quit - exit[/dim]")
    console.print(f"[dim]Session: {session_filename}[/dim]")
    console.print()

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
                        print_mode_info(current_mode, model)
                        # Save mode change
                        _auto_save(topic, current_mode, model, message_history, session_filename)
                    else:
                        console.print(f"[red]Unknown mode: {new_mode}[/red]")
                continue

            elif cmd == "save":
                if len(cmd_parts) > 1:
                    new_name = cmd_parts[1]
                    if not new_name.endswith(".json"):
                        new_name = f"{new_name}.json"
                    # Save with new name
                    filepath = save_session(
                        topic=topic,
                        mode=current_mode,
                        model=model,
                        message_history=message_history,
                        filename=new_name,
                    )
                    session_filename = filepath.name
                    console.print(f"[green]Session renamed to: {session_filename}[/green]")
                else:
                    console.print(f"[dim]Current session: {session_filename}[/dim]")
                continue

            elif cmd == "help":
                console.print("[dim]Commands:[/dim]")
                console.print("[dim]  /mode <name> - switch mode (practical/balanced/creative/wild)[/dim]")
                console.print("[dim]  /save <name> - rename session file[/dim]")
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
            model,
            message_history=message_history,
        )
        print_response(response)

        # Auto-save after each turn
        _auto_save(topic, current_mode, model, message_history, session_filename)


@app.command()
def modes(
    model: str = typer.Option(
        None,
        "--model",
        help="Show temperature values for a specific model",
    ),
) -> None:
    """List available creativity modes."""
    model = model or get_default_model()
    console.print(f"\n[bold]Creativity Modes[/bold] [dim](for {model})[/dim]\n")
    for name, preset in PRESETS.items():
        temp = get_temperature_for_provider(preset.creativity, model)
        console.print(f"[bold cyan]{name}[/bold cyan]")
        console.print(f"  Creativity: {preset.creativity} -> Temperature: {temp}, Top-P: {preset.top_p}")
        console.print(f"  [dim]{preset.prompt_suffix}[/dim]")
        console.print()


@app.command()
def sessions(
    delete: str = typer.Option(
        None,
        "--delete",
        "-d",
        help="Delete a session by filename",
    ),
) -> None:
    """List or manage saved sessions."""
    if delete:
        if delete_session(delete):
            console.print(f"[green]Deleted: {delete}[/green]")
        else:
            console.print(f"[red]Session not found: {delete}[/red]")
        return

    saved = list_sessions()
    if not saved:
        console.print("[dim]No saved sessions yet. Sessions are auto-saved during brainstorming.[/dim]")
        return

    table = Table(title="Saved Sessions")
    table.add_column("Filename", style="cyan")
    table.add_column("Topic", style="white")
    table.add_column("Mode", style="green")
    table.add_column("Updated", style="dim")

    for filename, session in saved:
        # Truncate topic if too long
        topic_display = session.topic[:40] + "..." if len(session.topic) > 40 else session.topic
        # Format date
        updated = session.updated_at[:16].replace("T", " ")
        table.add_row(filename, topic_display, session.mode, updated)

    console.print(table)
    console.print("\n[dim]Resume with: riddle brainstorm <topic> --resume <filename>[/dim]")


if __name__ == "__main__":
    app()
