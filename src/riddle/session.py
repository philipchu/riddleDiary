"""Session persistence for brainstorming conversations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class Session(BaseModel):
    """A saved brainstorming session."""

    topic: str
    mode: str
    model: str
    created_at: str
    updated_at: str
    message_history: list[dict[str, Any]]


def get_sessions_dir() -> Path:
    """Get the sessions directory, creating it if needed."""
    sessions_dir = Path.home() / ".riddle" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def generate_session_filename(topic: str) -> str:
    """Generate a filename from the topic."""
    # Sanitize topic for filename
    safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)
    safe_topic = safe_topic[:40].strip().replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_topic}_{timestamp}.json"


def save_session(
    topic: str,
    mode: str,
    model: str,
    message_history: list[Any],
    filename: str | None = None,
) -> Path:
    """Save the current session to a file.

    Args:
        topic: The brainstorming topic
        mode: Current creativity mode
        model: Model being used
        message_history: Pydantic AI message history
        filename: Optional specific filename, otherwise auto-generated

    Returns:
        Path to the saved session file
    """
    sessions_dir = get_sessions_dir()

    # Serialize message history (Pydantic AI messages are Pydantic models)
    serialized_history = []
    for msg in message_history:
        if hasattr(msg, "model_dump"):
            serialized_history.append(msg.model_dump(mode="json"))
        elif isinstance(msg, dict):
            serialized_history.append(msg)
        else:
            # Fallback for other types
            serialized_history.append({"type": str(type(msg)), "content": str(msg)})

    now = datetime.now().isoformat()
    session = Session(
        topic=topic,
        mode=mode,
        model=model,
        created_at=now,
        updated_at=now,
        message_history=serialized_history,
    )

    if filename is None:
        filename = generate_session_filename(topic)

    filepath = sessions_dir / filename
    filepath.write_text(session.model_dump_json(indent=2))

    return filepath


def load_session(filename: str) -> Session:
    """Load a session from a file.

    Args:
        filename: Name of the session file (with or without .json)

    Returns:
        The loaded Session

    Raises:
        FileNotFoundError: If session file doesn't exist
    """
    sessions_dir = get_sessions_dir()

    if not filename.endswith(".json"):
        filename = f"{filename}.json"

    filepath = sessions_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Session not found: {filepath}")

    return Session.model_validate_json(filepath.read_text())


def list_sessions() -> list[tuple[str, Session]]:
    """List all saved sessions.

    Returns:
        List of (filename, Session) tuples, sorted by updated_at descending
    """
    sessions_dir = get_sessions_dir()
    sessions = []

    for filepath in sessions_dir.glob("*.json"):
        try:
            session = Session.model_validate_json(filepath.read_text())
            sessions.append((filepath.name, session))
        except Exception:
            # Skip malformed session files
            continue

    # Sort by updated_at, newest first
    sessions.sort(key=lambda x: x[1].updated_at, reverse=True)
    return sessions


def delete_session(filename: str) -> bool:
    """Delete a session file.

    Args:
        filename: Name of the session file

    Returns:
        True if deleted, False if not found
    """
    sessions_dir = get_sessions_dir()

    if not filename.endswith(".json"):
        filename = f"{filename}.json"

    filepath = sessions_dir / filename
    if filepath.exists():
        filepath.unlink()
        return True
    return False
