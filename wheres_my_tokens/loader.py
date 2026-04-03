"""JSONL streaming parser — extracts Turn records from Claude Code conversation files."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .models import Profile, Turn


def load_all_turns(profiles: list[Profile], progress: bool = True) -> list[Turn]:
    """Load Turn records from all profiles."""
    all_turns = []
    for profile in profiles:
        turns = load_profile_turns(profile, progress=progress)
        all_turns.extend(turns)
    return all_turns


def load_profile_turns(profile: Profile, progress: bool = True) -> list[Turn]:
    """Load all Turn records from a single profile."""
    projects_dir = profile.path / "projects"
    if not projects_dir.exists():
        return []

    jsonl_files = list(projects_dir.rglob("*.jsonl"))
    if progress:
        print(f"  Loading {len(jsonl_files)} session files from {profile.label}...",
              file=sys.stderr, end="", flush=True)

    turns = []
    errors = 0
    for f in jsonl_files:
        try:
            file_turns = extract_turns_from_file(f, profile.name)
            turns.extend(file_turns)
        except Exception:
            errors += 1

    if progress:
        print(f" {len(turns)} turns loaded" +
              (f" ({errors} files skipped)" if errors else ""),
              file=sys.stderr)
    return turns


def extract_turns_from_file(filepath: Path, profile_name: str) -> list[Turn]:
    """Extract Turn records from a single conversation JSONL file.

    Groups assistant messages by requestId and keeps only the last message
    per requestId (which has the final cumulative token counts).
    """
    project_dir = filepath.parent.name
    project_path = decode_project_path(project_dir)

    turns: dict[str, Turn] = {}  # requestId -> Turn (last write wins)
    session_id = None
    first_cwd = None

    # Use filename (UUID) as fallback session_id when messages lack one
    fallback_session_id = filepath.stem

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Grab session_id and cwd from first available message
            if session_id is None:
                sid = msg.get("sessionId", "")
                if sid:  # Only accept non-empty sessionIds
                    session_id = sid
            if first_cwd is None:
                cwd = msg.get("cwd", "")
                if cwd:
                    first_cwd = cwd

            if msg.get("type") != "assistant":
                continue

            message = msg.get("message", {})
            if not isinstance(message, dict):
                continue

            # Skip <synthetic> messages — client-side error/rate-limit placeholders
            if message.get("model") == "<synthetic>":
                continue

            usage = message.get("usage", {})
            if not usage or usage.get("input_tokens") is None:
                continue

            request_id = msg.get("requestId", "")
            if not request_id:
                # Older files may lack requestId — use message id as fallback
                request_id = message.get("id", id(msg))

            content = message.get("content", [])
            if not isinstance(content, list):
                content = []

            tool_names = [
                c["name"] for c in content
                if isinstance(c, dict) and c.get("type") == "tool_use" and "name" in c
            ]

            has_thinking = any(
                isinstance(c, dict) and c.get("type") == "thinking"
                for c in content
            )
            thinking_len = sum(
                len(c.get("thinking", ""))
                for c in content
                if isinstance(c, dict) and c.get("type") == "thinking"
            )

            # Parse timestamp
            ts_str = msg.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = datetime.now(timezone.utc)

            cache_creation = usage.get("cache_creation", {})

            # Overwrite — last message per requestId has final token counts
            turns[request_id] = Turn(
                request_id=str(request_id),
                session_id=session_id or fallback_session_id,
                profile_name=profile_name,
                project_dir=project_dir,
                project_path=first_cwd or project_path,
                model=message.get("model", "unknown"),
                timestamp=ts,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                ephemeral_5m_tokens=cache_creation.get("ephemeral_5m_input_tokens", 0) if isinstance(cache_creation, dict) else 0,
                ephemeral_1h_tokens=cache_creation.get("ephemeral_1h_input_tokens", 0) if isinstance(cache_creation, dict) else 0,
                stop_reason=message.get("stop_reason", "") or "",
                tool_uses=tool_names,
                has_thinking=has_thinking,
                thinking_text_len=thinking_len,
                is_sidechain=bool(msg.get("isSidechain", False)),
                version=msg.get("version", ""),
                cwd=msg.get("cwd", first_cwd or ""),
            )

    return list(turns.values())


def decode_project_path(dirname: str) -> str:
    """Decode an encoded project directory name to a filesystem path.

    Claude Code encodes paths like /home/user/project as -home-user-project.
    This is ambiguous for names containing hyphens, so this is a best-effort decode.
    """
    if not dirname or not dirname.startswith("-"):
        return dirname
    # Replace leading dash with / and subsequent dashes with /
    # This is lossy — dirs with actual hyphens will be wrong.
    # The loader prefers cwd from the first message when available.
    return "/" + dirname[1:].replace("-", "/")
