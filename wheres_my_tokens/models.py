"""Data models for token usage analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Profile:
    """A Claude Code profile directory (e.g. ~/.claude)."""
    path: Path
    name: str  # directory name, e.g. ".claude" or ".claude-other"
    email: str | None = None  # from .claude.json oauthAccount

    @property
    def label(self) -> str:
        if self.email:
            return f"{self.name} ({self.email})"
        return self.name


@dataclass
class Turn:
    """One API request/response cycle, deduplicated by requestId.

    Token counts come from the final assistant message for each requestId
    (output_tokens are cumulative across streaming chunks).
    """
    request_id: str
    session_id: str
    profile_name: str
    project_dir: str  # encoded dirname like "-home-user-project"
    project_path: str  # decoded path like "/home/user/project"
    model: str
    timestamp: datetime

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    # Cache breakdown
    ephemeral_5m_tokens: int = 0
    ephemeral_1h_tokens: int = 0

    # Content analysis
    stop_reason: str = ""
    tool_uses: list[str] = field(default_factory=list)
    has_thinking: bool = False
    thinking_text_len: int = 0

    # Metadata
    is_sidechain: bool = False
    version: str = ""
    cwd: str = ""

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens + self.output_tokens +
                self.cache_creation_input_tokens + self.cache_read_input_tokens)

    @property
    def context_tokens(self) -> int:
        """Tokens representing the input context size for this turn."""
        return (self.input_tokens + self.cache_creation_input_tokens +
                self.cache_read_input_tokens)


@dataclass
class Session:
    """Aggregated stats for one session (one conversation)."""
    session_id: str
    profile_name: str
    project_dir: str
    project_path: str
    models: set[str] = field(default_factory=set)
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    turn_count: int = 0

    # Aggregated tokens
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0

    # Tool usage
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    thinking_turn_count: int = 0

    # Cost
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return (self.total_input_tokens + self.total_output_tokens +
                self.total_cache_creation_tokens + self.total_cache_read_tokens)

    @property
    def duration_seconds(self) -> float:
        if self.first_timestamp and self.last_timestamp:
            return (self.last_timestamp - self.first_timestamp).total_seconds()
        return 0.0

    @property
    def duration_display(self) -> str:
        s = self.duration_seconds
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s / 60:.0f}m"
        return f"{s / 3600:.1f}h"
