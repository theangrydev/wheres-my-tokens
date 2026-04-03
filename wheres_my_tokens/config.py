"""Configuration loading and profile discovery."""

import json
from pathlib import Path

import yaml

from .models import Profile

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(config_path: Path | None = None) -> dict:
    """Load config.yaml with model pricing and settings."""
    # Start with defaults
    config = _default_config()

    # Load config
    base_path = config_path or DEFAULT_CONFIG_PATH
    if base_path.exists():
        with open(base_path) as f:
            base = yaml.safe_load(f) or {}
        for key, value in base.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

    return config


def _default_config() -> dict:
    return {
        "pricing": {
            "_default": {
                "input": 3.0, "output": 15.0,
                "cache_creation": 3.75, "cache_read": 0.30,
            }
        },
        "output_dir": "./output",
        "top_n": 20,
    }


def get_pricing(config: dict, model: str) -> dict:
    """Get pricing for a model, falling back to default."""
    pricing = config.get("pricing", {})
    if model in pricing:
        return pricing[model]
    # Try prefix match (e.g. "claude-opus-4-6[1m]" -> "claude-opus-4-6")
    base_model = model.split("[")[0]
    if base_model in pricing:
        return pricing[base_model]
    # Try matching model family
    for key in pricing:
        if key.startswith("_"):
            continue
        if key.split("-")[:-1] == base_model.split("-")[:-1]:
            return pricing[key]
    return pricing.get("_default", {"input": 3.0, "output": 15.0,
                                     "cache_creation": 3.75, "cache_read": 0.30})


def estimate_cost(config: dict, model: str,
                  input_tokens: int, output_tokens: int,
                  cache_creation_tokens: int, cache_read_tokens: int) -> float:
    """Estimate API-equivalent cost in USD."""
    p = get_pricing(config, model)
    return (
        input_tokens * p["input"] / 1_000_000 +
        output_tokens * p["output"] / 1_000_000 +
        cache_creation_tokens * p["cache_creation"] / 1_000_000 +
        cache_read_tokens * p["cache_read"] / 1_000_000
    )


def discover_profiles(explicit_paths: list[str] | None = None) -> list[Profile]:
    """Find Claude Code profile directories."""
    if explicit_paths:
        profiles = []
        for p in explicit_paths:
            path = Path(p).expanduser()
            if path.is_dir():
                profiles.append(_make_profile(path))
        return profiles

    home = Path.home()
    profiles = []
    for candidate in sorted(home.iterdir()):
        if (candidate.name.startswith(".claude") and
                not candidate.name.endswith(".bak") and
                candidate.is_dir() and
                (candidate / "projects").is_dir()):
            profiles.append(_make_profile(candidate))
    return profiles


def _make_profile(path: Path) -> Profile:
    """Create a Profile from a directory, reading account info if available."""
    email = None
    # Check for .claude.json inside the profile dir (e.g. ~/.claude-other/.claude.json)
    # and also at ~/.claude.json for the default profile (~/.claude)
    candidates = [path / ".claude.json"]
    if path.name == ".claude":
        candidates.append(path.parent / ".claude.json")
    for config_file in candidates:
        if config_file.exists():
            try:
                with open(config_file) as f:
                    data = json.load(f)
                email = data.get("oauthAccount", {}).get("emailAddress")
                if email:
                    break
            except (json.JSONDecodeError, KeyError):
                pass
    return Profile(path=path, name=path.name, email=email)
