"""Pickle-based caching for parsed Turn data."""

import os
import pickle
import sys
import time
from pathlib import Path

from .models import Profile, Turn

CACHE_VERSION = 1


def get_cache_path() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "wheres-my-tokens" / "turns.pkl"


def load_cached_turns(profiles: list[Profile]) -> tuple[list[Turn], set[str]]:
    """Load cached turns and return (turns, set_of_cached_file_paths).

    Returns empty results if cache is missing or invalid.
    """
    cache_path = get_cache_path()
    if not cache_path.exists():
        return [], set()

    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != CACHE_VERSION:
            return [], set()
        return data["turns"], set(data.get("file_mtimes", {}).keys())
    except Exception:
        return [], set()


def save_cache(turns: list[Turn], file_mtimes: dict[str, float]) -> None:
    """Save turns to pickle cache."""
    cache_path = get_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": CACHE_VERSION,
        "turns": turns,
        "file_mtimes": file_mtimes,
        "saved_at": time.time(),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache saved to {cache_path}", file=sys.stderr)


def load_with_cache(profiles: list[Profile], use_cache: bool = True,
                    refresh: bool = False) -> list[Turn]:
    """Load turns, using cache for unchanged files.

    If use_cache is False, skips cache entirely.
    If refresh is True, re-parses everything and updates cache.
    """
    from .loader import extract_turns_from_file, load_all_turns

    if not use_cache or refresh:
        turns = load_all_turns(profiles)
        if use_cache:
            mtimes = _collect_mtimes(profiles)
            save_cache(turns, mtimes)
        return turns

    # Load cache
    cache_path = get_cache_path()
    cached_data = _load_raw_cache()

    if cached_data is None:
        # No valid cache — full load
        turns = load_all_turns(profiles)
        mtimes = _collect_mtimes(profiles)
        save_cache(turns, mtimes)
        return turns

    cached_turns = cached_data["turns"]
    cached_mtimes = cached_data.get("file_mtimes", {})

    # Find files that need re-parsing
    current_files = {}
    for profile in profiles:
        projects_dir = profile.path / "projects"
        if projects_dir.exists():
            for f in projects_dir.rglob("*.jsonl"):
                current_files[str(f)] = (f, profile.name, f.stat().st_mtime)

    stale_files = []
    for fpath, (path_obj, profile_name, mtime) in current_files.items():
        if fpath not in cached_mtimes or cached_mtimes[fpath] < mtime:
            stale_files.append((fpath, path_obj, profile_name))

    if not stale_files:
        print(f"  Cache hit: {len(cached_turns)} turns from cache", file=sys.stderr)
        return cached_turns

    # Remove turns from stale files and re-parse them
    stale_paths = {f[0] for f in stale_files}
    # We need to track which file each turn came from — but we don't store that.
    # For simplicity, just do a full reload if there are stale files.
    # This is still fast since we only do it when files change.
    print(f"  {len(stale_files)} files changed since last cache, reloading...",
          file=sys.stderr)
    turns = load_all_turns(profiles)
    mtimes = _collect_mtimes(profiles)
    save_cache(turns, mtimes)
    return turns


def _load_raw_cache() -> dict | None:
    cache_path = get_cache_path()
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != CACHE_VERSION:
            return None
        return data
    except Exception:
        return None


def _collect_mtimes(profiles: list[Profile]) -> dict[str, float]:
    mtimes = {}
    for profile in profiles:
        projects_dir = profile.path / "projects"
        if projects_dir.exists():
            for f in projects_dir.rglob("*.jsonl"):
                mtimes[str(f)] = f.stat().st_mtime
    return mtimes
