"""CLI entry point with argparse subcommands."""

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .cache import load_with_cache
from .config import discover_profiles, load_config
from .models import Turn, Session


def build_sessions(turns: list[Turn], config: dict) -> list[Session]:
    """Aggregate turns into sessions."""
    from .config import estimate_cost

    by_session: dict[str, list[Turn]] = {}
    for t in turns:
        by_session.setdefault(t.session_id, []).append(t)

    sessions = []
    for sid, session_turns in by_session.items():
        session_turns.sort(key=lambda t: t.timestamp)
        first = session_turns[0]
        s = Session(
            session_id=sid,
            profile_name=first.profile_name,
            project_dir=first.project_dir,
            project_path=first.project_path,
        )
        for t in session_turns:
            s.models.add(t.model)
            s.turn_count += 1
            s.total_input_tokens += t.input_tokens
            s.total_output_tokens += t.output_tokens
            s.total_cache_creation_tokens += t.cache_creation_input_tokens
            s.total_cache_read_tokens += t.cache_read_input_tokens
            if s.first_timestamp is None or t.timestamp < s.first_timestamp:
                s.first_timestamp = t.timestamp
            if s.last_timestamp is None or t.timestamp > s.last_timestamp:
                s.last_timestamp = t.timestamp
            for tool in t.tool_uses:
                s.tool_call_counts[tool] = s.tool_call_counts.get(tool, 0) + 1
            if t.has_thinking:
                s.thinking_turn_count += 1
            s.estimated_cost_usd += estimate_cost(
                config, t.model,
                t.input_tokens, t.output_tokens,
                t.cache_creation_input_tokens, t.cache_read_input_tokens
            )
        sessions.append(s)
    return sessions


def apply_filters(turns: list[Turn], args) -> list[Turn]:
    """Apply CLI filter arguments to turns."""
    from datetime import datetime, timezone

    filtered = turns
    if hasattr(args, "since") and args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        filtered = [t for t in filtered if t.timestamp >= since]
    if hasattr(args, "until") and args.until:
        until = datetime.strptime(args.until, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        filtered = [t for t in filtered if t.timestamp <= until]
    if hasattr(args, "project") and args.project:
        pat = args.project.lower()
        filtered = [t for t in filtered if pat in t.project_path.lower()]
    if hasattr(args, "model") and args.model:
        pat = args.model.lower()
        filtered = [t for t in filtered if pat in t.model.lower()]
    if hasattr(args, "profile_filter") and args.profile_filter:
        pat = args.profile_filter.lower()
        filtered = [t for t in filtered if pat in t.profile_name.lower()]
    return filtered


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="wheres-my-tokens",
        description=(
            "Find out what's eating your Claude Code session limits. "
            "Parses local Claude Code data to show where tokens go "
            "and what to change."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--profile", dest="profiles", action="append",
                        help="Claude profile directory (default: auto-discover)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip loading/saving parsed data cache")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Force re-parse all files")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--since", type=str, help="Only data from this date (YYYY-MM-DD)")
    parser.add_argument("--until", type=str, help="Only data up to this date (YYYY-MM-DD)")
    parser.add_argument("--project", type=str, help="Filter by project name")
    parser.add_argument("--model", type=str, help="Filter by model name")
    parser.add_argument("--profile-filter", type=str, dest="profile_filter",
                        help="Filter by profile name")
    parser.add_argument("-n", type=int, default=None, help="Top-N items in reports")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for charts")

    subs = parser.add_subparsers(dest="command", help="Report to generate")

    # Core reports
    subs.add_parser("limits", help="What's eating your session limit? (start here)")
    subs.add_parser("summary", help="Quick usage overview")
    subs.add_parser("daily", help="Usage trends over time")
    subs.add_parser("visualize", help="Generate shareable charts")

    # Advanced
    subs.add_parser("analyze", help="Reverse-engineer limit formula: regression, scatter plots, budget timeline")

    subs.add_parser("all", help="Run all text reports")
    subs.add_parser("clean", help="Delete generated output and caches")

    args = parser.parse_args(argv)

    if not args.command:
        # Default to limits report (the main thing users want)
        print("Usage: wheres-my-tokens <command>\n", file=sys.stderr)
        print("Reports:", file=sys.stderr)
        print("  limits      What's eating your session limit? (start here)", file=sys.stderr)
        print("  summary     Quick usage overview", file=sys.stderr)
        print("  daily       Usage trends over time", file=sys.stderr)
        print("  visualize   Generate shareable charts\n", file=sys.stderr)
        print("Advanced:", file=sys.stderr)
        print("  analyze     Reverse-engineer limit formula: regression, scatter plots, budget timeline\n", file=sys.stderr)
        print("Other:", file=sys.stderr)
        print("  all         Run all text reports", file=sys.stderr)
        print("  clean       Delete generated output and caches\n", file=sys.stderr)
        print("Run 'wheres-my-tokens limits' to get started.", file=sys.stderr)
        return

    # Handle clean command early (doesn't need data loading)
    if args.command == "clean":
        _run_clean(config_path=Path(args.config) if args.config else None,
                   output_dir=args.output_dir)
        return

    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    top_n = args.n or config.get("top_n", 20)
    output_dir = Path(args.output_dir or config.get("output_dir", "./output"))

    # Discover profiles
    profiles = discover_profiles(args.profiles)
    if not profiles:
        print("No Claude Code profiles found. Check ~/.claude* directories.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(profiles)} profile(s): {', '.join(p.label for p in profiles)}",
          file=sys.stderr)

    # Load data
    t0 = time.time()
    turns = load_with_cache(
        profiles,
        use_cache=not args.no_cache,
        refresh=args.refresh_cache,
    )
    elapsed = time.time() - t0
    print(f"  Loaded {len(turns)} turns in {elapsed:.1f}s", file=sys.stderr)

    # Apply filters
    if args.command == "analyze" and any([
        getattr(args, "project", None),
        getattr(args, "model", None),
        getattr(args, "since", None),
        getattr(args, "until", None)
    ]):
        print("Error: --project, --model, --since, and --until filters cannot be used with the 'analyze' command.", file=sys.stderr)
        print("The statistical calibration requires your complete, unfiltered token history to calculate limit budgets accurately.", file=sys.stderr)
        sys.exit(1)

    turns = apply_filters(turns, args)
    if not turns:
        print("No data after applying filters.", file=sys.stderr)
        sys.exit(1)

    print(f"  Analyzing {len(turns)} turns...\n", file=sys.stderr)

    # Build sessions
    sessions = build_sessions(turns, config)

    # Dispatch
    ctx = ReportContext(turns=turns, sessions=sessions, config=config,
                        profiles=profiles, top_n=top_n, output_dir=output_dir)

    text_reports = ["limits", "summary", "daily"]

    if args.command == "all":
        for cmd in text_reports:
            _run_report(cmd, ctx)
    elif args.command == "visualize":
        _run_visualize(ctx)
    else:
        _run_report(args.command, ctx)


class ReportContext:
    """Shared context passed to all report modules."""
    def __init__(self, turns, sessions, config, profiles, top_n, output_dir):
        self.turns = turns
        self.sessions = sessions
        self.config = config
        self.profiles = profiles
        self.top_n = top_n
        self.output_dir = output_dir


def _run_report(name: str, ctx: ReportContext):
    report_map = {
        "limits": "wheres_my_tokens.reports.limit_impact",
        "summary": "wheres_my_tokens.reports.summary",
        "daily": "wheres_my_tokens.reports.daily",
        "analyze": "wheres_my_tokens.reports.analyze",
    }
    module_name = report_map.get(name)
    if not module_name:
        print(f"Unknown report: {name}", file=sys.stderr)
        return
    import importlib
    mod = importlib.import_module(module_name)
    mod.run(ctx)


def _run_visualize(ctx: ReportContext):
    from .visualizations import generate_all
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    generate_all(ctx)


def _run_clean(config_path=None, output_dir=None):
    """Delete generated output and caches."""
    import shutil
    from .cache import get_cache_path
    from .config import load_config

    config = load_config(config_path)
    out = Path(output_dir or config.get("output_dir", "./output"))

    cleaned = []

    # Delete output directory
    if out.exists():
        count = sum(1 for _ in out.iterdir())
        shutil.rmtree(out)
        cleaned.append(f"output ({count} files): {out}")

    # Delete pickle cache
    cache_path = get_cache_path()
    if cache_path.exists():
        size = cache_path.stat().st_size / 1024 / 1024
        cache_path.unlink()
        cleaned.append(f"cache ({size:.1f} MB): {cache_path}")

    if cleaned:
        print("Cleaned:")
        for item in cleaned:
            print(f"  {item}")
    else:
        print("Nothing to clean.")
