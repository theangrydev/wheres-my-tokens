"""Generate shareable charts using matplotlib."""

import sys
from collections import defaultdict
from pathlib import Path

from .config import estimate_cost, get_pricing
from .formatting import format_tokens, format_pct, format_cost


def generate_all(ctx):
    """Generate all visualizations to the output directory."""
    output_dir = Path(ctx.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations to {output_dir}/\n", file=sys.stderr)

    _gen_daily_by_token_type(ctx, output_dir)
    _gen_session_bubble(ctx, output_dir)
    _gen_model_timeline(ctx, output_dir)

    print(f"\nDone! Charts saved to {output_dir}/", file=sys.stderr)


def _gen_daily_by_token_type(ctx, output_dir):
    """Daily usage with one subplot per token type, aggregated across all models."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    # Aggregate daily totals by token type (across all models)
    day_totals = defaultdict(lambda: {
        "input": 0, "output": 0, "cache_create": 0, "cache_read": 0,
    })
    for t in ctx.turns:
        day = t.timestamp.strftime("%Y-%m-%d")
        d = day_totals[day]
        d["input"] += t.input_tokens
        d["output"] += t.output_tokens
        d["cache_create"] += t.cache_creation_input_tokens
        d["cache_read"] += t.cache_read_input_tokens

    sorted_days = sorted(day_totals.keys())
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in sorted_days]

    token_types = [
        ("cache_read", "Cache Read", "#e74c3c"),
        ("cache_create", "Cache Create", "#f39c12"),
        ("input", "Input", "#2ecc71"),
        ("output", "Output", "#3498db"),
    ]

    # Only show token types that have data
    active_types = [
        (key, label, color) for key, label, color in token_types
        if any(day_totals[d][key] > 0 for d in sorted_days)
    ]

    n_types = len(active_types)
    if n_types == 0:
        return

    fig, axes = plt.subplots(n_types, 1, figsize=(14, 3.5 * n_types), sharex=True)
    if n_types == 1:
        axes = [axes]

    for i, (key, label, color) in enumerate(active_types):
        ax = axes[i]
        values = [day_totals[d][key] / 1e9 for d in sorted_days]
        total_tokens = sum(day_totals[d][key] for d in sorted_days)
        ax.fill_between(dates, values, alpha=0.6, color=color)
        ax.plot(dates, values, color=color, linewidth=1.5, alpha=0.8)

        ax.set_ylabel("Tokens (B)", fontsize=10)
        ax.set_title(f"{label} ({format_tokens(total_tokens)} total)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    axes[-1].set_xlabel("Date", fontsize=11)
    plt.suptitle("Daily Token Usage by Token Type", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "daily_usage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}", file=sys.stderr)


def _gen_session_bubble(ctx, output_dir):
    """Bubble chart: sessions by duration, cost, and turn count."""
    import matplotlib.pyplot as plt

    sessions = [s for s in ctx.sessions if s.duration_seconds > 0 and s.estimated_cost_usd > 0]
    if not sessions:
        return

    durations = [s.duration_seconds / 60 for s in sessions]
    costs = [s.estimated_cost_usd for s in sessions]
    sizes = [max(s.turn_count * 3, 10) for s in sessions]

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(durations, costs, s=sizes, alpha=0.5,
                          c=[s.total_tokens for s in sessions],
                          cmap="YlOrRd", edgecolors="gray", linewidth=0.5)

    ax.set_xlabel("Duration (minutes)", fontsize=12)
    ax.set_ylabel("Estimated Cost (USD)", fontsize=12)
    ax.set_title("Sessions: Duration vs Cost (bubble size = turns)", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Total Tokens")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "session_bubble.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}", file=sys.stderr)


def _gen_model_timeline(ctx, output_dir):
    """Timeline showing model usage over time."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    day_models = defaultdict(lambda: defaultdict(int))
    for t in ctx.turns:
        day = t.timestamp.strftime("%Y-%m-%d")
        day_models[day][t.model] += t.total_tokens

    sorted_days = sorted(day_models.keys())
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in sorted_days]

    model_totals = defaultdict(int)
    for dm in day_models.values():
        for m, tokens in dm.items():
            model_totals[m] += tokens
    top_models = sorted(model_totals, key=lambda m: -model_totals[m])[:5]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = [0] * len(sorted_days)

    for i, model in enumerate(top_models):
        values = [day_models[d].get(model, 0) / 1e9 for d in sorted_days]
        ax.bar(dates, values, bottom=bottom, color=colors[i % len(colors)],
               label=model[:25], width=0.8, alpha=0.8)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_ylabel("Tokens (Billions)", fontsize=12)
    ax.set_title("Model Usage Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    path = output_dir / "model_timeline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}", file=sys.stderr)
