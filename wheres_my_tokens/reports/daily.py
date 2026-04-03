"""Daily usage breakdown report."""

from collections import defaultdict

from ..config import estimate_cost
from ..formatting import (
    format_cost, format_number, format_tokens,
    section_header, subsection_header, table, sparkline,
)


def run(ctx):
    turns = ctx.turns
    config = ctx.config

    print(section_header("Daily Usage Report"))

    # Aggregate by day
    days = defaultdict(lambda: {
        "input": 0, "output": 0, "cache_create": 0, "cache_read": 0,
        "turns": 0, "sessions": set(), "cost": 0.0, "models": defaultdict(int),
    })

    for t in turns:
        day = t.timestamp.strftime("%Y-%m-%d")
        d = days[day]
        d["input"] += t.input_tokens
        d["output"] += t.output_tokens
        d["cache_create"] += t.cache_creation_input_tokens
        d["cache_read"] += t.cache_read_input_tokens
        d["turns"] += 1
        d["sessions"].add(t.session_id)
        d["models"][t.model] += 1
        d["cost"] += estimate_cost(config, t.model, t.input_tokens, t.output_tokens,
                                    t.cache_creation_input_tokens, t.cache_read_input_tokens)

    sorted_days = sorted(days.keys())

    # Daily table (last 30 days)
    recent = sorted_days[-30:]
    print(subsection_header("Last 30 Days"))
    rows = []
    for day in recent:
        d = days[day]
        total = d["input"] + d["output"] + d["cache_create"] + d["cache_read"]
        top_model = max(d["models"], key=d["models"].get) if d["models"] else ""
        rows.append([
            day,
            format_number(len(d["sessions"])),
            format_number(d["turns"]),
            format_tokens(total),
            format_tokens(d["output"]),
            format_cost(d["cost"]),
            top_model[:20],
        ])
    print(table(
        ["Date", "Sessions", "Turns", "Total Tokens", "Output", "Est. Cost", "Top Model"],
        rows, "lrrrrrr"
    ))

    # Sparkline of daily totals
    print(subsection_header("Daily Token Trend"))
    daily_totals = [
        days[d]["input"] + days[d]["output"] + days[d]["cache_create"] + days[d]["cache_read"]
        for d in sorted_days
    ]
    if daily_totals:
        line = sparkline(daily_totals)
        print(f"  {sorted_days[0]} {line} {sorted_days[-1]}")
        print(f"  Min: {format_tokens(min(daily_totals))}  "
              f"Max: {format_tokens(max(daily_totals))}  "
              f"Avg: {format_tokens(sum(daily_totals) / len(daily_totals))}")
    print()

    # Spike days (top 10 by total tokens)
    print(subsection_header("Top 10 Spike Days"))
    spike_days = sorted(days.items(), key=lambda kv: -(kv[1]["input"] + kv[1]["output"] + kv[1]["cache_create"] + kv[1]["cache_read"]))[:10]
    rows = []
    for day, d in spike_days:
        total = d["input"] + d["output"] + d["cache_create"] + d["cache_read"]
        rows.append([
            day,
            format_tokens(total),
            format_number(d["turns"]),
            format_number(len(d["sessions"])),
            format_cost(d["cost"]),
        ])
    print(table(["Date", "Total Tokens", "Turns", "Sessions", "Est. Cost"], rows, "lrrrr"))

    # Day-of-week pattern
    print(subsection_header("Day of Week Pattern"))
    dow_stats = defaultdict(lambda: {"tokens": 0, "days": 0})
    for day in sorted_days:
        from datetime import datetime
        dt = datetime.strptime(day, "%Y-%m-%d")
        dow_name = dt.strftime("%A")
        d = days[day]
        total = d["input"] + d["output"] + d["cache_create"] + d["cache_read"]
        dow_stats[dow_name]["tokens"] += total
        dow_stats[dow_name]["days"] += 1

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rows = []
    for dow in dow_order:
        s = dow_stats.get(dow, {"tokens": 0, "days": 0})
        avg = s["tokens"] / s["days"] if s["days"] > 0 else 0
        rows.append([dow, format_number(s["days"]), format_tokens(s["tokens"]), format_tokens(avg)])
    print(table(["Day", "Active Days", "Total Tokens", "Avg/Day"], rows, "lrrr"))
