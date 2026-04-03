"""Summary dashboard report."""

from collections import defaultdict

from ..config import estimate_cost
from ..formatting import (
    format_cost, format_number, format_pct, format_tokens,
    section_header, subsection_header, table,
)


def run(ctx):
    turns = ctx.turns
    sessions = ctx.sessions
    config = ctx.config

    print(section_header("WHERE'S MY TOKENS — Summary Dashboard"))

    # Overall stats
    total_input = sum(t.input_tokens for t in turns)
    total_output = sum(t.output_tokens for t in turns)
    total_cache_create = sum(t.cache_creation_input_tokens for t in turns)
    total_cache_read = sum(t.cache_read_input_tokens for t in turns)
    total_all = total_input + total_output + total_cache_create + total_cache_read

    dates = [t.timestamp for t in turns]
    min_date = min(dates).strftime("%Y-%m-%d")
    max_date = max(dates).strftime("%Y-%m-%d")

    # Per-profile breakdown
    profiles_seen = sorted(set(t.profile_name for t in turns))
    print(f"  Profiles:     {', '.join(profiles_seen)}")
    print(f"  Date range:   {min_date} to {max_date}")
    print(f"  Sessions:     {format_number(len(sessions))}")
    print(f"  API turns:    {format_number(len(turns))}")
    print()

    # Token breakdown
    print(subsection_header("Token Breakdown"))
    rows = [
        ["Input tokens", format_tokens(total_input), format_number(total_input), format_pct(total_input, total_all)],
        ["Output tokens", format_tokens(total_output), format_number(total_output), format_pct(total_output, total_all)],
        ["Cache creation", format_tokens(total_cache_create), format_number(total_cache_create), format_pct(total_cache_create, total_all)],
        ["Cache read", format_tokens(total_cache_read), format_number(total_cache_read), format_pct(total_cache_read, total_all)],
        ["─" * 15, "─" * 8, "─" * 20, "─" * 6],
        ["TOTAL", format_tokens(total_all), format_number(total_all), "100%"],
    ]
    print(table(["Type", "Short", "Exact", "%"], rows, "lrrr"))

    # Per-model breakdown
    print(subsection_header("By Model"))
    model_stats = defaultdict(lambda: {"input": 0, "output": 0, "cache_create": 0, "cache_read": 0, "turns": 0})
    for t in turns:
        m = model_stats[t.model]
        m["input"] += t.input_tokens
        m["output"] += t.output_tokens
        m["cache_create"] += t.cache_creation_input_tokens
        m["cache_read"] += t.cache_read_input_tokens
        m["turns"] += 1

    model_rows = []
    for model in sorted(model_stats, key=lambda m: -(model_stats[m]["input"] + model_stats[m]["output"] + model_stats[m]["cache_create"] + model_stats[m]["cache_read"])):
        s = model_stats[model]
        total = s["input"] + s["output"] + s["cache_create"] + s["cache_read"]
        cost = estimate_cost(config, model, s["input"], s["output"], s["cache_create"], s["cache_read"])
        model_rows.append([
            model[:30],
            format_number(s["turns"]),
            format_tokens(total),
            format_tokens(s["cache_read"]),
            format_tokens(s["output"]),
            format_cost(cost),
        ])
    print(table(["Model", "Turns", "Total", "Cache Read", "Output", "Est. Cost"], model_rows, "lrrrrr"))

    # Per-profile breakdown
    if len(profiles_seen) > 1:
        print(subsection_header("By Profile"))
        profile_stats = defaultdict(lambda: {"turns": 0, "tokens": 0, "sessions": 0, "cost": 0.0})
        for t in turns:
            ps = profile_stats[t.profile_name]
            ps["turns"] += 1
            ps["tokens"] += t.total_tokens
            cost = estimate_cost(config, t.model, t.input_tokens, t.output_tokens,
                                 t.cache_creation_input_tokens, t.cache_read_input_tokens)
            ps["cost"] += cost
        for s in sessions:
            profile_stats[s.profile_name]["sessions"] += 1

        profile_rows = []
        for pname in sorted(profile_stats, key=lambda p: -profile_stats[p]["tokens"]):
            ps = profile_stats[pname]
            profile_rows.append([
                pname,
                format_number(ps["sessions"]),
                format_number(ps["turns"]),
                format_tokens(ps["tokens"]),
                format_cost(ps["cost"]),
            ])
        print(table(["Profile", "Sessions", "Turns", "Tokens", "Est. Cost"], profile_rows, "lrrrr"))

    # Quick insight
    print(subsection_header("Key Insight"))
    cache_pct = total_cache_read / total_all * 100 if total_all > 0 else 0
    print(f"  Cache reads account for {cache_pct:.1f}% of all tokens.")
    print(f"  This means the conversation context (system prompt + history + tool results)")
    print(f"  is re-read from cache on every turn. Longer sessions = exponentially more tokens.")
    print()
