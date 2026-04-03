"""Limit Impact Analysis — What's eating your 5-hour session limit?

Uses API-cost-weighted token usage as the limit proxy. Logistic regression
on 700+ windows with 100+ real rate-limit events shows cost-weighted usage
is the best single predictor of limit hits (AUC~0.82). No single token type
dominates — it's the weighted combination that matters.

This report shows:
1. Where your limit budget goes (by project, action, model)
2. Which patterns burn through limits fastest
3. Concrete, ranked recommendations in terms of % of limit budget saved
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import statistics

from ..config import estimate_cost, get_pricing
from ..formatting import (
    format_cost, format_number, format_pct, format_tokens,
    section_header, subsection_header, table, bar_chart,
)


def _turn_cost(config, t):
    """Estimate API-equivalent cost for a single turn."""
    return estimate_cost(config, t.model, t.input_tokens, t.output_tokens,
                         t.cache_creation_input_tokens, t.cache_read_input_tokens)


def run(ctx):
    turns = ctx.turns
    sessions = ctx.sessions
    config = ctx.config
    top_n = ctx.top_n

    print(section_header("LIMIT IMPACT ANALYSIS"))
    print("  What's eating your 5-hour session limit?\n")

    print("  Your limit budget tracks API-equivalent cost — a weighted")
    print("  combination of all token types (input, output, cache creation,")
    print("  cache read). This report shows where that budget goes.\n")

    _budget_breakdown(turns, sessions, config, top_n)
    _cost_by_action(turns, config)
    _cost_by_model(turns, config)
    _thinking_vs_response(turns, config)
    _session_cost_ranking(sessions, top_n)
    _cost_per_turn_growth(turns, sessions, config)
    _what_to_change(turns, sessions, config)


def _budget_breakdown(turns, sessions, config, top_n):
    """Show where limit budget goes by project."""
    print(subsection_header("Where Your Limit Budget Goes"))

    total_cost = sum(_turn_cost(config, t) for t in turns)
    print(f"  Total estimated limit cost: {format_cost(total_cost)}\n")

    # By project
    proj_stats = defaultdict(lambda: {"cost": 0.0, "turns": 0, "sessions": set()})
    for t in turns:
        p = proj_stats[t.project_path]
        p["cost"] += _turn_cost(config, t)
        p["turns"] += 1
        p["sessions"].add(t.session_id)

    print(f"  By Project (Top {top_n}):\n")
    sorted_projects = sorted(proj_stats.items(), key=lambda kv: -kv[1]["cost"])
    rows = []
    for proj, d in sorted_projects[:top_n]:
        avg_per_turn = d["cost"] / d["turns"] if d["turns"] > 0 else 0
        rows.append([
            _shorten(proj, 40),
            format_cost(d["cost"]),
            format_pct(d["cost"], total_cost),
            format_number(d["turns"]),
            format_cost(avg_per_turn),
        ])
    print(table(
        ["Project", "Limit Cost", "% of Total", "Turns", "Avg Cost/Turn"],
        rows, "lrrrr"
    ))


def _cost_by_action(turns, config):
    """What actions consume the most limit budget?"""
    print(subsection_header("Limit Cost by Action Type"))
    print("  What is Claude spending your limit budget on?\n")

    categories = defaultdict(lambda: {"cost": 0.0, "turns": 0})

    for t in turns:
        if not t.tool_uses:
            if t.has_thinking:
                cat = "Thinking + text response"
            else:
                cat = "Text response (no tools)"
        elif "Agent" in t.tool_uses:
            cat = "Spawning subagent"
        elif "Edit" in t.tool_uses or "Write" in t.tool_uses:
            cat = "Writing/editing code"
        elif "Read" in t.tool_uses:
            cat = "Planning after reading files"
        elif "Bash" in t.tool_uses:
            cat = "Running & analyzing commands"
        elif "Grep" in t.tool_uses or "Glob" in t.tool_uses:
            cat = "Searching codebase"
        elif any("mcp__" in tool for tool in t.tool_uses):
            cat = "MCP tool calls"
        else:
            cat = "Other tool calls"

        c = categories[cat]
        c["cost"] += _turn_cost(config, t)
        c["turns"] += 1

    total_cost = sum(c["cost"] for c in categories.values())

    rows = []
    labels = []
    values = []
    for cat in sorted(categories, key=lambda c: -categories[c]["cost"]):
        c = categories[cat]
        avg = c["cost"] / c["turns"] if c["turns"] > 0 else 0
        rows.append([
            cat[:30],
            format_cost(c["cost"]),
            format_pct(c["cost"], total_cost),
            format_number(c["turns"]),
            format_cost(avg),
        ])
        labels.append(cat[:25])
        values.append(c["cost"])

    print(table(
        ["Action", "Limit Cost", "% of Budget", "Turns", "Avg Cost/Turn"],
        rows, "lrrrr"
    ))
    print(bar_chart(labels[:8], [float(v) for v in values[:8]]))


def _cost_by_model(turns, config):
    """Compare limit cost by model."""
    print(subsection_header("Limit Cost by Model"))
    print("  Does model choice affect limit consumption?\n")

    model_stats = defaultdict(lambda: {"cost": 0.0, "turns": 0})
    for t in turns:
        if t.model == "<synthetic>":
            continue
        m = model_stats[t.model]
        m["cost"] += _turn_cost(config, t)
        m["turns"] += 1

    total_cost = sum(m["cost"] for m in model_stats.values())

    rows = []
    for model in sorted(model_stats, key=lambda m: -model_stats[m]["cost"]):
        s = model_stats[model]
        avg = s["cost"] / s["turns"] if s["turns"] > 0 else 0
        rows.append([
            model[:28],
            format_cost(s["cost"]),
            format_pct(s["cost"], total_cost),
            format_number(s["turns"]),
            format_cost(avg),
        ])
    print(table(["Model", "Limit Cost", "% of Budget", "Turns", "Avg Cost/Turn"], rows, "lrrrr"))

    # Highlight if one model costs much more per turn
    avgs = {m: model_stats[m]["cost"] / model_stats[m]["turns"]
            for m in model_stats if model_stats[m]["turns"] > 100}
    if len(avgs) >= 2:
        most = max(avgs, key=avgs.get)
        least = min(avgs, key=avgs.get)
        ratio = avgs[most] / avgs[least] if avgs[least] > 0 else 0
        if ratio > 1.5:
            print(f"  {most[:20]} costs {ratio:.1f}x more per turn than {least[:20]}.")
            print(f"  Using {least[:20]} for simple tasks would reduce limit usage.\n")


def _thinking_vs_response(turns, config):
    """Break down limit cost into thinking vs non-thinking turns."""
    print(subsection_header("Thinking vs Non-Thinking Turns"))

    thinking_turns = [t for t in turns if t.has_thinking]
    non_thinking = [t for t in turns if not t.has_thinking and t.output_tokens > 0]

    if not thinking_turns:
        print("  No extended thinking detected in your data.\n")
        return

    think_cost = sum(_turn_cost(config, t) for t in thinking_turns)
    non_think_cost = sum(_turn_cost(config, t) for t in non_thinking)
    total = think_cost + non_think_cost

    think_avg = think_cost / len(thinking_turns)
    non_think_avg = non_think_cost / len(non_thinking) if non_thinking else 0

    print(f"  Turns with thinking:    {format_number(len(thinking_turns))} "
          f"({format_cost(think_cost)} total, avg {format_cost(think_avg)}/turn)")
    print(f"  Turns without thinking: {format_number(len(non_thinking))} "
          f"({format_cost(non_think_cost)} total, avg {format_cost(non_think_avg)}/turn)")

    if non_think_avg > 0:
        ratio = think_avg / non_think_avg
        print(f"\n  Thinking turns cost {ratio:.1f}x more per turn.")
        if ratio > 3:
            print(f"  Extended thinking is a major limit driver — {format_pct(think_cost, total)} of budget.")
            print(f"  For routine tasks, consider disabling extended thinking.\n")
    print()


def _session_cost_ranking(sessions, top_n):
    """Rank sessions by limit cost."""
    print(subsection_header(f"Most Expensive Sessions (Top {top_n})"))

    total_cost = sum(s.estimated_cost_usd for s in sessions)

    sorted_sessions = sorted(sessions, key=lambda s: -s.estimated_cost_usd)[:top_n]
    rows = []
    for s in sorted_sessions:
        avg_per_turn = s.estimated_cost_usd / s.turn_count if s.turn_count > 0 else 0
        rows.append([
            s.session_id[:12],
            _shorten(s.project_path, 30),
            ", ".join(sorted(s.models))[:20],
            format_number(s.turn_count),
            format_cost(s.estimated_cost_usd),
            format_pct(s.estimated_cost_usd, total_cost),
            format_cost(avg_per_turn),
        ])
    print(table(
        ["Session", "Project", "Model", "Turns", "Cost", "% Total", "Cost/Turn"],
        rows, "lllrrrr"
    ))


def _cost_per_turn_growth(turns, sessions, config):
    """Show how cost per turn changes as sessions get longer."""
    print(subsection_header("Cost Growth Over Session Length"))
    print("  Does limit consumption per turn increase in longer sessions?\n")

    session_turns = defaultdict(list)
    for t in turns:
        session_turns[t.session_id].append(t)

    # Bucket turns by position in session
    buckets = defaultdict(list)
    for sid, s_turns in session_turns.items():
        if len(s_turns) < 20:
            continue
        s_turns.sort(key=lambda t: t.timestamp)
        for i, t in enumerate(s_turns):
            cost = _turn_cost(config, t)
            if i < 10:
                buckets["turns 1-10"].append(cost)
            elif i < 25:
                buckets["turns 11-25"].append(cost)
            elif i < 50:
                buckets["turns 26-50"].append(cost)
            elif i < 100:
                buckets["turns 51-100"].append(cost)
            else:
                buckets["turns 100+"].append(cost)

    if not buckets:
        return

    rows = []
    bucket_order = ["turns 1-10", "turns 11-25", "turns 26-50", "turns 51-100", "turns 100+"]
    first_avg = None
    for bucket in bucket_order:
        if bucket in buckets:
            vals = buckets[bucket]
            avg = sum(vals) / len(vals)
            if first_avg is None:
                first_avg = avg
            multiplier = avg / first_avg if first_avg > 0 else 0
            rows.append([
                bucket,
                format_number(len(vals)),
                format_cost(avg),
                f"{multiplier:.1f}x",
            ])
    print(table(["Turn Position", "Samples", "Avg Cost/Turn", "vs First 10"], rows, "lrrr"))
    print()


def _what_to_change(turns, sessions, config):
    """Concrete, ranked recommendations expressed in limit cost savings."""
    print(subsection_header("What to Change — Ranked by Limit Impact"))
    print("  Recommendations ranked by estimated limit cost savings.\n")

    total_cost = sum(_turn_cost(config, t) for t in turns)
    recommendations = []

    # 1. Subagent cost
    sc_turns = [t for t in turns if t.is_sidechain]
    if sc_turns:
        sc_cost = sum(_turn_cost(config, t) for t in sc_turns)
        pct = sc_cost / total_cost * 100
        if pct > 10:
            recommendations.append((
                sc_cost, pct,
                "Reduce subagent spawning",
                f"Subagents consumed {format_cost(sc_cost)} ({pct:.0f}% of total budget). "
                f"Each subagent independently generates responses and builds context. "
                f"Stay in the main conversation for sequential work."
            ))

    # 2. Thinking overhead
    thinking_turns = [t for t in turns if t.has_thinking]
    non_thinking = [t for t in turns if not t.has_thinking and t.output_tokens > 0]
    if thinking_turns and non_thinking:
        think_avg = sum(_turn_cost(config, t) for t in thinking_turns) / len(thinking_turns)
        non_think_avg = sum(_turn_cost(config, t) for t in non_thinking) / len(non_thinking)
        if think_avg > non_think_avg * 2:
            overhead = (think_avg - non_think_avg) * len(thinking_turns)
            pct = overhead / total_cost * 100
            recommendations.append((
                overhead, pct,
                "Reduce extended thinking on simple tasks",
                f"Thinking turns cost {think_avg/non_think_avg:.1f}x more per turn. "
                f"Estimated overhead: {format_cost(overhead)} ({pct:.0f}% of budget). "
                f"Use simpler prompts or Haiku for routine file operations."
            ))

    # 3. Verbose model
    model_avgs = {}
    for t in turns:
        if t.model == "<synthetic>":
            continue
        model_avgs.setdefault(t.model, []).append(_turn_cost(config, t))
    model_avgs = {m: sum(v)/len(v) for m, v in model_avgs.items() if len(v) > 100}

    if len(model_avgs) >= 2:
        most_expensive = max(model_avgs, key=model_avgs.get)
        least_expensive = min(model_avgs, key=model_avgs.get)
        if model_avgs[most_expensive] > model_avgs[least_expensive] * 1.5:
            expensive_turns = [t for t in turns if t.model == most_expensive]
            current_cost = sum(_turn_cost(config, t) for t in expensive_turns)
            projected = len(expensive_turns) * model_avgs[least_expensive]
            savings = current_cost - projected
            pct = savings / total_cost * 100
            if pct > 5:
                recommendations.append((
                    savings, pct,
                    f"Use {least_expensive[:20]} for routine tasks",
                    f"{most_expensive[:20]} costs {model_avgs[most_expensive]/model_avgs[least_expensive]:.1f}x more "
                    f"per turn than {least_expensive[:20]}. "
                    f"Switching routine tasks would save ~{format_cost(savings)} ({pct:.0f}%)."
                ))

    # 4. Long sessions cost more per turn
    session_turns = defaultdict(list)
    for t in turns:
        session_turns[t.session_id].append(t)

    long_late_cost = 0
    long_count = 0
    for sid, s_turns in session_turns.items():
        if len(s_turns) > 50:
            s_turns.sort(key=lambda t: t.timestamp)
            early_costs = [_turn_cost(config, t) for t in s_turns[:25]]
            early_avg = sum(early_costs) / len(early_costs)
            late = s_turns[50:]
            if late:
                late_avg = sum(_turn_cost(config, t) for t in late) / len(late)
                if late_avg > early_avg * 1.3:
                    overhead = sum(_turn_cost(config, t) - early_avg for t in late
                                   if _turn_cost(config, t) > early_avg)
                    long_late_cost += overhead
                    long_count += 1

    if long_late_cost > 0 and long_count > 0:
        pct = long_late_cost / total_cost * 100
        recommendations.append((
            long_late_cost, pct,
            "Start fresh sessions more often",
            f"In {long_count} long sessions, turns after #50 had excess cost "
            f"({format_cost(long_late_cost)}, {pct:.0f}% of budget). "
            f"Context growth makes later turns more expensive. "
            f"Fresh sessions keep costs lower."
        ))

    # 5. Write/Edit tool generates lots of output (full code blocks)
    write_turns = [t for t in turns if any(tool in t.tool_uses for tool in ["Write", "Edit"])]
    other_turns = [t for t in turns if t.tool_uses and not any(tool in t.tool_uses for tool in ["Write", "Edit"])]
    if write_turns and other_turns:
        write_avg = sum(_turn_cost(config, t) for t in write_turns) / len(write_turns)
        other_avg = sum(_turn_cost(config, t) for t in other_turns) / len(other_turns)
        if write_avg > other_avg * 2:
            total_write_cost = sum(_turn_cost(config, t) for t in write_turns)
            pct = total_write_cost / total_cost * 100
            if pct > 15:
                recommendations.append((
                    total_write_cost, pct,
                    "Prefer Edit over Write for modifications",
                    f"Write/Edit turns cost {write_avg/other_avg:.1f}x more per turn "
                    f"({format_cost(total_write_cost)} total, {pct:.0f}%). "
                    f"Edit sends only the diff; Write sends the full file. "
                    f"For modifications, always prefer targeted Edit over full file Write."
                ))

    # Sort by impact
    recommendations.sort(key=lambda r: -r[0])

    if not recommendations:
        print("  No significant optimization opportunities detected.\n")
        return

    for i, (savings, pct, title, detail) in enumerate(recommendations, 1):
        print(f"  {i}. {title} (saves ~{pct:.0f}% of limit budget)")
        print(f"     {detail}")
        print()

    total_savings_pct = sum(r[1] for r in recommendations)
    print(f"  Combined potential: reduce limit cost by ~{total_savings_pct:.0f}%")
    print(f"  (some recommendations overlap, so actual savings will be less)\n")


def _shorten(path: str, max_len: int) -> str:
    import os
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home):]
    if len(path) > max_len:
        return "..." + path[-(max_len - 3):]
    return path
