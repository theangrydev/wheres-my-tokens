"""Calibration Diagnostics — investigate the unknown limit model.

Data science approach to reverse-engineering Anthropic's limit formula:
1. Correlation analysis of each feature vs budget-at-limit
2. Feature scatter plots with trend lines
3. Budget timeline (cost at each limit hit over time)
4. Logistic regression (binary classification: hit limit or not)
"""

import json
import re
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from pathlib import Path

from ..config import estimate_cost
from ..formatting import (
    format_cost, format_number, format_tokens,
    section_header, subsection_header, table,
)


# The two limits are separate buckets with separate causes, so each gets its
# own window and its own fit. Mixing them is not a rounding error: the 5 hours
# before a WEEKLY hit are usually unremarkable, so labelling them a positive
# teaches the fit that ordinary usage hits limits.
#
# stride is how far apart the non-hit (control) windows are sampled. For the
# 5-hour bucket a stride equal to the window gives independent, non-overlapping
# controls and thousands of them. A week-long window stepped by a week yields
# only ~45 controls per account over a year, which is too few to fit, so the
# weekly bucket samples daily. Those controls OVERLAP and are therefore not
# independent -- read the weekly AUC as a ranking, not as a calibrated
# probability, and do not compare its confidence to the 5-hour model's.
WINDOW_SPECS = {
    "5h": {
        "key": "5h",
        "label": "5-hour",
        "duration": timedelta(hours=5),
        "stride": timedelta(hours=5),
        "limit_types": ("5-hour",),
        "overlapping_controls": False,
    },
    "1w": {
        "key": "1w",
        "label": "weekly",
        "duration": timedelta(days=7),
        "stride": timedelta(days=1),
        "limit_types": ("weekly",),
        "overlapping_controls": True,
    },
}

# Promotional / one-off balances, not a recurring constraint: extra usage bought
# or granted on top of the plan. Classified so they cannot land in "unknown" and
# trip the wording guard, but never fitted.
PROMO_LIMIT_TYPES = ("extra-usage", "credits")

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# "resets 4pm (UTC)" / "resets Apr 7, 4pm" / "resets Aug 3 at 9pm (Europe/London)"
_RESET_RE = re.compile(
    r"resets\s+"
    r"(?:(?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+"
    r"(?P<day>\d{1,2})\s*(?:,|\s+at)?\s*)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)",
    re.IGNORECASE,
)


def _reset_delta_hours(text, event_ts):
    """Hours from the event to the reset time quoted in its message text.

    Returns None if no reset time is present. The timezone label in the text
    ("(UTC)" / "(Europe/London)") is deliberately NOT resolved: callers use this
    only to separate a same-day reset from one days away, so an hour of DST slop
    cannot change the answer, and resolving it would add a tzdata dependency for
    no gain.
    """
    m = _RESET_RE.search(text or "")
    if not m:
        return None
    hour = int(m.group("hour")) % 12
    if m.group("ampm").lower() == "pm":
        hour += 12
    minute = int(m.group("minute") or 0)

    if m.group("mon"):
        month = _MONTHS[m.group("mon").lower()]
        day = int(m.group("day"))
        year = event_ts.year
        # A reset quoted in December against a January event is last year's
        # wording wrapping the year boundary, not a reset 11 months out.
        if month - event_ts.month > 6:
            year -= 1
        elif event_ts.month - month > 6:
            year += 1
        try:
            reset = event_ts.replace(year=year, month=month, day=day,
                                     hour=hour, minute=minute,
                                     second=0, microsecond=0)
        except ValueError:
            return None
    else:
        # A bare clock time is the next occurrence of that time.
        reset = event_ts.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if reset < event_ts:
            reset += timedelta(days=1)

    return (reset - event_ts).total_seconds() / 3600.0


def _find_rate_limit_events(profiles):
    """Find all rate_limit events from JSONL conversation files and telemetry."""
    events = []

    for profile in profiles:
        # Scan conversation JSONL files for <synthetic> rate_limit messages
        projects_dir = profile.path / "projects"
        if not projects_dir.exists():
            continue

        for jsonl_file in projects_dir.rglob("*.jsonl"):
            try:
                with open(jsonl_file) as f:
                    prev_session_id = None
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        sid = msg.get("sessionId", "")
                        if sid:
                            prev_session_id = sid

                        # error field is at OUTER message level, not inside message
                        error = msg.get("error", "")
                        # `invalid_request` is a limit hit in disguise: when the
                        # limit lands on a full context window, auto-compaction
                        # cannot run (it needs an API call of its own), so the
                        # turn dies as "Prompt is too long" with the real cause
                        # quoted inside it. Keying on error == "rate_limit"
                        # alone drops the entire class.
                        if error in ("rate_limit", "invalid_request"):
                            ts_str = msg.get("timestamp", "")
                            try:
                                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            except (ValueError, AttributeError):
                                continue

                            # Extract reset time from message text
                            message = msg.get("message", {})
                            content = message.get("content", []) if isinstance(message, dict) else []
                            text = ""
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and c.get("type") == "text":
                                        text = c.get("text", "")

                            # Most invalid_request errors are ordinary bad
                            # requests and are not limit events at all. Keep
                            # only the ones that quote a limit as their cause.
                            if error == "invalid_request" and "limit" not in text.lower():
                                continue

                            events.append({
                                "timestamp": ts,
                                "profile": profile.name,
                                "session_id": prev_session_id or "",
                                "text": text,
                                "source": "conversation",
                                "project_dir": jsonl_file.parent.name,
                            })
            except Exception:
                continue

        # Also scan telemetry for limit status events
        telemetry_dir = profile.path / "telemetry"
        if telemetry_dir.exists():
            for tf in telemetry_dir.glob("1p_failed_events*.json"):
                try:
                    with open(tf) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for event in data:
                            ed = event.get("event_data", {})
                            if ed.get("event_name") == "tengu_claudeai_limits_status_changed":
                                meta = ed.get("additional_metadata", {})
                                status = meta.get("status", "")
                                ts_str = ed.get("client_timestamp", "")
                                try:
                                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                except (ValueError, AttributeError):
                                    continue
                                events.append({
                                    "timestamp": ts,
                                    "profile": profile.name,
                                    "session_id": ed.get("session_id", ""),
                                    "text": f"status={status}",
                                    "source": "telemetry",
                                    "status": status,
                                    "hours_till_reset": meta.get("hoursTillReset"),
                                    "fallback_available": meta.get("unifiedRateLimitFallbackAvailable"),
                                    "model": ed.get("model", ""),
                                })
                except Exception:
                    continue

    # Classify limit type from message text.
    #
    # Prefer EXPLICIT wording and fall back to the reset delta. Do NOT infer the
    # type from whether the text carries a date: that heuristic is wrong in both
    # directions and was silently wrong for months.
    #   - "hit your weekly limit - resets 9pm"     weekly, no date
    #   - "hit your limit - resets Apr 7, 4pm"     5-hour hit at 11am, has a date
    # The wording also CHANGED in June 2026, from "hit your limit" to "hit your
    # weekly limit", and a plain `"hit your limit" in text` substring test does
    # not match the new string. That dropped every event after 2026-05 into
    # "unknown" (1,863 of them) and emptied the budget timeline. The
    # unknown-rate guard below exists so the next rewording is loud, not silent.
    for e in events:
        text = e.get("text", "")
        text_lower = text.lower()
        if ("5-hour" in text_lower or "5 hour" in text_lower
                # Renamed to "session limit" alongside the "weekly limit"
                # rename. Same rolling ~5h bucket, new label.
                or "session limit" in text_lower):
            e["limit_type"] = "5-hour"
        elif "extra usage" in text_lower:
            e["limit_type"] = "extra-usage"
        elif "usage credits" in text_lower:
            # Model-scoped credit balance, exhausted independently of the
            # rolling limits. Its own class: a pinned model stops dead here
            # while the account still has ordinary headroom.
            e["limit_type"] = "credits"
        elif "weekly limit" in text_lower:
            e["limit_type"] = "weekly"
        elif "rate limit reached" in text_lower:
            e["limit_type"] = "api-ratelimit"
        elif "hit your limit" in text_lower:
            delta = _reset_delta_hours(text, e["timestamp"])
            if delta is None:
                e["limit_type"] = "unknown"
            else:
                # A 5-hour window can never reset more than 5h out; allow an
                # hour of slop for the timezone label we do not resolve.
                e["limit_type"] = "5-hour" if delta <= 6 else "weekly"
        elif e.get("hours_till_reset") and e["hours_till_reset"] > 24:
            e["limit_type"] = "weekly"
        elif e.get("hours_till_reset") and e["hours_till_reset"] <= 5:
            e["limit_type"] = "5-hour"
        else:
            e["limit_type"] = "unknown"

    # Unknowns are the failure mode this whole classifier has, and it is SILENT:
    # downstream stages filter on limit_type, so a wording change does not raise
    # anything, it just empties the charts and shrinks the training set. Say so
    # loudly, and print a sample so the new wording can be read off directly.
    conv = [e for e in events if e["source"] == "conversation"]
    unknown = [e for e in conv if e["limit_type"] == "unknown"]
    if conv and len(unknown) / len(conv) > 0.02:
        from collections import Counter
        samples = Counter(e["text"][:100] for e in unknown)
        print(f"  WARNING: {len(unknown)}/{len(conv)} limit events "
              f"({100 * len(unknown) / len(conv):.0f}%) could not be classified.")
        print("  The message wording has probably changed. Most common unmatched text:")
        for text, n in samples.most_common(3):
            print(f"    {n:>5}x  {text}")
        print()

    # Deduplicate: keep only the FIRST hit per limit window
    # (users retry after hitting limit, creating duplicate events)
    events.sort(key=lambda e: e["timestamp"])
    deduped = []
    for e in events:
        if not deduped:
            deduped.append(e)
        elif (e["timestamp"] - deduped[-1]["timestamp"]).total_seconds() > 300:
            # More than 5 minutes apart = probably a new event
            deduped.append(e)
        elif e["source"] == "telemetry" and deduped[-1]["source"] != "telemetry":
            deduped[-1] = e

    return deduped


def _calculate_window_costs(limit_events, sorted_turns, config, group_of=None,
                            duration=timedelta(hours=5)):
    """For each rate-limit event, total the cost in the window preceding it."""
    points = []
    group_of = group_of or {}

    # Only use conversation-source rate_limit events (not "allowed" telemetry)
    actual_limits = [e for e in limit_events
                     if e["source"] == "conversation"
                     or e.get("status") in ("rate_limited", "allowed_warning")]

    # Index turns by budget group ONCE, keeping each list in timestamp order.
    # A per-event scan of the whole turn list is O(events x turns), which is
    # ~4e9 comparisons on a real fleet archive and takes longer than the rest
    # of the report put together.
    by_group = {}
    for t in sorted_turns:
        if t.model == "<synthetic>":
            continue
        by_group.setdefault(group_of.get(t.profile_name, t.profile_name), []).append(t)
    group_times = {g: [t.timestamp for t in ts_] for g, ts_ in by_group.items()}

    for event in actual_limits:
        ts = event["timestamp"]
        window_start = ts - duration
        group = event.get("group") or group_of.get(event["profile"], event["profile"])

        # Take the turns in this 5h window that drew on the SAME BUDGET. Limits
        # are per account, so every profile dir logged into that account counts
        # against the window -- filtering to the one dir that happened to log
        # the error undercounts the usage that actually caused it, by however
        # many dirs share the account.
        turns_in_group = by_group.get(group)
        if not turns_in_group:
            continue
        times = group_times[group]
        lo = bisect_left(times, window_start)
        hi = bisect_right(times, ts)
        window_turns = turns_in_group[lo:hi]

        if not window_turns:
            continue

        # Calculate total cost in window
        total_cost = sum(
            estimate_cost(config, t.model, t.input_tokens, t.output_tokens,
                          t.cache_creation_input_tokens, t.cache_read_input_tokens)
            for t in window_turns
        )

        # Also track token breakdown
        total_input = sum(t.input_tokens for t in window_turns)
        total_output = sum(t.output_tokens for t in window_turns)
        total_cc = sum(t.cache_creation_input_tokens for t in window_turns)
        total_cr = sum(t.cache_read_input_tokens for t in window_turns)

        points.append({
            "event": event,
            "window_start": window_start,
            "window_end": ts,
            "cost": total_cost,
            "turns": len(window_turns),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_creation_tokens": total_cc,
            "cache_read_tokens": total_cr,
            "total_tokens": total_input + total_output + total_cc + total_cr,
            "is_hard_limit": event["source"] == "conversation",
            "status": event.get("status", "rate_limited"),
        })

    return points


def run(ctx):
    turns = ctx.turns
    config = ctx.config
    profiles = ctx.profiles
    output_dir = Path(ctx.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(section_header("CALIBRATION DIAGNOSTICS"))
    print("  Investigating the unknown limit model with data science techniques.\n")

    # Usage limits are per ACCOUNT, not per profile directory. Several profile
    # dirs can share one account (e.g. a fleet of agents all logged in as the
    # same user), and then a per-dir window sees only that dir's slice of the
    # usage that actually caused the limit event -- which biases the fitted
    # threshold down by however many dirs share the account. Group by account
    # unless explicitly asked for the old per-dir view.
    group_by = getattr(ctx, "group_by", "account") or "account"
    if group_by == "account":
        group_of = {p.name: (p.email or p.name) for p in profiles}
    else:
        group_of = {p.name: p.name for p in profiles}

    shared = {}
    for p in profiles:
        shared.setdefault(p.email or p.name, []).append(p.name)
    multi = {k: v for k, v in shared.items() if len(v) > 1}
    print(f"  Grouping windows by: {group_by}")
    if multi:
        for acct, dirs in sorted(multi.items()):
            print(f"    {acct}: {len(dirs)} profile dirs share this budget "
                  f"({', '.join(sorted(dirs))})")
        if group_by != "account":
            print("    WARNING: grouping by dir splits those shared budgets, so each\n"
                  "    window undercounts the usage that caused its limit event.")
    print()

    # Get rate limit events once -- they are shared by every window pass.
    limit_events = _find_rate_limit_events(profiles)
    for e in limit_events:
        e["group"] = group_of.get(e["profile"], e["profile"])
    sorted_turns = sorted(turns, key=lambda t: t.timestamp)

    # Show the whole event population BEFORE any window narrows it. Each pass
    # legitimately keeps one type, but when that filter silently ate everything
    # after 2026-05 it looked exactly like "no limits were hit".
    from collections import Counter
    conv = [e for e in limit_events if e["source"] == "conversation"]
    breakdown = Counter(e["limit_type"] for e in conv)
    if breakdown:
        print("  All limit events by type: "
              + ", ".join(f"{n} {t}" for t, n in breakdown.most_common()))
        print(f"  Most recent limit event: "
              f"{max(e['timestamp'] for e in conv):%Y-%m-%d %H:%M} UTC")
        promo = sum(n for t, n in breakdown.items() if t in PROMO_LIMIT_TYPES)
        if promo:
            print(f"  Excluded as promotional/one-off balances: {promo} "
                  f"({', '.join(PROMO_LIMIT_TYPES)})")
    print()

    requested = getattr(ctx, "window", "both") or "both"
    keys = list(WINDOW_SPECS) if requested == "both" else [requested]

    for key in keys:
        spec = WINDOW_SPECS[key]
        # Each window writes to its own subdirectory so the two passes' charts
        # do not overwrite each other.
        win_dir = output_dir / key
        win_dir.mkdir(parents=True, exist_ok=True)
        _run_window(spec, limit_events, sorted_turns, config, group_of, win_dir)


def _run_window(spec, limit_events, sorted_turns, config, group_of, output_dir):
    """One full calibration pass for a single limit bucket."""
    label = spec["label"]
    print(section_header(f"{label.upper()} LIMIT"))
    print(f"  Window: {spec['duration']}, control windows sampled every {spec['stride']}.")
    if spec["overlapping_controls"]:
        print("  NOTE: controls overlap at this stride, so they are not independent")
        print("  samples. Read the AUC as a ranking, not a calibrated probability.")
    print(f"  Charts: {output_dir}\n")

    points = _calculate_window_costs(limit_events, sorted_turns, config, group_of,
                                     duration=spec["duration"])
    raw = [p for p in points
           if p["is_hard_limit"] and p["event"].get("limit_type") in spec["limit_types"]]
    unique = _dedupe_by_reset_window(raw)

    print(f"  Total calibration points: {len(points)}")
    print(f"  {label} limit events: {len(raw)} raw, {len(unique)} unique windows\n")

    if len(unique) < 5:
        print(f"  Not enough {label} limit events for diagnostics.\n")
        return

    _correlation_analysis(unique)
    _feature_scatter_plots(unique, output_dir, label=label)
    _budget_timeline(unique, sorted_turns, config, output_dir,
                     group_of=group_of, duration=spec["duration"],
                     stride=spec["stride"], label=label,
                     all_limit_events=[e for e in limit_events
                                       if e["source"] == "conversation"])
    _logistic_regression_analysis(limit_events, sorted_turns, config, output_dir,
                                  group_of=group_of,
                                  limit_types=spec["limit_types"],
                                  duration=spec["duration"],
                                  stride=spec["stride"],
                                  label=label)


def _correlation_analysis(points):
    """Compute correlation between each feature and the budget (cost at limit)."""
    print(subsection_header("Correlation Analysis"))
    print("  How strongly does each feature correlate with budget-at-limit?\n")

    import numpy as np

    costs = np.array([p["cost"] for p in points])

    features = {
        "total_tokens": np.array([p["total_tokens"] for p in points]),
        "cache_read": np.array([p["cache_read_tokens"] for p in points]),
        "output_tokens": np.array([p["output_tokens"] for p in points]),
        "cache_creation": np.array([p["cache_creation_tokens"] for p in points]),
        "input_tokens": np.array([p["input_tokens"] for p in points]),
        "turns": np.array([p["turns"] for p in points], dtype=float),
        "non_cache_tokens": np.array([
            p["input_tokens"] + p["output_tokens"] + p["cache_creation_tokens"]
            for p in points
        ]),
    }

    rows = []
    for name, values in sorted(features.items(),
                                key=lambda kv: -abs(np.corrcoef(kv[1], costs)[0, 1])):
        r = np.corrcoef(values, costs)[0, 1]
        r_sq = r ** 2
        rows.append([name, f"{r:.4f}", f"{r_sq:.4f}"])

    print(table(["Feature", "Pearson r", "r-squared"], rows, "lrr"))
    print()


def _feature_scatter_plots(points, output_dir, label="5-hour"):
    """Generate scatter plots of each feature vs budget."""
    print(subsection_header("Feature Scatter Plots"))

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        costs = [p["cost"] for p in points]

        features = [
            ("Total Tokens", [p["total_tokens"] / 1e6 for p in points], "M tokens"),
            ("Cache Read", [p["cache_read_tokens"] / 1e6 for p in points], "M tokens"),
            ("Cache Creation", [p["cache_creation_tokens"] / 1e6 for p in points], "M tokens"),
            ("Output Tokens", [p["output_tokens"] / 1e3 for p in points], "K tokens"),
            ("Non-Cache Tokens", [(p["input_tokens"] + p["output_tokens"] + p["cache_creation_tokens"]) / 1e3 for p in points], "K tokens"),
            ("Turn Count", [p["turns"] for p in points], "turns"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for idx, (name, values, unit) in enumerate(features):
            ax = axes[idx // 3][idx % 3]

            x = np.array(values)
            y = np.array(costs)

            # Clip outliers beyond 99th percentile for cleaner plots
            x_cap = np.percentile(x, 99)
            y_cap = np.percentile(y, 99)
            mask = (x <= x_cap) & (y <= y_cap)
            n_removed = len(x) - mask.sum()

            ax.scatter(x[mask], y[mask], alpha=0.5, s=30, c="#e74c3c",
                       edgecolors="gray", linewidth=0.5)
            if n_removed > 0:
                ax.scatter(x[~mask], y[~mask], alpha=0.3, s=20, c="gray",
                           linewidth=0.5, marker="x")

            # Trend line fitted on all data, drawn over inlier range
            if len(x) > 2 and np.std(x) > 0:
                z = np.polyfit(x, y, 1)
                p_line = np.poly1d(z)
                x_line = np.sort(x[mask]) if mask.any() else np.sort(x)
                ax.plot(x_line, p_line(x_line), "--", color="#3498db", alpha=0.7)
                r = np.corrcoef(x, y)[0, 1]
                suffix = f" ({n_removed} outliers excluded)" if n_removed else ""
                ax.set_title(f"{name}\nr={r:.3f}, r²={r**2:.3f}{suffix}", fontsize=11)
                # Tighten axis to inlier data range
                if mask.any():
                    x_pad = (x[mask].max() - x[mask].min()) * 0.05
                    y_pad = (y[mask].max() - y[mask].min()) * 0.05
                    ax.set_xlim(x[mask].min() - x_pad, x[mask].max() + x_pad)
                    ax.set_ylim(y[mask].min() - y_pad, y[mask].max() + y_pad)
            else:
                ax.set_title(name, fontsize=11)

            ax.set_xlabel(f"{name} ({unit})", fontsize=9)
            ax.set_ylabel("Budget at Limit ($)", fontsize=9)
            ax.grid(True, alpha=0.2)

        plt.suptitle(f"Feature vs Budget at Rate-Limit Hit ({label} limit)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = output_dir / "calibration_scatter.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Scatter plots saved: {path}\n")
    except Exception as e:
        print(f"  Could not generate scatter plots: {e}\n")


def _limit_window_key(event, group_of=None):
    """Identity of the limit WINDOW an event belongs to.

    Retries fire every few minutes for as long as the limit holds, and every one
    of them quotes the SAME reset time, so the quoted reset is the window's
    identity. A fixed time bucket is not: it splits one limit into a positive
    per retry, which duplicates near-identical rows into the fit. That matters
    far more for the weekly bucket, where a limit is retried for DAYS.

    The account is part of the key because every profile directory sharing an
    account logs the same limit independently.

    A BARE reset time ("resets 4pm") repeats daily, so the date disambiguates
    it. A DATED reset ("resets Aug 3 at 9pm") is already unique, and including
    the date would split one week-long limit into one window per day of retries.
    """
    group = event.get("group") or (group_of or {}).get(event["profile"], event["profile"])
    text = (event.get("text") or "").lower()
    m = re.search(r"resets\s+(.+?)\s*(?:\(|$)", text)
    if not m:
        # No reset quoted: fall back to a coarse bucket so retries still merge.
        return (group, "nots", int(event["timestamp"].timestamp()) // 3600)
    reset = m.group(1).strip()
    dated = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", reset)
    if dated:
        return (group, "dated", reset)
    return (group, event["timestamp"].strftime("%Y-%m-%d"), reset)


def _dedupe_by_reset_window(points):
    """Keep only the first calibration point per unique limit window.

    See _limit_window_key for what "unique" means. Both this and the regression's
    hit-window construction use that one helper deliberately: they drifted apart
    once already (this keyed on the profile directory, so one account-level
    limit produced an identical row per agent), and a shared key is the only
    thing that stops it happening again.
    """
    seen = set()
    deduped = []
    for p in sorted(points, key=lambda x: x["window_end"]):
        key = _limit_window_key(p["event"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def _budget_timeline(points, sorted_turns, config, output_dir, group_of=None,
                     duration=timedelta(hours=5), stride=timedelta(hours=5),
                     label="5-hour", all_limit_events=None):
    """Plot cost-at-limit over time, with non-hit windows as baseline."""
    group_of = group_of or {}
    print(subsection_header("Budget Timeline"))
    print("  How has the budget at each limit hit changed over time?\n")

    if len(points) < 3:
        print("  Not enough limit windows for timeline.\n")
        return

    print(f"  {len(points)} unique limit windows\n")

    # Print table
    rows = []
    for p in points:
        rows.append([
            p["window_end"].strftime("%Y-%m-%d %H:%M"),
            p["event"]["profile"],
            format_cost(p["cost"]),
            format_number(p["turns"]),
            format_tokens(p["output_tokens"]),
        ])
    print(table(["Limit Hit", "Profile", "Budget", "Turns", "Output"], rows, "llrrr"))

    # Generate chart
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))

        # Baseline of non-hit windows, grouped by ACCOUNT and sliced by bisect,
        # matching how the regression builds its controls. Doing it per profile
        # directory here would draw a different baseline from the one the model
        # is fitted on, for the same chart.
        by_group = {}
        for t in sorted_turns:
            if t.model == "<synthetic>":
                continue
            by_group.setdefault(group_of.get(t.profile_name, t.profile_name), []).append(t)
        group_times = {g: [t.timestamp for t in ts_] for g, ts_ in by_group.items()}

        non_hit_times = []
        non_hit_costs = []
        for g, group_turns in by_group.items():
            # Screen against EVERY limit event, matching how the regression
            # builds its controls. Screening only against the plotted type
            # leaves windows where extra usage was bought and exhausted sitting
            # in the baseline as ordinary non-hit windows, at costs far above
            # anything the plotted limit ever allowed.
            limit_times = sorted(e["timestamp"] for e in (all_limit_events or [])
                                 if (e.get("group") or e["profile"]) == g)
            if not limit_times:
                limit_times = sorted(p["window_end"] for p in points
                                     if (p["event"].get("group")
                                         or p["event"]["profile"]) == g)
            times = group_times[g]
            current = group_turns[0].timestamp + duration
            last_ts = group_turns[-1].timestamp
            while current < last_ts:
                start = current - duration
                i = bisect_left(limit_times, start)
                if not (i < len(limit_times) and limit_times[i] <= current):
                    wt = group_turns[bisect_left(times, start):bisect_right(times, current)]
                    if wt:
                        non_hit_times.append(current)
                        non_hit_costs.append(sum(
                            estimate_cost(config, t.model, t.input_tokens, t.output_tokens,
                                          t.cache_creation_input_tokens,
                                          t.cache_read_input_tokens)
                            for t in wt
                        ))
                current += stride

        # Plot non-hit windows as light background scatter
        if non_hit_times:
            ax.scatter(non_hit_times, non_hit_costs, color="#bdc3c7", s=12, alpha=0.3,
                       label=f"Non-hit {label} windows ({len(non_hit_times)})", zorder=1)

        # Plot limit-hit points
        hit_times = [p["window_end"] for p in points]
        hit_costs = [p["cost"] for p in points]
        ax.scatter(hit_times, hit_costs, color="#c0392b", s=60, alpha=0.9,
                   edgecolors="#922b21", linewidth=0.5, label=f"Limit hits ({len(points)})",
                   zorder=3)

        # Trim x-axis to data range
        min_date = min(p["window_end"] for p in points)
        max_date = max(p["window_end"] for p in points)
        pad = (max_date - min_date) * 0.03
        ax.set_xlim(min_date - pad, max_date + pad)

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cost at Limit Hit ($)", fontsize=12)
        # Name the bucket in the title: both windows produce this chart, and
        # side by side they are otherwise indistinguishable.
        ax.set_title(f"Budget at Each Rate-Limit Hit Over Time ({label} limit)",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        path = output_dir / "budget_timeline.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"\n  Chart saved: {path}\n")
    except Exception as e:
        print(f"  Could not generate chart: {e}\n")


def _logistic_regression_analysis(limit_events, sorted_turns, config, output_dir,
                                  group_of=None, limit_types=("5-hour",),
                                  duration=timedelta(hours=5),
                                  stride=timedelta(hours=5), label="5-hour"):
    """Binary classification: predict whether a window hits the limit."""
    print(subsection_header("Logistic Regression Analysis"))
    print("  Can we predict limit hits from token usage patterns?\n")

    from collections import Counter

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve

    import numpy as np

    group_of = group_of or {}
    conv = [e for e in limit_events if e["source"] == "conversation"]

    # The features describe THIS window, so only a limit on this bucket is
    # explained by them. Keep the buckets separate and say what was set aside.
    hard_limits = [e for e in conv if e["limit_type"] in limit_types]
    excluded = Counter(e["limit_type"] for e in conv if e["limit_type"] not in limit_types)
    print(f"  Fitting on limit types: {', '.join(limit_types)}  "
          f"({len(hard_limits)} of {len(conv)} events)")
    if excluded:
        print(f"  Set aside (not explained by a {label} window): "
              + ", ".join(f"{n} {t}" for t, n in excluded.most_common()))
    print()

    # Index turns by group ONCE, and keep a parallel timestamp list so a window
    # is a bisect rather than a scan. Otherwise every limit event rescans its
    # whole group, which is where this report spends most of its time.
    by_group = {}
    for t in sorted_turns:
        if t.model == "<synthetic>":
            continue
        by_group.setdefault(group_of.get(t.profile_name, t.profile_name), []).append(t)
    group_times = {g: [t.timestamp for t in ts_] for g, ts_ in by_group.items()}

    def window_slice(g, start, end):
        turns_in_group = by_group.get(g)
        if not turns_in_group:
            return []
        times = group_times[g]
        return turns_in_group[bisect_left(times, start):bisect_right(times, end)]

    windows = []

    # Collapse retries and per-directory copies into one event per limit window.
    # Without this the same limit enters the fit once per retry and once per
    # profile dir, duplicating near-identical rows and inflating the hit class.
    seen = set()
    deduped_hits = []
    for e in sorted(hard_limits, key=lambda x: x["timestamp"]):
        key = _limit_window_key(e, group_of)
        if key in seen:
            continue
        seen.add(key)
        deduped_hits.append(e)
    print(f"  Hit windows: {len(hard_limits)} events -> {len(deduped_hits)} "
          f"distinct limit windows (retries and per-directory copies collapsed)")

    # Build hit windows per group (limits are per-account)
    for e in deduped_hits:
        ts = e["timestamp"]
        g = e.get("group", e["profile"])
        wt = window_slice(g, ts - duration, ts)
        if wt:
            windows.append(_build_window(wt, config, hit=1))

    # Build non-hit windows per group
    groups_seen = set(e.get("group", e["profile"]) for e in deduped_hits)
    for g in groups_seen:
        group_turns = by_group.get(g, [])
        if not group_turns:
            continue
        # Controls are screened against EVERY limit event, not just the fitted
        # type. A window in which extra usage was bought and then exhausted is
        # not a clean negative: it exceeded the ordinary allowance and simply
        # did not report THIS limit, so counting it as "no hit" teaches the fit
        # that such weeks are fine. Sorted, so the containment test is a bisect.
        group_limit_times = sorted(e["timestamp"] for e in conv
                                   if e.get("group", e["profile"]) == g)
        first_ts = group_turns[0].timestamp
        last_ts = group_turns[-1].timestamp
        current = first_ts + duration
        while current < last_ts:
            # A control must not CONTAIN a limit event. A window that does is
            # not a negative: the usage inside it is exactly what tripped the
            # limit, so labelling it "no hit" teaches the fit the opposite of
            # the truth. Scaling with the window matters -- a fixed +/-1h zone
            # rejects almost nothing from a 7-day window.
            start = current - duration
            i = bisect_left(group_limit_times, start)
            contains_limit = (i < len(group_limit_times)
                              and group_limit_times[i] <= current)
            if not contains_limit:
                wt = window_slice(g, start, current)
                if wt:
                    windows.append(_build_window(wt, config, hit=0))
            current += stride

    hit_count = sum(1 for w in windows if w["hit"])
    nohit_count = len(windows) - hit_count
    print(f"  Windows: {len(windows)} total ({hit_count} hit limit, {nohit_count} no hit)\n")

    if hit_count < 5 or nohit_count < 5:
        print("  Not enough windows for logistic regression.\n")
        return

    y = np.array([w["hit"] for w in windows])

    feature_sets = {
        "All 4 token types": ["input", "output", "cache_create", "cache_read"],
        "Cost-weighted": ["cost"],
        "Output tokens only": ["output"],
        "Cache read only": ["cache_read"],
        "Cache create only": ["cache_create"],
        "Input tokens only": ["input"],
        "Output + cache_create": ["output", "cache_create"],
        "Cost + cache_create": ["cost", "cache_create"],
    }

    results = {}
    for name, features in feature_sets.items():
        X = np.array([[w[f] for f in features] for w in windows])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr.fit(Xs, y)
        probs = lr.predict_proba(Xs)[:, 1]
        auc = roc_auc_score(y, probs)
        results[name] = {
            "auc": auc, "features": features, "model": lr,
            "scaler": scaler, "probs": probs, "coefs": lr.coef_[0],
        }

    rows = []
    for name in sorted(results, key=lambda n: -results[n]["auc"]):
        r = results[name]
        coef_str = ", ".join(f"{f}={c:+.3f}" for f, c in zip(r["features"], r["coefs"]))
        # Do NOT truncate: the table auto-sizes, and a 60-char cap cut the
        # cache_read coefficient off exactly the rows that fit all four
        # features, which are the rows the coefficient comparison is about.
        rows.append([name, f"{r['auc']:.4f}", coef_str])
    print(table(["Model", "AUC", "Coefficients (standardized)"], rows, "lrl"))

    best_name = max(results, key=lambda n: results[n]["auc"])
    best = results[best_name]
    cost_auc = results["Cost-weighted"]["auc"]
    cost_cc_auc = results["Cost + cache_create"]["auc"]

    print(f"  Best model: {best_name} (AUC={best['auc']:.4f})\n")

    cc_delta = cost_cc_auc - cost_auc
    if cc_delta > 0.02:
        print(f"  Cache create adds +{cc_delta:.4f} AUC beyond cost alone.")
        print(f"  The limit formula may weight cache creation differently from API pricing.\n")

    # Generate chart
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        top_models = ["All 4 token types", "Cost-weighted", "Cost + cache_create",
                      "Cache create only", "Output tokens only"]
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        for name, color in zip(top_models, colors):
            if name not in results:
                continue
            r = results[name]
            fpr, tpr, _ = roc_curve(y, r["probs"])
            ax1.plot(fpr, tpr, color=color, linewidth=2,
                     label=f"{name} (AUC={r['auc']:.3f})")
        ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
        ax1.set_xlabel("False Positive Rate", fontsize=11)
        ax1.set_ylabel("True Positive Rate", fontsize=11)
        ax1.set_title("ROC Curves: Predicting Limit Hits", fontsize=13)
        ax1.legend(fontsize=9, loc="lower right")
        ax1.grid(True, alpha=0.2)

        all4 = results["All 4 token types"]
        feature_names = all4["features"]
        coefs = all4["coefs"]
        abs_coefs = np.abs(coefs)
        sorted_idx = np.argsort(abs_coefs)
        ax2.barh(
            [feature_names[i] for i in sorted_idx],
            [abs_coefs[i] for i in sorted_idx],
            color="#e74c3c", alpha=0.8,
        )
        ax2.set_xlabel("|Coefficient| (standardized)", fontsize=11)
        ax2.set_title("Feature Importance (All 4 Token Types)", fontsize=13)
        ax2.grid(True, alpha=0.2, axis="x")

        # .capitalize() leaves "5-hour" alone (leading digit) and gives "Weekly".
        plt.suptitle(f"Logistic Regression: What Predicts {label.capitalize()} Limit Hits?",
                      fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = output_dir / "logistic_regression.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Chart saved: {path}\n")
    except Exception as e:
        print(f"  Could not generate chart: {e}\n")


def _build_window(turns, config, hit):
    """Build a feature dict for a 5h window of turns."""
    input_tok = sum(t.input_tokens for t in turns)
    output_tok = sum(t.output_tokens for t in turns)
    cc_tok = sum(t.cache_creation_input_tokens for t in turns)
    cr_tok = sum(t.cache_read_input_tokens for t in turns)
    cost = sum(
        estimate_cost(config, t.model, t.input_tokens, t.output_tokens,
                      t.cache_creation_input_tokens, t.cache_read_input_tokens)
        for t in turns
    )
    return {
        "hit": hit,
        "input": input_tok,
        "output": output_tok,
        "cache_create": cc_tok,
        "cache_read": cr_tok,
        "cost": cost,
        "turns": len(turns),
    }
