"""Calibration Diagnostics — investigate the unknown limit model.

Data science approach to reverse-engineering Anthropic's limit formula:
1. Correlation analysis of each feature vs budget-at-limit
2. Feature scatter plots with trend lines
3. Budget timeline (cost at each limit hit over time)
4. Logistic regression (binary classification: hit limit or not)
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from ..config import estimate_cost
from ..formatting import (
    format_cost, format_number, format_tokens,
    section_header, subsection_header, table,
)


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
                        if error == "rate_limit":
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

    # Classify limit type from message text
    import re
    for e in events:
        text = e.get("text", "")
        text_lower = text.lower()
        if "5-hour" in text_lower or "5 hour" in text_lower:
            e["limit_type"] = "5-hour"
        elif "extra usage" in text_lower:
            e["limit_type"] = "extra-usage"
        elif "hit your limit" in text_lower:
            # Distinguish 5h from weekly by reset time
            # "resets Feb 19 at 10am" = weekly (has date); "resets 6pm" = 5-hour
            if re.search(r"resets\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", text_lower):
                e["limit_type"] = "weekly"
            else:
                e["limit_type"] = "5-hour"
        elif "rate limit reached" in text_lower:
            e["limit_type"] = "api-ratelimit"
        elif e.get("hours_till_reset") and e["hours_till_reset"] > 24:
            e["limit_type"] = "weekly"
        elif e.get("hours_till_reset") and e["hours_till_reset"] <= 5:
            e["limit_type"] = "5-hour"
        else:
            e["limit_type"] = "unknown"

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


def _calculate_window_costs(limit_events, sorted_turns, config):
    """For each rate-limit event, calculate total cost in the preceding 5h window."""
    points = []

    # Only use conversation-source rate_limit events (not "allowed" telemetry)
    actual_limits = [e for e in limit_events
                     if e["source"] == "conversation"
                     or e.get("status") in ("rate_limited", "allowed_warning")]

    for event in actual_limits:
        ts = event["timestamp"]
        window_start = ts - timedelta(hours=5)
        profile = event["profile"]

        # Find all turns in this 5h window for the same profile
        # (limits are per-account, and concurrent chats all count)
        window_turns = [t for t in sorted_turns
                        if window_start <= t.timestamp <= ts
                        and t.model != "<synthetic>"
                        and t.profile_name == profile]

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

    # Get rate limit events and calibration points
    # Get rate limit events and calibration points
    limit_events = _find_rate_limit_events(profiles)
    sorted_turns = sorted(turns, key=lambda t: t.timestamp)
    points = _calculate_window_costs(limit_events, sorted_turns, config)

    # Filter to 5-hour hard limits, then dedupe retries (same reset window)
    five_hour_raw = [p for p in points
                     if p["is_hard_limit"] and p["event"].get("limit_type") == "5-hour"]
    five_hour = _dedupe_by_reset_window(five_hour_raw)

    print(f"  Total calibration points: {len(points)}")
    print(f"  5-hour limit events: {len(five_hour_raw)} raw, {len(five_hour)} unique windows")
    print()

    if len(five_hour) < 5:
        print("  Not enough 5-hour limit events for diagnostics.\n")
        return

    _correlation_analysis(five_hour)
    _feature_scatter_plots(five_hour, output_dir)
    _budget_timeline(five_hour, sorted_turns, config, output_dir)
    _logistic_regression_analysis(limit_events, sorted_turns, config, output_dir)


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


def _feature_scatter_plots(points, output_dir):
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

        plt.suptitle("Feature vs Budget at Rate-Limit Hit", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = output_dir / "calibration_scatter.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Scatter plots saved: {path}\n")
    except Exception as e:
        print(f"  Could not generate scatter plots: {e}\n")


def _dedupe_by_reset_window(points):
    """Keep only the first calibration point per unique reset window.

    Groups by profile + date + reset time from message text (e.g. "resets 5pm").
    This filters out retry events that fire every 10 minutes against the same limit.
    """
    seen = set()
    deduped = []
    for p in sorted(points, key=lambda x: x["window_end"]):
        text = p["event"].get("text", "")
        match = re.search(r"resets\s+(\S+)", text.lower())
        reset = match.group(1) if match else "x"
        key = f"{p['event']['profile']}_{p['window_end'].strftime('%Y-%m-%d')}_{reset}"
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def _budget_timeline(points, sorted_turns, config, output_dir):
    """Plot cost-at-limit over time, with non-hit windows as baseline."""
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

        profiles_in_data = sorted(set(p["event"]["profile"] for p in deduped))

        # Sample non-hit 5h windows as baseline (per-profile internally)
        non_hit_times = []
        non_hit_costs = []
        for profile in profiles_in_data:
            profile_turns = [t for t in sorted_turns
                             if t.profile_name == profile and t.model != "<synthetic>"]
            if not profile_turns:
                continue
            profile_limit_times = {p["window_end"] for p in points
                                   if p["event"]["profile"] == profile}
            first_ts = profile_turns[0].timestamp
            last_ts = profile_turns[-1].timestamp
            current = first_ts + timedelta(hours=5)
            while current < last_ts:
                near_limit = any(abs((current - lt).total_seconds()) < 3600
                                 for lt in profile_limit_times)
                if not near_limit:
                    window_start = current - timedelta(hours=5)
                    wt = [t for t in profile_turns
                          if window_start <= t.timestamp <= current]
                    if wt:
                        cost = sum(
                            estimate_cost(config, t.model, t.input_tokens, t.output_tokens,
                                          t.cache_creation_input_tokens, t.cache_read_input_tokens)
                            for t in wt
                        )
                        non_hit_times.append(current)
                        non_hit_costs.append(cost)
                current += timedelta(hours=5)

        # Plot non-hit windows as light background scatter
        if non_hit_times:
            ax.scatter(non_hit_times, non_hit_costs, color="#bdc3c7", s=12, alpha=0.3,
                       label=f"Non-hit 5h windows ({len(non_hit_times)})", zorder=1)

        # Plot limit-hit points
        hit_times = [p["window_end"] for p in points]
        hit_costs = [p["cost"] for p in points]
        ax.scatter(hit_times, hit_costs, color="#c0392b", s=60, alpha=0.9,
                   edgecolors="#922b21", linewidth=0.5, label=f"Limit hits ({len(points)})",
                   zorder=3)

        # Trim x-axis to data range
        min_date = min(p["window_end"] for p in deduped)
        max_date = max(p["window_end"] for p in deduped)
        pad = (max_date - min_date) * 0.03
        ax.set_xlim(min_date - pad, max_date + pad)

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cost at Limit Hit ($)", fontsize=12)
        ax.set_title("Budget at Each Rate-Limit Hit Over Time", fontsize=14, fontweight="bold")
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


def _logistic_regression_analysis(limit_events, sorted_turns, config, output_dir):
    """Binary classification: predict whether a 5h window hits the limit."""
    print(subsection_header("Logistic Regression Analysis"))
    print("  Can we predict limit hits from token usage patterns?\n")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve

    import numpy as np

    hard_limits = [e for e in limit_events if e["source"] == "conversation"]
    limit_times = {e["timestamp"] for e in hard_limits}

    windows = []

    # Build hit windows per-profile (limits are per-account)
    for e in hard_limits:
        ts = e["timestamp"]
        profile = e["profile"]
        window_start = ts - timedelta(hours=5)
        wt = [t for t in sorted_turns
              if window_start <= t.timestamp <= ts and t.model != "<synthetic>"
              and t.profile_name == profile]
        if wt:
            windows.append(_build_window(wt, config, hit=1))

    # Build non-hit windows per-profile
    profiles_seen = set(e["profile"] for e in hard_limits)
    for profile in profiles_seen:
        profile_turns = [t for t in sorted_turns
                         if t.profile_name == profile and t.model != "<synthetic>"]
        if not profile_turns:
            continue
        profile_limit_times = {e["timestamp"] for e in hard_limits
                               if e["profile"] == profile}
        first_ts = profile_turns[0].timestamp
        last_ts = profile_turns[-1].timestamp
        current = first_ts + timedelta(hours=5)
        while current < last_ts:
            near_limit = any(abs((current - lt).total_seconds()) < 3600
                            for lt in profile_limit_times)
            if not near_limit:
                window_start = current - timedelta(hours=5)
                wt = [t for t in profile_turns
                      if window_start <= t.timestamp <= current]
                if wt:
                    windows.append(_build_window(wt, config, hit=0))
            current += timedelta(hours=5)

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
        rows.append([name, f"{r['auc']:.4f}", coef_str[:60]])
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

        plt.suptitle("Logistic Regression: What Predicts Limit Hits?",
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
