"""Text formatting utilities for terminal reports."""

from collections.abc import Sequence

SPARK_CHARS = "▁▂▃▄▅▆▇█"
BAR_CHARS = "▏▎▍▌▋▊▉█"


def format_number(n: int | float) -> str:
    """Format with commas: 1234567 -> '1,234,567'."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def format_tokens(n: int | float) -> str:
    """Human-readable token count: 1.2B, 345.6M, 12.3K."""
    n = int(n)
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_cost(usd: float) -> str:
    """Format as $1,234.56."""
    return f"${usd:,.2f}"


def format_pct(value: float, total: float) -> str:
    """Format as percentage."""
    if total == 0:
        return "0.0%"
    return f"{value / total * 100:.1f}%"


def format_duration(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = seconds / 3600
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def table(headers: list[str], rows: list[list[str]],
          alignments: str | None = None) -> str:
    """Render an ASCII table.

    alignments: string of 'l'/'r'/'c' per column (default: left).
    """
    if not rows:
        return "(no data)\n"

    ncols = len(headers)
    alignments = alignments or ("l" * ncols)

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < ncols:
                widths[i] = max(widths[i], len(cell))

    def fmt_cell(text: str, width: int, align: str) -> str:
        if align == "r":
            return text.rjust(width)
        if align == "c":
            return text.center(width)
        return text.ljust(width)

    lines = []
    # Header
    header = "  ".join(fmt_cell(h, widths[i], alignments[i])
                       for i, h in enumerate(headers))
    lines.append(header)
    lines.append("  ".join("─" * w for w in widths))

    # Rows
    for row in rows:
        line = "  ".join(
            fmt_cell(row[i] if i < len(row) else "", widths[i], alignments[i])
            for i in range(ncols)
        )
        lines.append(line)

    return "\n".join(lines) + "\n"


def bar_chart(labels: list[str], values: list[float],
              width: int = 40, show_values: bool = True) -> str:
    """Horizontal bar chart using block characters."""
    if not values:
        return "(no data)\n"

    max_val = max(values) if values else 1
    max_label = max(len(l) for l in labels) if labels else 0
    lines = []

    for label, val in zip(labels, values):
        bar_len = int(val / max_val * width) if max_val > 0 else 0
        full = bar_len // 8
        remainder = bar_len % 8
        bar = "█" * full
        if remainder > 0:
            bar += BAR_CHARS[remainder - 1]
        suffix = f" {format_tokens(val)}" if show_values else ""
        lines.append(f"  {label:>{max_label}}  {bar}{suffix}")

    return "\n".join(lines) + "\n"


def sparkline(values: Sequence[float]) -> str:
    """Compact sparkline using block characters."""
    if not values:
        return ""
    min_v = min(values)
    max_v = max(values)
    rng = max_v - min_v
    if rng == 0:
        return SPARK_CHARS[4] * len(values)
    return "".join(
        SPARK_CHARS[min(7, int((v - min_v) / rng * 7))]
        for v in values
    )


def section_header(title: str, width: int = 70) -> str:
    """Format a section header."""
    return f"\n{'=' * width}\n  {title}\n{'=' * width}\n"


def subsection_header(title: str) -> str:
    """Format a subsection header."""
    return f"\n--- {title} ---\n"
