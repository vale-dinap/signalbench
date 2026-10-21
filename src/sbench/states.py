"""Natural-language wording: state builders and horizon phrases.

State builders take a ForecastTask and return the text a decision model sees.
Only data at or before the decision time is used. Variants (lookback, related
series detail) are configured with functools.partial, e.g.
    partial(build_price_state, lookback=60, related_lookback=-1)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .data import Series

if TYPE_CHECKING:
    from .forecast import ForecastTask


def horizon_text(steps: int, freq) -> str:
    """'at the next daily observation', '3 daily observations from now', ..."""
    freq = pd.Timedelta(freq)
    if abs(freq - pd.Timedelta(days=1)) <= pd.Timedelta(hours=2):
        unit = "daily"
    elif abs(freq - pd.Timedelta(hours=1)) <= pd.Timedelta(minutes=5):
        unit = "hourly"
    elif abs(freq - pd.Timedelta(weeks=1)) <= pd.Timedelta(days=1):
        unit = "weekly"
    else:
        unit = f"{freq}-spaced"
    if steps == 1:
        return f"at the next {unit} observation"
    return f"{steps} {unit} observations from now"


def _time_fmt(series: Series) -> str:
    return "%Y-%m-%d" if series.freq >= pd.Timedelta(hours=20) else "%Y-%m-%d %H:%M"


def history_lines(series: Series, end_pos: int, lookback: int = -1) -> list[str]:
    """'  2026-10-01: 0.705' lines, oldest first, rows [.., end_pos) of series.df.

    lookback: -1 = all available rows, N = last N rows. Blinded series
    (meta["blind"], see forecast.blind_task) show 'day -59' ... 'day 0' instead of dates.
    """
    start = 0 if lookback == -1 else max(0, end_pos - lookback)
    window = series.df.iloc[start:end_pos]
    if series.meta.get("blind"):
        labels = [f"day {i - (end_pos - 1)}" for i in range(start, end_pos)]
    else:
        labels = [t.strftime(_time_fmt(series)) for t in window["t"]]
    return [f"  {lab}: {series.fmt(v)}" for lab, v in zip(labels, window["close"])]


def build_price_state(task: ForecastTask, lookback: int = -1, related_lookback: int = 0) -> str:
    """Description, decision time, current value, history; then each related series.

    lookback         : rows of the main series to show (-1 = full history)
    related_lookback : 0 = latest value only, -1 = full history, N = last N rows
    """
    s = task.series
    lines = [
        s.description,
        *([] if s.meta.get("blind") else [f"Decision time: {task.t.strftime(_time_fmt(s))}"]),
        f"Current {s.value_label}: {s.fmt(task.current)}",
        "History (oldest → newest):",
        *history_lines(s, task.idx + 1, lookback),
    ]
    for other in task.related:
        pos = int(other.df["t"].searchsorted(task.t, side="right"))
        if pos == 0:
            continue
        lines += ["", f"Related: {other.description}"]
        if related_lookback == 0:
            lines.append(f"  Latest {other.value_label}: {other.fmt(other.df['close'].iloc[pos - 1])}")
        else:
            lines.extend(history_lines(other, pos, related_lookback))
    return "\n".join(lines)
