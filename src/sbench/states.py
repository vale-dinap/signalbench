"""Natural-language wording: state builders and horizon phrases.

State builders take a ForecastTask and return the text a decision model sees.
Only data at or before the decision time is used. Variants (lookback, related
series detail) are configured with functools.partial, e.g.
    partial(build_price_state, lookback=60, related_lookback=-1)
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .data import Series, known_rows

if TYPE_CHECKING:
    from .context import NewsFeed, SearchInterest
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
        pos = int(known_rows(other, task.known_at))
        if pos == 0:
            continue
        lines += ["", f"Related: {other.description}"]
        if related_lookback == 0:
            lines.append(f"  Latest {other.value_label}: {other.fmt(other.df['close'].iloc[pos - 1])}")
        else:
            lines.extend(history_lines(other, pos, related_lookback))
    return "\n".join(lines)


def indicator_lines(series: Series, end_pos: int) -> list[str]:
    """Common technical indicators at row end_pos - 1, computed from rows before end_pos only.

    Moving averages (20, 50, 200), Bollinger bands (20, 2 sd) with %B, Wilder's
    14-period RSI and the change over the last 20 observations; "n/a" where the
    history is too short.
    """
    c = series.df["close"].iloc[:end_pos].reset_index(drop=True)
    last = float(c.iloc[-1])

    def ma(n: int) -> float:
        return float(c.iloc[-n:].mean()) if len(c) >= n else np.nan

    def vs(level: float) -> str:
        return "n/a" if np.isnan(level) else f"{series.fmt(level)} (price {100 * (last / level - 1):+.1f}%)"

    ma20 = ma(20)
    sd20 = float(c.iloc[-20:].std()) if len(c) >= 20 else np.nan
    lo, hi = ma20 - 2 * sd20, ma20 + 2 * sd20
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    rsi = np.nan if len(c) < 15 else 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    change = 100 * (last / float(c.iloc[-21]) - 1) if len(c) >= 21 else np.nan
    na = "n/a"
    return [
        "Indicators at the current close:",
        f"  20-day moving average: {vs(ma20)}",
        f"  50-day moving average: {vs(ma(50))}",
        f"  200-day moving average: {vs(ma(200))}",
        "  Bollinger bands (20, 2 sd): " + (na if np.isnan(sd20) else
                                            f"{series.fmt(lo)} to {series.fmt(hi)}, "
                                            f"%B {(last - lo) / (hi - lo) if hi > lo else 0.5:.2f}"),
        f"  14-period RSI: {na if np.isnan(rsi) else f'{rsi:.0f}'}",
        f"  Change over the last 20 observations: {na if np.isnan(change) else f'{change:+.1f}%'}",
    ]


def build_indicator_state(task: ForecastTask, lookback: int = -1, related_lookback: int = 0) -> str:
    """build_price_state plus technical indicators at the decision time (indicator_lines)."""
    return "\n".join([build_price_state(task, lookback, related_lookback), "",
                      *indicator_lines(task.series, task.idx + 1)])


def build_news_state(task: ForecastTask, feed: NewsFeed, lookback: int = -1) -> str:
    """Price state plus news coverage and headlines known at the decision time (context.NewsFeed).

    Related series are left out: in the news variant they carry TimesFM's news covariates.
    """
    return "\n".join([build_price_state(replace(task, related=()), lookback), "",
                      *feed.lines(task.series.key, task.known_at)])


def build_search_state(task: ForecastTask, interest: SearchInterest, lookback: int = -1) -> str:
    """Price state plus Google search interest known at the decision time (context.SearchInterest).

    Related series are left out: in the search variant they carry TimesFM's search covariates.
    """
    return "\n".join([build_price_state(replace(task, related=()), lookback), "",
                      *interest.lines(task.series.key, task.known_at)])
