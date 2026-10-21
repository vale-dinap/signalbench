"""Forecast tasks, threshold grids and distributions derived from P(value > threshold).

Every predictor (decision model, time-series foundation model, baseline) receives
the same ForecastTask and returns exceedance probabilities on the task's threshold
grid. Scores and quantiles are then computed identically for all of them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Series

DEFAULT_SIGMAS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
PROB_EPS = 1e-3          # keep probability-series thresholds strictly inside (0, 1)
MIN_SCALE_PRICE = 1e-3   # floor on per-horizon log-return std (0.1 %)
MIN_SCALE_PROB = 5e-3    # floor on per-horizon absolute-change std (0.5 pp)


# ── Scale and grid ────────────────────────────────────────────────────────────

def horizon_scale(series: Series, idx: int, horizon: int, window: int = 30) -> float:
    """Std of the value change over `horizon` steps, estimated from data up to idx.

    price       : std of 1-step log returns × sqrt(horizon)
    probability : std of 1-step absolute changes × sqrt(horizon)
    Floored so flat stretches still produce a usable grid.
    """
    x = series.df["close"].to_numpy()[max(0, idx - window):idx + 1]
    if series.kind == "price":
        d = np.diff(np.log(x))
        floor = MIN_SCALE_PRICE
    else:
        d = np.diff(x)
        floor = MIN_SCALE_PROB
    s = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return max(s, floor) * math.sqrt(horizon)


def threshold_grid(series: Series, idx: int, horizon: int, sigmas=DEFAULT_SIGMAS,
                   window: int = 30) -> tuple[np.ndarray, np.ndarray, float]:
    """Thresholds at current value shifted by sigma multiples of the horizon scale.

    Returns (thresholds, sigmas_kept, scale). Thresholds are rounded to display
    precision and deduplicated, so a sigma level can be dropped on coarse series.
    The 0-sigma threshold is always exactly the current value, so P(up) is asked
    directly.
    """
    current = float(series.df.iloc[idx]["close"])
    scale = horizon_scale(series, idx, horizon, window)
    sig = np.asarray(sigmas, dtype=float)
    if series.kind == "price":
        th = current * np.exp(sig * scale)
    else:
        th = np.clip(current + sig * scale, PROB_EPS, 1 - PROB_EPS)
    th = series.round(th)
    th[sig == 0] = current
    # rounding can collapse neighbours onto the current value: the 0-sigma one wins
    drop = (sig != 0) & (th == current)
    th, sig = th[~drop], sig[~drop]
    th, keep = np.unique(th, return_index=True)
    return th, sig[keep], scale


# ── Task ──────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class ForecastTask:
    """One forecasting question: where will `series` be `horizon` steps after `idx`?

    Built once per decision point and handed unchanged to every predictor, so all
    models answer exactly the same question on the same grid. Only data at or
    before idx is exposed (history, related_history); the realised value is kept
    out of the task and handled by the backtest.
    """
    series: Series
    idx: int
    horizon: int
    thresholds: np.ndarray
    sigmas: np.ndarray
    scale: float
    related: tuple[Series, ...] = ()

    @property
    def t(self) -> pd.Timestamp:
        return self.series.df["t"].iloc[self.idx]

    @property
    def current(self) -> float:
        return float(self.series.df["close"].iloc[self.idx])

    @property
    def history(self) -> np.ndarray:
        """Values up to and including the decision point."""
        return self.series.df["close"].to_numpy()[:self.idx + 1]

    @property
    def i0(self) -> int:
        """Position of the current-value (0-sigma) threshold."""
        return int(np.flatnonzero(self.sigmas == 0)[0])

    def related_history(self, other: Series) -> np.ndarray:
        """Another series' values at or before the decision time (as-of, no look-ahead)."""
        pos = int(other.df["t"].searchsorted(self.t, side="right"))
        return other.df["close"].to_numpy()[:pos]


def make_task(series: Series, idx: int, horizon: int, sigmas=DEFAULT_SIGMAS, vol_window: int = 30,
              related: tuple[Series, ...] = ()) -> ForecastTask:
    th, sig, scale = threshold_grid(series, idx, horizon, sigmas, vol_window)
    return ForecastTask(series, idx, horizon, th, sig, scale, tuple(related))


BLIND_DESCRIPTION = "Asset: daily close price, rebased to 100 at the first day shown"


def blind_task(task: ForecastTask, window: int = 60) -> ForecastTask:
    """The same question with the cues a model could recall removed.

    Keeps the last `window` observations up to the decision point, rescales them so
    the first one shown is 100 and drops the name and unit; meta["blind"] tells the
    state builder to show positions instead of dates. Thresholds are rescaled by the
    same factor, so P(value > threshold) is unchanged and answers map back onto the
    original grid. Price series only, without related series.
    """
    if task.series.kind != "price" or task.related:
        raise ValueError("blind_task supports price series without related series")
    w = task.series.df.iloc[max(0, task.idx - window + 1):task.idx + 1].reset_index(drop=True)
    k = 100.0 / float(w["close"].iloc[0])
    s = Series(key=task.series.key, description=BLIND_DESCRIPTION, df=w.assign(close=w["close"] * k),
               value_label=task.series.value_label, decimals=2, meta={"blind": True})
    return ForecastTask(s, len(w) - 1, task.horizon, task.thresholds * k, task.sigmas, task.scale)


# ── Monotone CDF from raw answers ─────────────────────────────────────────────

def isotonic_decreasing(y, w=None) -> np.ndarray:
    """Pool-adjacent-violators fit of a non-increasing sequence."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            vals[-2:] = [v]; wts[-2:] = [wts[-2] + wts[-1]]; cnt[-2:] = [cnt[-2] + cnt[-1]]
    return np.repeat(vals, cnt)


def coherence_violation(p_above) -> float:
    """Total upward movement in P(>x) as x increases (0 = perfectly coherent)."""
    d = np.diff(np.asarray(p_above, dtype=float))
    return float(d[d > 0].sum())


def monotone_cdf(p_above) -> np.ndarray:
    """F(threshold) = 1 - isotonic P(> threshold), clipped to [0, 1]."""
    return np.clip(1.0 - isotonic_decreasing(p_above), 0.0, 1.0)


def _support(thresholds: np.ndarray, kind: str) -> tuple[float, float]:
    """Where the outermost tail mass is placed: one extra grid step beyond each end."""
    th = np.asarray(thresholds, dtype=float)
    step = np.median(np.diff(th)) if len(th) > 1 else max(abs(th[0]) * 0.01, 1e-3)
    lo, hi = th[0] - step, th[-1] + step
    if kind == "probability":
        lo, hi = max(lo, 0.0), min(hi, 1.0)
    elif kind == "price":
        lo = max(lo, 0.0)
    return lo, hi  # kind == "logprice": unbounded


def cdf_knots(thresholds, p_above, kind: str = "price") -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear CDF knots including tails: F(lo) = 0, F(hi) = 1."""
    th = np.asarray(thresholds, dtype=float)
    F = monotone_cdf(p_above)
    lo, hi = _support(th, kind)
    return np.concatenate([[lo], th, [hi]]), np.concatenate([[0.0], F, [1.0]])


def quantiles(thresholds, p_above, levels=(0.1, 0.25, 0.5, 0.75, 0.9), kind: str = "price") -> np.ndarray:
    """Invert the piecewise-linear CDF."""
    x, F = cdf_knots(thresholds, p_above, kind)
    # np.interp needs strictly increasing xp; flat CDF segments are handled by nudging
    F = F + np.arange(len(F)) * 1e-12
    return np.interp(np.asarray(levels, dtype=float), F, x)


def p_above_at(thresholds, p_above, x: float, kind: str = "price") -> float:
    """P(value > x) by linear interpolation of the monotone CDF."""
    xs, F = cdf_knots(thresholds, p_above, kind)
    return float(1.0 - np.interp(x, xs, F))


def crps(thresholds, p_above, outcome: float, kind: str = "price", n: int = 400) -> float:
    """CRPS of the piecewise-linear CDF against a realised value (in value units).

    Integration range covers both the CDF support and the outcome.
    """
    xs, F = cdf_knots(thresholds, p_above, kind)
    lo, hi = min(xs[0], outcome), max(xs[-1], outcome)
    grid = np.linspace(lo, hi, n)
    Fg = np.interp(grid, xs, F)
    H = (grid >= outcome).astype(float)
    return float(np.trapezoid((Fg - H) ** 2, grid))


# ── Quantile forecasts -> exceedance probabilities ────────────────────────────

def p_above_from_quantiles(levels, values, thresholds) -> np.ndarray:
    """Map a quantile forecast onto thresholds: P(value > x) = 1 - F(x).

    F is piecewise linear through (quantile value, level); crossing quantiles are
    sorted; tails are extended linearly to levels 0 and 1 using the outermost
    spacing, mirroring how threshold answers are turned into a CDF.
    """
    lv = np.asarray(levels, dtype=float)
    q = np.sort(np.asarray(values, dtype=float))
    if len(q) < 2:
        raise ValueError("need at least 2 quantiles")
    lo = q[0] - (q[1] - q[0]) * lv[0] / (lv[1] - lv[0])
    hi = q[-1] + (q[-1] - q[-2]) * (1 - lv[-1]) / (lv[-1] - lv[-2])
    xs = np.concatenate([[lo], q, [hi]])
    F = np.concatenate([[0.0], lv, [1.0]])
    xs = xs + np.arange(len(xs)) * 1e-12 * max(1.0, abs(xs).max())  # strictly increasing for interp
    return 1.0 - np.interp(np.asarray(thresholds, dtype=float), xs, F)


# ── Parametric helpers for baselines ──────────────────────────────────────────

def normal_sf(z) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))
