"""Judging pre-registered hypotheses (docs/forward_test_plan.md).

A hypothesis states the side of zero its effect should fall on. With a 95%
interval for the effect:

    holds         the interval lies entirely on the stated side
    fails         the point estimate is on the other side
    inconclusive  anything else

Skill comparisons use scoring.skill_table. The two statistics that are not
skills, the change in the spread of P(up) and the slope of the up-rate on
Bollinger %B, use paired_bootstrap below, with the same resampling (decision
points as pairs, 2,000 resamples, seed 0).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

SIDES = ("above", "below")


def verdict(estimate: float, lo: float, hi: float, side: str) -> str:
    """'holds', 'fails' or 'inconclusive' for an effect expected `side` ('above' or 'below') zero."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    sign = 1 if side == "above" else -1
    if min(sign * lo, sign * hi) > 0:
        return "holds"
    if sign * estimate < 0:
        return "fails"
    return "inconclusive"


def paired_bootstrap(fn: Callable[..., float], *arrays, n: int = 2000, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    """fn(*arrays) with a bootstrap interval that resamples rows of all arrays together.

    Rows with a NaN in any array are dropped first. Returns estimate, lo, hi, N.
    """
    a = np.column_stack([np.asarray(x, dtype=float) for x in arrays])
    a = a[~np.isnan(a).any(axis=1)]
    if len(a) < 3:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "N": len(a)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    boot = np.array([fn(*a[i].T) for i in idx])
    lo, hi = np.nanquantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(fn(*a.T)), "lo": float(lo), "hi": float(hi), "N": len(a)}


def slope_per_sd(x, y) -> float:
    """Least-squares slope of y on x, per standard deviation of x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sd = x.std(ddof=1)
    return float(np.polyfit(x, y, 1)[0] * sd) if sd > 0 else np.nan


def sd_difference(a, b) -> float:
    """Standard deviation of a minus that of b (e.g. P(up) with and without news, on the same points)."""
    return float(np.std(a, ddof=1) - np.std(b, ddof=1))
