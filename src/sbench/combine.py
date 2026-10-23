"""Combining stored forecasts: linear pools and agreement gates.

Combinations take long-format results (backtest.COLUMNS) and return rows in the
same format under a new predictor name, so they are scored by the same code as
any single predictor (scoring.per_forecast). Members must have answered the same
tasks: same series, decision times, horizon and threshold grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import COLUMNS
from .forecast import coherence_violation
from .scoring import per_forecast

POINT = ["series", "t", "horizon"]          # one decision point
ROW = [*POINT, "sigma"]                     # one threshold of one decision point
WEIGHTS = np.linspace(0, 1, 21)             # weights tried by fit_pool_weight


def _member(results: pd.DataFrame, name: str) -> pd.DataFrame:
    m = results[results["predictor"] == name]
    if m.empty:
        raise ValueError(f"no results for predictor {name!r}")
    return m.set_index(ROW).sort_index()


def linear_pool(results: pd.DataFrame, weights: dict[str, float], name: str) -> pd.DataFrame:
    """Weighted average of the members' P(above) at each threshold, on the decision points all answered.

    Weights are normalised to sum to 1. A threshold any member left unanswered
    (direction-only formats answer only the current value) stays unanswered, so
    pooling a full distribution with a direction-only answer gives a direction answer.
    """
    total = sum(weights.values())
    members = {k: _member(results, k) for k in weights}
    first = next(iter(members.values()))
    common = first.index
    for m in members.values():
        common = common.intersection(m.index)
    p = sum(members[k].loc[common, "p_above"].to_numpy(dtype=float) * w / total for k, w in weights.items())
    out = first.loc[common].reset_index().assign(predictor=name, p_above=p, input_tokens=np.nan, latency_s=np.nan)
    out["coherence"] = out.groupby(POINT)["p_above"].transform(lambda s: coherence_violation(s.dropna()))
    return out[COLUMNS]


def fit_pool_weight(results: pd.DataFrame, a: str, b: str, metric: str = "crps_n",
                    grid=WEIGHTS) -> tuple[float, pd.Series]:
    """Weight on `a` (1 - weight on `b`) minimising the mean `metric` over `results`.

    Pass only the fitting period's results. Returns the best weight and the mean
    metric at every weight of the grid.
    """
    scores = {}
    for w in grid:
        pooled = linear_pool(results, {a: w, b: 1 - w}, "pool")
        scores[float(w)] = per_forecast(pooled)[metric].mean()
    curve = pd.Series(scores, name=metric).rename_axis(f"weight on {a}")
    return float(curve.idxmin()), curve


def agreement_gate(results: pd.DataFrame, base: str, check: str, fallback: str, name: str) -> pd.DataFrame:
    """`base`'s forecast where `base` and `check` lean the same way on direction, else `fallback`'s.

    Leaning = P(up) above or below 0.5; a P(up) of exactly 0.5 leans neither way,
    so it never agrees. Scoring the gated forecast on every decision point, rather
    than scoring `base` only where the models agree, keeps the comparison fair.
    """
    def lean(pred: str) -> pd.Series:
        m = results[(results["predictor"] == pred) & (results["sigma"] == 0)]
        return np.sign(m.set_index(POINT)["p_above"] - 0.5)

    a, c = lean(base), lean(check)
    points = a.index.intersection(c.index)
    agree = (a.loc[points] == c.loc[points]) & (a.loc[points] != 0)
    use_base = set(agree[agree].index)
    rows = []
    for pred, chosen in ((base, True), (fallback, False)):
        m = results[results["predictor"] == pred]
        keys = pd.MultiIndex.from_frame(m[POINT])
        mask = keys.isin(points) & (keys.isin(list(use_base)) == chosen)
        rows.append(m[mask])
    out = pd.concat(rows, ignore_index=True).assign(predictor=name)
    out["gate"] = pd.MultiIndex.from_frame(out[POINT]).isin(list(use_base))
    return out[[*COLUMNS, "gate"]]
