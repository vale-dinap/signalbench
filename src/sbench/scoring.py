"""Scoring for long-format backtest results (see backtest.run_forecast_backtest).

Per forecast (one predictor at one decision time):
    p_up       : P(future > current), the answer at the 0-sigma threshold
    brier_up   : Brier score of p_up; defined for every predictor, including
                 direction-only ones (lower is better)
    logloss_up : log loss of p_up (clipped at 1e-6; punishes confident misses)
    dir_hit    : 1 if (p_up > 0.5) == (future > current); NaN when p_up == 0.5
    brier      : mean Brier score over all answered thresholds (raw answers,
                 so incoherent answers are penalised)
    crps_n     : CRPS of the monotone CDF, divided by the horizon scale so it is
                 comparable across assets and volatility regimes (log-return units
                 for prices). Only for full-grid forecasts (lower is better)
    q10/q50/q90, in_q10_q90 : quantiles and whether the outcome fell in the 80% band
    coherence  : how much raw answers violate monotonicity (0 = coherent)

Ties: an outcome equal to the current value counts as "not up", matching the
strict "above" wording of the questions. flat_rate in the summary shows how often
that happens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import KEY
from .forecast import crps, quantiles

EPS = 1e-6


def per_forecast(results: pd.DataFrame, q_levels=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    """One row per (series, predictor, t, horizon) with derived metrics."""
    rows = []
    for key, g in results.groupby(KEY, sort=False):
        g = g.sort_values("threshold")
        th, p = g["threshold"].to_numpy(dtype=float), g["p_above"].to_numpy(dtype=float)
        outcome = g["outcome"].astype(bool).to_numpy()
        answered = ~np.isnan(p)
        if not answered.any():
            continue
        cur, fut, scale, kind = (g[c].iloc[0] for c in ("current", "future", "scale", "kind"))
        up = bool(fut > cur)
        at0 = g["sigma"].to_numpy() == 0
        p_up = float(p[at0][0]) if at0.any() and answered[at0][0] else np.nan

        row = {
            **dict(zip(KEY, key)),
            "kind": kind, "current": cur, "future": fut, "scale": scale,
            "up": up, "flat": bool(fut == cur),
            "p_up": p_up,
            "brier_up": (p_up - up) ** 2 if not np.isnan(p_up) else np.nan,
            "logloss_up": (-np.log(np.clip(p_up if up else 1 - p_up, EPS, 1)) if not np.isnan(p_up) else np.nan),
            "dir_hit": np.nan if np.isnan(p_up) or p_up == 0.5 else float((p_up > 0.5) == up),
            "brier": float(np.mean((p[answered] - outcome[answered]) ** 2)),
            "n_thresholds": int(answered.sum()),
            "coherence": g["coherence"].iloc[0],
            "p_spread": float(np.nanmax(p) - np.nanmin(p)),
        }
        if answered.all() and len(th) >= 3:
            if kind == "price":
                row["crps_n"] = crps(np.log(th), p, np.log(fut), kind="logprice") / scale
                qs = np.exp(quantiles(np.log(th), p, q_levels, kind="logprice"))
            else:
                row["crps_n"] = crps(th, p, fut, kind=kind) / scale
                qs = quantiles(th, p, q_levels, kind=kind)
            row.update({f"q{round(lv * 100)}": v for lv, v in zip(q_levels, qs)})
            row["in_q10_q90"] = float(qs[0] <= fut <= qs[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(x, fn=np.mean, n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = [fn(rng.choice(x, len(x))) for _ in range(n)]
    return tuple(np.quantile(stats, [alpha / 2, 1 - alpha / 2]))


def summary(per: pd.DataFrame, baseline: str | None = "random_walk", ci: bool = True,
            by=("series", "horizon")) -> pd.DataFrame:
    """Per (*by, predictor) table; by=("horizon",) pools all series.

    *_skill: 1 - mean(score) / mean(baseline score) on decision points both
    answered (> 0 = better than baseline). *_ci: 95% bootstrap interval of the
    paired difference (baseline - model); entirely above 0 = reliably better.
    """
    def mean(m, c):
        return m[c].mean() if c in m else np.nan

    by = list(by)
    out = []
    for keys, g in per.groupby(by):
        base = g[g["predictor"] == baseline].set_index(["series", "t"]) if baseline else None
        for pred, m in g.groupby("predictor", sort=False):
            row = {
                **dict(zip(by, keys)), "predictor": pred, "N": len(m),
                "up_rate": m["up"].mean(), "flat_rate": m["flat"].mean(),
                "dir_acc": m["dir_hit"].mean(), "dir_N": int(m["dir_hit"].notna().sum()),
                "brier_up": mean(m, "brier_up"), "logloss_up": mean(m, "logloss_up"),
                "brier": mean(m, "brier"), "crps_n": mean(m, "crps_n"),
                "cover_80": mean(m, "in_q10_q90"), "coherence": mean(m, "coherence"),
                "p_spread": mean(m, "p_spread"),
            }
            if base is not None and pred != baseline and len(base):
                j = m.set_index(["series", "t"]).join(base, rsuffix="_b", how="inner")
                for metric in ("brier_up", "crps_n"):
                    if metric not in j or f"{metric}_b" not in j:
                        continue
                    pair = j[[metric, f"{metric}_b"]].dropna()
                    if pair.empty:
                        continue
                    row[f"{metric}_skill"] = 1 - pair[metric].mean() / pair[f"{metric}_b"].mean()
                    if ci:
                        lo, hi = bootstrap_ci((pair[f"{metric}_b"] - pair[metric]).to_numpy())
                        row[f"{metric}_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"
            out.append(row)
    t = pd.DataFrame(out)
    return t.sort_values([*by, "brier_up"]).set_index([*by, "predictor"]).round(4)


def skill_table(per: pd.DataFrame, metric: str = "crps_n", baseline: str = "random_walk",
                by=("horizon",), n: int = 2000, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Skill 1 - mean(score) / mean(baseline score) with a paired bootstrap CI, for charts.

    One row per (*by, predictor) with columns skill, lo, hi, N. Pairs are matched
    on (series, t), so by=("horizon",) pools series; decision points are resampled
    as pairs, so the interval reflects the comparison rather than each score alone.
    """
    by = list(by)
    rng = np.random.default_rng(seed)
    out = []
    for keys, g in per.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = g[g["predictor"] == baseline].set_index(["series", "t"])[metric]
        for pred, m in g.groupby("predictor", sort=False):
            if pred == baseline or metric not in m:
                continue
            pair = pd.concat([m.set_index(["series", "t"])[metric], base], axis=1, join="inner").dropna()
            if len(pair) < 2:
                continue
            x, b = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
            idx = rng.integers(0, len(x), (n, len(x)))
            boot = 1 - x[idx].mean(axis=1) / b[idx].mean(axis=1)
            lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
            out.append({**dict(zip(by, keys)), "predictor": pred, "skill": 1 - x.mean() / b.mean(),
                        "lo": lo, "hi": hi, "N": len(x)})
    return pd.DataFrame(out)


def reliability(results: pd.DataFrame, bins: int = 10, sigma0_only: bool = False) -> pd.DataFrame:
    """Calibration of raw answers: mean forecast vs observed frequency per probability bin."""
    r = results.dropna(subset=["p_above"])
    if sigma0_only:
        r = r[r["sigma"] == 0]
    r = r.assign(bin=pd.cut(r["p_above"], np.linspace(0, 1, bins + 1), include_lowest=True))
    return (r.groupby(["predictor", "bin"], observed=True)
             .agg(n=("outcome", "size"), forecast=("p_above", "mean"), observed=("outcome", "mean"))
             .reset_index())


def rank_ic(df: pd.DataFrame, x: str, y: str, date: str = "t", n: int = 2000, alpha: float = 0.05,
            seed: int = 0) -> dict:
    """Information coefficient: rank correlation of a signal `x` with an outcome `y`, with a bootstrap interval.

    The bootstrap resamples dates, keeping every instrument of a date together, so
    the interval allows for moves shared across instruments on the same day.
    Returns ic, lo, hi, N (rows) and dates.
    """
    d = df[[x, y, date]].dropna()
    if len(d) < 3:
        return {"ic": np.nan, "lo": np.nan, "hi": np.nan, "N": len(d), "dates": d[date].nunique()}
    rx, ry = d[x].rank().to_numpy(), d[y].rank().to_numpy()
    ic = float(np.corrcoef(rx, ry)[0, 1])
    groups = d.groupby(date).indices
    blocks = list(groups.values())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        bx, by = pd.Series(d[x].to_numpy()[idx]).rank(), pd.Series(d[y].to_numpy()[idx]).rank()
        boot.append(np.corrcoef(bx, by)[0, 1])
    lo, hi = np.nanquantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"ic": ic, "lo": float(lo), "hi": float(hi), "N": len(d), "dates": len(blocks)}
