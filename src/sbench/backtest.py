"""Resumable forecast backtests.

For each decision index one ForecastTask is built and given to every predictor.

Files written next to csv_path:
    <name>.csv        long format, one row per (series, predictor, t, horizon, threshold)
    <name>.meta.json  run configuration per (series, horizon): data fingerprint, grid
                      settings, predictor descriptions; resuming with a changed
                      configuration raises instead of silently mixing results
    <name>.raw.jsonl  audit trail per forecast (state text, questions, answers,
                      quantiles), when the predictor provides one

Rows are appended after each forecast, so an interrupted run resumes where it
stopped. Failed forecasts are reported and not written, so they are retried on
the next run.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Series
from .forecast import DEFAULT_SIGMAS, coherence_violation, make_task

KEY = ["series", "predictor", "t", "horizon"]
COLUMNS = ["series", "kind", "predictor", "t", "horizon", "current", "future", "scale", "sigma",
           "threshold", "p_above", "outcome", "coherence", "input_tokens", "latency_s"]


def decision_points(series: Series, horizon: int, min_history: int = 30, step: int = 1,
                    max_points: int | None = None, start=None, end=None) -> list[int]:
    """Indices with at least min_history past points and a realised value at idx+horizon.

    step / max_points thin the evaluation set (max_points spaces points evenly).
    start / end restrict decision times.
    """
    t = series.df["t"]
    idxs = np.arange(max(min_history, 1), len(series) - horizon)
    if start is not None:
        idxs = idxs[(t.iloc[idxs] >= pd.Timestamp(start)).to_numpy()]
    if end is not None:
        idxs = idxs[(t.iloc[idxs] <= pd.Timestamp(end)).to_numpy()]
    idxs = idxs[::step]
    if max_points is not None and len(idxs) > max_points:
        idxs = idxs[np.linspace(0, len(idxs) - 1, max_points).round().astype(int)]
    return [int(i) for i in idxs]


def load_results(csv_path: str | Path) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p, parse_dates=["t"])


def load_raw(csv_path: str | Path) -> list[dict]:
    p = _sidecar(csv_path, ".raw.jsonl")
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── Run metadata ──────────────────────────────────────────────────────────────

def _sidecar(csv_path, suffix: str) -> Path:
    p = Path(csv_path)
    return p.with_name(p.stem + suffix)


def _fingerprint(series: Series, n: int, unit: str = "ns") -> str:
    """Hash of the first n rows, so appending new data keeps old runs valid."""
    df = series.df.iloc[:n]
    h = hashlib.sha1(df["t"].astype(f"datetime64[{unit}]").astype("int64").to_numpy().tobytes())
    h.update(df["close"].to_numpy(dtype=float).tobytes())
    return h.hexdigest()[:16]


def _same_data(series: Series, n: int, stored: str) -> bool:
    # Fingerprints written before timestamps were normalised to ns hashed the source's own resolution.
    return any(_fingerprint(series, n, unit) == stored for unit in ("ns", "us", "ms", "s"))


def _check_meta(csv_path: Path, series: Series, horizon: int, sigmas, vol_window: int,
                related: tuple, predictors: list) -> None:
    path = _sidecar(csv_path, ".meta.json")
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = f"{series.key}|h={horizon}"
    cfg = {"sigmas": [float(s) for s in sigmas], "vol_window": vol_window,
           "related": [r.key for r in related], "kind": series.kind}
    entry = meta.get(key)
    if entry is None:
        entry = {"n_rows": len(series), "fingerprint": _fingerprint(series, len(series)), **cfg, "predictors": {}}
    else:
        problems = [f"{k}: stored {entry.get(k)!r}, now {v!r}" for k, v in cfg.items() if entry.get(k) != v]
        n = entry["n_rows"]
        if len(series) < n or not _same_data(series, n, entry["fingerprint"]):
            problems.append(f"series data differs from the first {n} rows used before")
        if problems:
            raise ValueError(f"{path.name} [{key}] configuration changed:\n  " + "\n  ".join(problems)
                             + "\nUse a new csv_path for a new configuration.")
        entry["fingerprint"] = _fingerprint(series, n)
    for pred in predictors:
        desc = json.loads(json.dumps(pred.describe(), default=str))
        old = entry["predictors"].get(pred.name)
        if old is not None and old != desc:
            raise ValueError(f"{path.name} [{key}] predictor '{pred.name}' configuration changed:\n"
                             f"  stored {old}\n  now    {desc}\nGive it a new name or use a new csv_path.")
        entry["predictors"][pred.name] = desc
    meta[key] = entry
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ── Runner ────────────────────────────────────────────────────────────────────

def _append_csv(csv_path: Path, rows: pd.DataFrame) -> None:
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        f.write(rows[COLUMNS].to_csv(header=new, index=False))


def _check_header(csv_path: Path) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != COLUMNS:
            raise ValueError(f"{csv_path.name} has columns {header}, expected {COLUMNS}")


def run_forecast_backtest(
    series: Series,
    predictors: list,
    horizon: int,
    csv_path: str | Path,
    idxs: list[int] | None = None,
    sigmas=DEFAULT_SIGMAS,
    vol_window: int = 30,
    related: tuple[Series, ...] = (),
    save_raw: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every predictor at every decision index; resume from csv_path.

    Returns all rows of csv_path for this series and horizon.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _check_header(csv_path)
    _check_meta(csv_path, series, horizon, sigmas, vol_window, tuple(related), predictors)

    existing = load_results(csv_path)
    done = set(zip(existing["series"], existing["predictor"], map(pd.Timestamp, existing["t"]),
                   existing["horizon"]))
    if idxs is None:
        idxs = decision_points(series, horizon)

    raw_path = _sidecar(csv_path, ".raw.jsonl")
    tasks: dict[int, object] = {}
    closes = series.df["close"].to_numpy()
    for pred in predictors:
        todo = [i for i in idxs if (series.key, pred.name, series.df["t"].iloc[i], horizon) not in done]
        if verbose:
            print(f"{pred.name}: {len(idxs) - len(todo)} cached, {len(todo)} to run")
        t0, failures = time.time(), 0
        for n, i in enumerate(todo, 1):
            if i not in tasks:
                tasks[i] = make_task(series, i, horizon, sigmas, vol_window, related)
            task = tasks[i]
            future = float(closes[i + horizon])
            ts = time.time()
            try:
                fc = pred.predict(task)
                p = np.asarray(fc.p_above, dtype=float)
                if p.shape != task.thresholds.shape:
                    raise ValueError(f"p_above has shape {p.shape}, grid has {task.thresholds.shape}")
            except Exception as e:  # noqa: BLE001 -- one bad forecast must not stop a long run; logged and retried
                failures += 1
                print(f"  {pred.name} idx {i} ({task.t}): {type(e).__name__}: {e}")
                continue
            rows = pd.DataFrame({
                "series": series.key, "kind": series.kind, "predictor": pred.name, "t": task.t,
                "horizon": horizon, "current": task.current, "future": future, "scale": task.scale,
                "sigma": task.sigmas, "threshold": task.thresholds, "p_above": p,
                "outcome": future > task.thresholds, "coherence": coherence_violation(p[~np.isnan(p)]),
                "input_tokens": fc.input_tokens, "latency_s": round(time.time() - ts, 3),
            })
            _append_csv(csv_path, rows)
            if save_raw and fc.raw is not None:
                rec = {"series": series.key, "predictor": pred.name, "t": str(task.t), "horizon": horizon, **fc.raw}
                with open(raw_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=_json_default) + "\n")
            if verbose and (n % 10 == 0 or n == len(todo)):
                eta = (time.time() - t0) / n * (len(todo) - n)
                print(f"  {n}/{len(todo)}  eta {eta / 60:.1f} min")
        if failures:
            print(f"  {pred.name}: {failures} failed (not saved; rerun to retry)")

    res = load_results(csv_path)
    return res[(res["series"] == series.key) & (res["horizon"] == horizon)].reset_index(drop=True)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
