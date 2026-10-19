"""Time-series foundation models (e.g. Google TimesFM 3.0) as predictors.

A QuantileForecaster takes a numeric context and returns quantiles of the value
at the forecast horizon. QuantilePredictor feeds it the task's history and maps
the quantiles onto the task's threshold grid, so these models are scored exactly
like decision models.

Backends import their libraries lazily, so the package works without them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .forecast import ForecastTask, normal_sf, p_above_from_quantiles
from .predictors import Forecast, describe_dataclass

DECILES = np.round(np.arange(1, 10) / 10, 2)


class QuantileForecaster(Protocol):
    levels: np.ndarray

    def forecast(self, context: np.ndarray, horizon: int,
                 covariates: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Quantiles (aligned with self.levels) of the value `horizon` steps after context[-1]."""
        ...

    def describe(self) -> dict: ...


@dataclass
class QuantilePredictor:
    """Wraps a QuantileForecaster as a Predictor.

    max_context   : most recent points passed as context (None = all)
    log_space     : forecast log values for price series (returns the model
                    quantiles back in price units); ignored for probability series
    use_related   : pass the task's related series as covariates, keyed by
                    series.key and aligned as-of to the main series timestamps
    """
    forecaster: QuantileForecaster
    name: str = "tsfm"
    max_context: int | None = 1024
    log_space: bool = False
    use_related: bool = False

    def predict(self, task: ForecastTask) -> Forecast:
        ctx = task.history if self.max_context is None else task.history[-self.max_context:]
        log = self.log_space and task.series.kind == "price"
        if log:
            ctx = np.log(ctx)
        cov = self._covariates(task, len(ctx)) if self.use_related and task.related else None
        q = np.asarray(self.forecaster.forecast(ctx, task.horizon, cov), dtype=float)
        if log:
            q = np.exp(q)
        if task.series.kind == "probability":
            q = np.clip(q, 0.0, 1.0)
        p = p_above_from_quantiles(self.forecaster.levels, q, task.thresholds)
        return Forecast(p, raw={"levels": list(map(float, self.forecaster.levels)), "quantiles": q.tolist()})

    @staticmethod
    def _covariates(task: ForecastTask, n: int) -> dict[str, np.ndarray]:
        """Each related series as-of aligned to the last n timestamps of the main series."""
        times = task.series.df["t"].to_numpy()[task.idx + 1 - n:task.idx + 1]
        out = {}
        for other in task.related:
            pos = other.df["t"].searchsorted(times, side="right") - 1
            vals = other.df["close"].to_numpy()
            out[other.key] = np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)
        return out

    def describe(self) -> dict:
        return describe_dataclass(self)


# ── Backends ──────────────────────────────────────────────────────────────────

@dataclass
class GaussianForecaster:
    """Reference backend: zero-drift Gaussian random walk on the context.

    Not a real model; used to validate the quantile -> threshold mapping (it must
    reproduce RandomWalkPredictor up to discretisation).
    """
    window: int = 30
    min_scale: float = 1e-9

    def __post_init__(self):
        self.levels = DECILES

    def forecast(self, context, horizon, covariates=None):
        d = np.diff(np.asarray(context, dtype=float)[-(self.window + 1):])
        s = max(float(np.std(d, ddof=1)) if len(d) > 1 else 0.0, self.min_scale) * np.sqrt(horizon)
        z = _normal_ppf(self.levels)
        return context[-1] + z * s

    def describe(self) -> dict:
        return describe_dataclass(self)


def _normal_ppf(p) -> np.ndarray:
    """Inverse standard-normal CDF by bisection on normal_sf (no scipy dependency)."""
    p = np.asarray(p, dtype=float)
    lo, hi = np.full_like(p, -10.0), np.full_like(p, 10.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        below = (1 - normal_sf(mid)) < p
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2
