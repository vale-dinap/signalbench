"""Time-series foundation models (e.g. Google TimesFM 3.0) as predictors.

A QuantileForecaster takes a numeric context and returns quantiles of the value
at the forecast horizon. QuantilePredictor feeds it the task's history and maps
the quantiles onto the task's threshold grid, so these models are scored exactly
like decision models.

Backends: TimesFM3 (main, with related series as covariates), TimesFM25
(previous generation, univariate), GaussianForecaster (reference for tests).
They import their libraries lazily, so the package works without them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data import known_rows
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
                    quantiles back in price units); ignored for probability series.
                    Related price series are logged too
    use_related   : pass the task's related series as covariates, keyed by
                    series.key, with the values known at each main-series close
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
        cov = self._covariates(task, len(ctx), log) if self.use_related and task.related else None
        q = np.asarray(self.forecaster.forecast(ctx, task.horizon, cov), dtype=float)
        if log:
            q = np.exp(q)
        if task.series.kind == "probability":
            q = np.clip(q, 0.0, 1.0)
        p = p_above_from_quantiles(self.forecaster.levels, q, task.thresholds)
        return Forecast(p, raw={"levels": list(map(float, self.forecaster.levels)), "quantiles": q.tolist()})

    @staticmethod
    def _covariates(task: ForecastTask, n: int, log: bool = False) -> dict[str, np.ndarray]:
        """Each related series' latest known value at each of the main series' last n closes.

        Aligned on availability times (Series.available_at), so a related bar that
        closes after the main one is not used. NaN before a related series starts.
        With log=True, price series are logged like the main context.
        """
        times = task.series.available_at()[task.idx + 1 - n:task.idx + 1]
        out = {}
        for other in task.related:
            pos = known_rows(other, times) - 1
            vals = other.df["close"].to_numpy()
            if log and other.kind == "price":
                vals = np.log(vals)
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


_MODELS: dict[tuple, object] = {}   # loaded checkpoints, shared by backends with the same settings


def _load(key: tuple, loader):
    if key not in _MODELS:
        _MODELS[key] = loader()
    return _MODELS[key]


def release() -> None:
    """Drop every loaded checkpoint and return its GPU memory to the device.

    Call it before other GPU-heavy work (e.g. a decision-model server sharing the
    card); the next forecast reloads the checkpoint.
    """
    import gc
    import sys
    _MODELS.clear()
    gc.collect()
    torch = sys.modules.get("torch")   # only if a backend already imported it
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _version(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return "not installed"


@dataclass
class TimesFM3:
    """Google TimesFM 3.0 (PyTorch), pip install "signalbench[tsfm]".

    Related series are passed as past-only covariates (values known up to the
    decision time only), one row per series; leading NaNs are interpolated by
    timesfm3 and all-NaN covariates are dropped.

    checkpoint : Hugging Face repo id or local checkpoint
    device     : "cuda" / "cpu"; None picks cuda when available. Not part of
                 describe(), so a run can resume on other hardware
    """
    checkpoint: str = "google/timesfm-3.0-pytorch"
    device: str | None = None

    def __post_init__(self):
        self.levels = DECILES

    def _model(self):
        def load():
            from timesfm3 import TimesFM3Forecaster
            m = TimesFM3Forecaster.from_pretrained(self.checkpoint, device=self.device)
            if not np.allclose(m.config.quantiles, self.levels):
                raise ValueError(f"{self.checkpoint} predicts quantiles {m.config.quantiles}, expected deciles")
            return m
        return _load(("timesfm3", self.checkpoint, self.device), load)

    def forecast(self, context, horizon, covariates=None):
        cov = [c for c in (covariates or {}).values() if not np.isnan(c).all()]
        out = self._model().predict(np.asarray(context, dtype=float), horizon,
                                    past_only_covariates=np.vstack(cov) if cov else None,
                                    return_quantiles=True)
        return np.asarray(out.quantiles)[horizon - 1]

    def describe(self) -> dict:
        d = describe_dataclass(self)
        d.pop("device")
        return {**d, "timesfm": _version("timesfm")}


@dataclass
class TimesFM25:
    """Google TimesFM 2.5 200M (PyTorch), the previous-generation reference.

    Univariate only: 2.5 takes covariates through its XReg linear model, which
    needs JAX and future values of every covariate, so it raises if given any.

    max_context / max_horizon : compiled decode sizes (contexts are truncated
                                to max_context; horizons above max_horizon fail)
    """
    checkpoint: str = "google/timesfm-2.5-200m-pytorch"
    max_context: int = 1024
    max_horizon: int = 256

    def __post_init__(self):
        self.levels = DECILES

    def _model(self):
        def load():
            import timesfm
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint, torch_compile=False)
            m.compile(timesfm.ForecastConfig(
                max_context=self.max_context, max_horizon=self.max_horizon, normalize_inputs=True,
                use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
                fix_quantile_crossing=True))
            return m
        return _load(("timesfm2.5", self.checkpoint, self.max_context, self.max_horizon), load)

    def forecast(self, context, horizon, covariates=None):
        if covariates:
            raise ValueError("TimesFM25 does not support past-only covariates; use TimesFM3")
        _, q = self._model().forecast(horizon=horizon, inputs=[np.asarray(context, dtype=float)])
        return q[0, horizon - 1, 1:]     # column 0 is the mean, then the deciles

    def describe(self) -> dict:
        return {**describe_dataclass(self), "timesfm": _version("timesfm")}


def _normal_ppf(p) -> np.ndarray:
    """Inverse standard-normal CDF by bisection on normal_sf (no scipy dependency)."""
    p = np.asarray(p, dtype=float)
    lo, hi = np.full_like(p, -10.0), np.full_like(p, 10.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        below = (1 - normal_sf(mid)) < p
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2
