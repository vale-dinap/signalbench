"""Common predictor interface.

    predictor.predict(task: ForecastTask) -> Forecast

Forecast.p_above holds P(value at idx+horizon > threshold) for every threshold in
task.thresholds (NaN where the predictor does not answer, e.g. direction-only
modes). Decision models, time-series foundation models (see tsfm.py) and
baselines all implement this, so they are run, stored and scored identically.

predictor.describe() returns a JSON-serialisable configuration. The backtest
stores it and refuses to resume a run whose predictor configuration changed
under the same name.
"""
from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .client import DEFAULT_URL, ask
from .elicitation import ThresholdNoul
from .forecast import ForecastTask, blind_task, horizon_scale, normal_sf
from .states import build_price_state


@dataclass
class Forecast:
    p_above: np.ndarray
    input_tokens: int = 0
    raw: dict | None = None     # audit trail (state, questions, answers, quantiles...), saved as JSONL


class Predictor(Protocol):
    name: str

    def predict(self, task: ForecastTask) -> Forecast: ...

    def describe(self) -> dict: ...


def callable_name(fn) -> str:
    """Stable description of a function or functools.partial for run metadata."""
    if isinstance(fn, functools.partial):
        args = [repr(a) for a in fn.args] + [f"{k}={v!r}" for k, v in sorted(fn.keywords.items())]
        return f"{callable_name(fn.func)}({', '.join(args)})"
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"


def describe_dataclass(obj) -> dict:
    """Generic describe(): class name plus fields, callables by qualified name."""
    out = {"type": type(obj).__name__}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if hasattr(v, "describe"):
            v = v.describe()
        elif callable(v):
            v = callable_name(v)
        out[f.name] = v
    return out


# ── Decision models ───────────────────────────────────────────────────────────

@dataclass
class DecisionModelPredictor:
    """A System One decision model queried with a given elicitation and state builder.

    elicitation   : how the forecast is asked (see elicitation.py)
    state_builder : callable(task) -> str; default full-history price state.
                    Any context (related series, news, another model's forecast)
                    goes in through here.
    blind         : ask on forecast.blind_task(task, blind_window) instead: no name,
                    no dates, prices rebased, so the model has nothing to recall.
    """
    model: str
    elicitation: object = field(default_factory=ThresholdNoul)
    state_builder: Callable[[ForecastTask], str] = build_price_state
    name: str = ""
    blind: bool = False
    blind_window: int = 60
    url: str = DEFAULT_URL
    timeout: int = 300

    def __post_init__(self):
        self.name = self.name or f"{self.model}/{self.elicitation.name}" + ("/blind" if self.blind else "")

    def predict(self, task: ForecastTask) -> Forecast:
        if self.blind:
            task = blind_task(task, self.blind_window)
        state = self.state_builder(task)
        questions = self.elicitation.questions(task)
        resp = ask(self.model, state, questions, timeout=self.timeout, url=self.url)
        p = self.elicitation.decode(resp.answers, task)
        return Forecast(p, resp.input_tokens, {"state": state, "questions": questions, "answers": resp.answers})

    def describe(self) -> dict:
        d = describe_dataclass(self)
        d.pop("url"); d.pop("timeout")   # transport settings don't change results
        if not self.blind:               # keeps earlier runs' stored descriptions valid
            d.pop("blind"); d.pop("blind_window")
        return d


# ── Baselines ─────────────────────────────────────────────────────────────────

@dataclass
class RandomWalkPredictor:
    """Zero-drift Gaussian random walk with volatility from the recent window.

    The reference every model has to beat.
    """
    window: int = 30
    name: str = "random_walk"

    def predict(self, task: ForecastTask) -> Forecast:
        scale = horizon_scale(task.series, task.idx, task.horizon, self.window)
        th = task.thresholds
        z = np.log(th / task.current) / scale if task.series.kind == "price" else (th - task.current) / scale
        return Forecast(normal_sf(z))

    def describe(self) -> dict:
        return describe_dataclass(self)


@dataclass
class EmpiricalPredictor:
    """Applies the empirical distribution of past `horizon`-step changes to the current value.

    Captures skew, fat tails and the 'often flat' behaviour of prediction markets,
    which the Gaussian walk does not.
    """
    window: int = 90
    min_samples: int = 10
    name: str = "empirical"

    def predict(self, task: ForecastTask) -> Forecast:
        x = task.history[-(self.window + 1):]
        h, th = task.horizon, task.thresholds
        if len(x) - h < self.min_samples:
            return Forecast(np.full(len(th), np.nan))
        if task.series.kind == "price":
            outcomes = task.current * np.exp(np.log(x[h:] / x[:-h]))
        else:
            outcomes = np.clip(task.current + (x[h:] - x[:-h]), 0.0, 1.0)
        return Forecast((outcomes[None, :] > th[:, None]).mean(axis=1))

    def describe(self) -> dict:
        return describe_dataclass(self)


@dataclass
class ConstantPredictor:
    """Fixed P(> x) at every threshold, e.g. 'always 0.5' for sanity checks."""
    p: float = 0.5
    name: str = "constant"

    def predict(self, task: ForecastTask) -> Forecast:
        return Forecast(np.full(len(task.thresholds), self.p))

    def describe(self) -> dict:
        return describe_dataclass(self)
