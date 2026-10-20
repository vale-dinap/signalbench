"""Synthetic price series with known conditional distributions, and the oracle.

Each process is simulated in log-price space. Given everything up to a decision
index, including latent state such as the volatility regime, the true
distribution of the log price `h` steps ahead is a Gaussian mixture. The
OraclePredictor evaluates it on the task's threshold grid, giving the best
achievable forecast: the ceiling every other predictor is measured against.

    RandomWalk        r ~ N(mu, sigma)                       mu != 0 gives a trend
    MeanReversion     AR(1) in log price around a fixed level
    RegimeVolatility  zero-drift walk, sigma switches between two Markov regimes
    JumpDiffusion     Gaussian walk plus Poisson jumps (Merton)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .data import Series
from .forecast import ForecastTask, normal_sf
from .predictors import Forecast, describe_dataclass


@dataclass
class Mixture:
    """Gaussian mixture over the log price at the horizon."""
    weights: np.ndarray
    means: np.ndarray
    stds: np.ndarray

    def sf(self, x) -> np.ndarray:
        """P(log value > x) for each x."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        z = (x[:, None] - self.means[None, :]) / self.stds[None, :]
        return normal_sf(z) @ self.weights

    def ppf(self, levels) -> np.ndarray:
        """Quantiles by bisection on the CDF."""
        levels = np.asarray(levels, dtype=float)
        spread = 10 * self.stds.max()
        lo, hi = np.full_like(levels, self.means.min() - spread), np.full_like(levels, self.means.max() + spread)
        for _ in range(60):
            mid = (lo + hi) / 2
            below = (1 - self.sf(mid)) < levels
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return (lo + hi) / 2


def _single(mean: float, std: float) -> Mixture:
    return Mixture(np.array([1.0]), np.array([mean]), np.array([std]))


class Process(Protocol):
    def simulate(self, n: int, y0: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
        """n log prices starting at y0, plus the latent state per step (None if there is none)."""
        ...

    def conditional(self, y: np.ndarray, latent: np.ndarray | None, h: int) -> Mixture:
        """Distribution of the log price h steps after y[-1], given y and latent up to now."""
        ...


# ── Processes ─────────────────────────────────────────────────────────────────

@dataclass
class RandomWalk:
    mu: float = 0.0
    sigma: float = 0.02

    def simulate(self, n, y0, rng):
        r = rng.normal(self.mu, self.sigma, n - 1)
        return y0 + np.concatenate([[0.0], np.cumsum(r)]), None

    def conditional(self, y, latent, h):
        return _single(y[-1] + h * self.mu, self.sigma * math.sqrt(h))


@dataclass
class MeanReversion:
    """y[t+1] = level + phi * (y[t] - level) + sigma * e; level defaults to y0."""
    phi: float = 0.9
    sigma: float = 0.02
    level: float | None = None

    def simulate(self, n, y0, rng):
        m = y0 if self.level is None else self.level
        y = np.empty(n)
        y[0] = y0
        for t in range(1, n):
            y[t] = m + self.phi * (y[t - 1] - m) + rng.normal(0, self.sigma)
        return y, None

    def conditional(self, y, latent, h):
        m = y[0] if self.level is None else self.level
        var = self.sigma ** 2 * (1 - self.phi ** (2 * h)) / (1 - self.phi ** 2)
        return _single(m + self.phi ** h * (y[-1] - m), math.sqrt(var))


@dataclass
class RegimeVolatility:
    """Two-state Markov volatility. The regime moves first, then r[t] ~ N(0, sigma[regime[t]]).

    The latent state is the regime per step (0 = calm, 1 = turbulent); the
    oracle knows the current regime, other predictors have to infer it.
    """
    sigma_calm: float = 0.01
    sigma_turbulent: float = 0.035
    stay_calm: float = 0.98
    stay_turbulent: float = 0.97

    def _transition(self) -> np.ndarray:
        return np.array([[self.stay_calm, 1 - self.stay_calm], [1 - self.stay_turbulent, self.stay_turbulent]])

    def simulate(self, n, y0, rng):
        P = self._transition()
        s = np.zeros(n, dtype=int)
        for t in range(1, n):
            s[t] = int(rng.random() > P[s[t - 1], s[t - 1]]) ^ s[t - 1]
        sig = np.where(s == 1, self.sigma_turbulent, self.sigma_calm)
        r = rng.normal(0, 1, n - 1) * sig[1:]
        return y0 + np.concatenate([[0.0], np.cumsum(r)]), s

    def conditional(self, y, latent, h):
        # dist[s, k] = P(regime s now, k turbulent steps so far), propagated h steps
        P = self._transition()
        dist = np.zeros((2, h + 1))
        dist[int(latent[-1]), 0] = 1.0
        for _ in range(h):
            nxt = np.zeros_like(dist)
            nxt[0] = dist[0] * P[0, 0] + dist[1] * P[1, 0]
            nxt[1, 1:] = (dist[0] * P[0, 1] + dist[1] * P[1, 1])[:-1]
            dist = nxt
        w = dist.sum(axis=0)
        k = np.arange(h + 1)
        std = np.sqrt(k * self.sigma_turbulent ** 2 + (h - k) * self.sigma_calm ** 2)
        return Mixture(w, np.full(h + 1, y[-1]), std)


@dataclass
class JumpDiffusion:
    """r[t] = mu + sigma * e + sum of N[t] ~ Poisson(jump_rate) jumps ~ N(jump_mean, jump_std).

    The latent state is the number of jumps per step (for charts; the oracle
    does not need it).
    """
    mu: float = 0.0
    sigma: float = 0.01
    jump_rate: float = 0.05
    jump_mean: float = 0.0
    jump_std: float = 0.05
    max_jumps: int = 25

    def simulate(self, n, y0, rng):
        k = rng.poisson(self.jump_rate, n - 1)
        jumps = rng.normal(self.jump_mean * k, self.jump_std * np.sqrt(k))
        r = rng.normal(self.mu, self.sigma, n - 1) + jumps
        return y0 + np.concatenate([[0.0], np.cumsum(r)]), np.concatenate([[0], k])

    def conditional(self, y, latent, h):
        lam = self.jump_rate * h
        k = np.arange(self.max_jumps + 1)
        w = np.cumprod(np.concatenate([[math.exp(-lam)], lam / k[1:]]))   # Poisson pmf
        w = w / w.sum()
        return Mixture(w, y[-1] + h * self.mu + k * self.jump_mean,
                       np.sqrt(h * self.sigma ** 2 + k * self.jump_std ** 2))


# ── Series and oracle ─────────────────────────────────────────────────────────

def synthetic_series(key: str, process: Process, n: int = 400, seed: int = 0, p0: float = 100.0,
                     start="2025-01-01", description: str = "Asset: daily close price",
                     unit: str = "USD") -> Series:
    """Simulate a daily price Series. The process and latent path are kept in meta for the oracle.

    description is what decision models see, so it should not reveal the process.
    """
    y, latent = process.simulate(n, math.log(p0), np.random.default_rng(seed))
    df = pd.DataFrame({"t": pd.date_range(start, periods=n, freq="D"), "close": np.exp(y)})
    return Series(key, description, df, kind="price", unit=unit,
                  meta={"source": "synthetic", "process": process, "latent": latent, "seed": seed})


def true_distribution(task: ForecastTask) -> Mixture:
    """The process's distribution of the log price at the task's horizon, from data up to task.idx."""
    meta = task.series.meta
    if meta.get("source") != "synthetic":
        raise ValueError(f"{task.series.key} is not a synthetic series")
    y = np.log(task.history)
    latent = meta["latent"][:task.idx + 1] if meta["latent"] is not None else None
    return meta["process"].conditional(y, latent, task.horizon)


@dataclass
class OraclePredictor:
    """Exact P(value > threshold) from the generating process (synthetic series only)."""
    name: str = "oracle"

    def predict(self, task: ForecastTask) -> Forecast:
        return Forecast(true_distribution(task).sf(np.log(task.thresholds)))

    def describe(self) -> dict:
        return describe_dataclass(self)
