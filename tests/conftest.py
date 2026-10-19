import numpy as np
import pandas as pd
import pytest

from sbench.data import Series


def make_gbm(n=300, vol=0.02, seed=1, key="gbm") -> Series:
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    t = pd.date_range("2025-01-01", periods=n, freq="D")
    return Series(key, "Synthetic asset", pd.DataFrame({"t": t, "close": px}), unit="USD")


def make_market(n=120, seed=2, key="pm") -> Series:
    """Prediction-market-like: bounded, often flat, occasional jumps."""
    rng = np.random.default_rng(seed)
    steps = np.where(rng.random(n) < 0.5, 0.0, rng.normal(0, 0.02, n))
    p = np.clip(0.6 + np.cumsum(steps), 0.02, 0.98).round(3)
    t = pd.date_range("2026-01-01", periods=n, freq="D")
    return Series(key, "Prediction market: test", pd.DataFrame({"t": t, "close": p}),
                  kind="probability", value_label="Yes price")


@pytest.fixture
def gbm() -> Series:
    return make_gbm()


@pytest.fixture
def market() -> Series:
    return make_market()
