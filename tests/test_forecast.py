import numpy as np
import pandas as pd
import pytest

from sbench.data import Series
from sbench.forecast import (
    coherence_violation,
    crps,
    isotonic_decreasing,
    make_task,
    p_above_from_quantiles,
    quantiles,
    threshold_grid,
)


def test_isotonic_decreasing_pools_violations():
    y = isotonic_decreasing([0.9, 0.8, 0.85, 0.5, 0.6, 0.1])
    assert np.all(np.diff(y) <= 0)
    np.testing.assert_allclose(y, [0.9, 0.825, 0.825, 0.55, 0.55, 0.1])


def test_coherence_violation():
    assert coherence_violation([0.9, 0.7, 0.2]) == 0
    assert coherence_violation([0.9, 0.7, 0.8, 0.2]) == pytest.approx(0.1)


def test_threshold_grid_contains_current_value(gbm):
    th, sig, scale = threshold_grid(gbm, 100, horizon=1)
    assert np.all(np.diff(th) > 0)
    assert th[sig == 0][0] == gbm.df["close"].iloc[100]
    assert scale > 0


def test_probability_grid_stays_inside_unit_interval():
    t = pd.date_range("2026-01-01", periods=60, freq="D")
    s = Series("pm", "Market", pd.DataFrame({"t": t, "close": np.linspace(0.97, 0.995, 60)}),
               kind="probability", value_label="Yes price")
    th, sig, _ = threshold_grid(s, 59, horizon=1)
    assert th.min() > 0 and th.max() < 1
    assert 0 in sig


def test_zero_sigma_survives_rounding_collisions():
    t = pd.date_range("2026-01-01", periods=40, freq="D")
    s = Series("flat", "Flat market", pd.DataFrame({"t": t, "close": [0.5] * 40}),
               kind="probability", value_label="Yes price")
    th, sig, _ = threshold_grid(s, 39, horizon=1)
    assert 0 in sig and th[sig == 0][0] == 0.5


def test_task_exposes_no_future_data(gbm):
    task = make_task(gbm, 100, horizon=3)
    assert len(task.history) == 101
    assert task.history[-1] == task.current
    assert task.t == gbm.df["t"].iloc[100]


def test_task_related_history_is_as_of(gbm):
    other = Series("other", "Other", gbm.df.iloc[::2].reset_index(drop=True))
    task = make_task(gbm, 101, 1, related=(other,))
    rel = task.related_history(other)
    assert other.df["t"].iloc[len(rel) - 1] <= task.t


def test_quantiles_are_monotone():
    th = np.array([90, 95, 100, 105, 110.0])
    p = np.array([0.95, 0.8, 0.5, 0.2, 0.05])
    q = quantiles(th, p, (0.1, 0.5, 0.9))
    assert q[0] < q[1] < q[2]
    assert q[1] == pytest.approx(100)


def test_quantile_mapping_roundtrip():
    levels = np.arange(1, 10) / 10
    values = 100 + np.linspace(-4, 4, 9)
    p = p_above_from_quantiles(levels, values, values)
    np.testing.assert_allclose(p, 1 - levels, atol=1e-9)


def test_quantile_mapping_handles_crossing():
    p = p_above_from_quantiles([0.1, 0.5, 0.9], [101, 99, 100], [98, 100, 102])
    assert np.all(np.diff(p) <= 0)


def test_crps_prefers_sharp_correct_forecast():
    th = np.array([97, 98, 99, 100, 101, 102, 103.0])
    sharp = np.array([1, 1, 0.95, 0.5, 0.05, 0, 0])
    wide = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    assert crps(th, sharp, 100.0) < crps(th, wide, 100.0)
    assert crps(th, sharp, 103.0) > crps(th, wide, 103.0)
