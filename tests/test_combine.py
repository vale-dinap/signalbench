import numpy as np
import pandas as pd
import pytest

from sbench import combine
from sbench.backtest import COLUMNS
from sbench.scoring import per_forecast

SIGMAS = np.array([-1.0, 0.0, 1.0])


def rows(predictor, p_above, t="2026-01-05", future=101.0, direction_only=False):
    p = np.array(p_above, dtype=float)
    if direction_only:
        p = np.where(SIGMAS == 0, p, np.nan)
    th = np.array([99.0, 100.0, 101.5])
    return pd.DataFrame({
        "series": "X", "kind": "price", "predictor": predictor, "t": pd.Timestamp(t), "horizon": 1,
        "current": 100.0, "future": future, "scale": 0.01, "sigma": SIGMAS, "threshold": th, "p_above": p,
        "outcome": future > th, "coherence": 0.0, "input_tokens": np.nan, "latency_s": 0.1,
    })[COLUMNS]


def test_linear_pool_averages_each_threshold_and_keeps_the_format():
    r = pd.concat([rows("a", [0.9, 0.6, 0.2]), rows("b", [0.7, 0.2, 0.0])])
    out = combine.linear_pool(r, {"a": 3, "b": 1}, "pool")
    assert list(out.columns) == COLUMNS and set(out["predictor"]) == {"pool"}
    np.testing.assert_allclose(out.sort_values("sigma")["p_above"], [0.85, 0.5, 0.15])
    assert per_forecast(out)["crps_n"].notna().all()


def test_pooling_with_a_direction_only_answer_gives_a_direction_answer():
    r = pd.concat([rows("a", [0.9, 0.6, 0.2]), rows("lev", [np.nan, 0.8, np.nan], direction_only=True)])
    pf = per_forecast(combine.linear_pool(r, {"a": 0.5, "lev": 0.5}, "pool"))
    assert pf["p_up"].iloc[0] == pytest.approx(0.7) and "crps_n" not in pf


def test_pool_uses_only_decision_points_both_answered():
    r = pd.concat([rows("a", [0.9, 0.6, 0.2]), rows("a", [0.9, 0.6, 0.2], t="2026-01-06"),
                   rows("b", [0.7, 0.2, 0.0])])
    assert combine.linear_pool(r, {"a": 1, "b": 1}, "pool")["t"].nunique() == 1


def test_fitted_weight_prefers_the_better_member():
    # outcome above every threshold: higher P(above) is better, so all weight goes to "a"
    r = pd.concat([rows("a", [0.95, 0.9, 0.8], future=103), rows("b", [0.5, 0.3, 0.1], future=103)])
    w, curve = combine.fit_pool_weight(r, "a", "b", metric="brier_up", grid=[0.0, 0.5, 1.0])
    assert w == 1.0 and curve.is_monotonic_decreasing


def test_gate_uses_base_where_models_agree_and_fallback_elsewhere():
    days = ["2026-01-05", "2026-01-06", "2026-01-07"]
    r = pd.concat([
        rows("tfm", [0.9, 0.6, 0.2], t=days[0]), rows("dm", [0.8, 0.7, 0.3], t=days[0]),      # agree: up
        rows("tfm", [0.9, 0.6, 0.2], t=days[1]), rows("dm", [0.6, 0.4, 0.1], t=days[1]),      # disagree
        rows("tfm", [0.9, 0.6, 0.2], t=days[2]), rows("dm", [0.7, 0.5, 0.2], t=days[2]),      # dm neutral
        *[rows("rw", [1.0, 0.5, 0.0], t=d) for d in days],
    ])
    out = combine.agreement_gate(r, "tfm", "dm", "rw", "gated")
    p_up = out[out["sigma"] == 0].set_index("t")["p_above"]
    assert p_up.tolist() == [0.6, 0.5, 0.5]
    assert out.groupby("t")["gate"].first().tolist() == [True, False, False]
    assert set(out["predictor"]) == {"gated"} and len(out) == 9
