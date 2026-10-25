import numpy as np
import pandas as pd
import pytest

from sbench.scoring import rank_ic, skill_table


def panel(signal_strength, seed=0, dates=200, instruments=4):
    rng = np.random.default_rng(seed)
    t = np.repeat(pd.date_range("2026-01-01", periods=dates), instruments)
    common = np.repeat(rng.normal(size=dates), instruments)          # a move shared by all instruments on a date
    x = rng.normal(size=len(t))
    y = signal_strength * x + common + rng.normal(size=len(t))
    return pd.DataFrame({"t": t, "x": x, "y": y})


def test_rank_ic_of_a_monotone_signal_is_one():
    d = panel(0)
    d["y"] = np.exp(d["x"])
    r = rank_ic(d, "x", "y", n=200)
    assert r["ic"] == pytest.approx(1.0) and r["lo"] == pytest.approx(1.0) and r["dates"] == 200


def test_rank_ic_interval_covers_zero_without_signal_and_excludes_it_with_one():
    none, some = rank_ic(panel(0), "x", "y", n=500), rank_ic(panel(0.5), "x", "y", n=500)
    assert none["lo"] < 0 < none["hi"]
    assert some["lo"] > 0 and some["N"] == 800


def test_rank_ic_ignores_rows_with_missing_values():
    d = panel(0.5)
    d.loc[:9, "x"] = np.nan
    assert rank_ic(d, "x", "y", n=50)["N"] == 790


def test_rank_ic_without_enough_rows_is_nan():
    r = rank_ic(panel(0.5).iloc[:2], "x", "y", n=10)
    assert np.isnan(r["ic"]) and r["N"] == 2


def scores(predictors, seed=0, points=300):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-01", periods=points)
    return pd.concat([pd.DataFrame({"series": "A", "t": t, "horizon": 5, "predictor": p,
                                    "crps_n": rng.gamma(2.0, 0.5, points)}) for p in predictors], ignore_index=True)


def test_skill_table_interval_depends_only_on_its_own_pairs():
    per = scores(["random_walk", "model", "other", "third"])
    alone = skill_table(per[per["predictor"].isin(["random_walk", "model"])]).set_index("predictor").loc["model"]
    shuffled = per.sample(frac=1, random_state=1)                      # other predictors present, rows reordered
    together = skill_table(shuffled).set_index("predictor").loc["model"]
    assert together[["skill", "lo", "hi"]].tolist() == pytest.approx(alone[["skill", "lo", "hi"]].tolist())
