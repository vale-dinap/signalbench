import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_gbm
from sbench.backtest import COLUMNS, _fingerprint, decision_points, load_raw, run_forecast_backtest
from sbench.data import Series
from sbench.forecast import make_task
from sbench.predictors import ConstantPredictor, Forecast, RandomWalkPredictor
from sbench.scoring import per_forecast, skill_table, summary


class Flaky:
    """Fails on chosen indices, to test that failures are retried."""
    name = "flaky"

    def __init__(self, fail_on):
        self.fail_on = set(fail_on)

    def predict(self, task):
        if task.idx in self.fail_on:
            raise RuntimeError("boom")
        return Forecast(np.full(len(task.thresholds), 0.5), raw={"note": "ok"})

    def describe(self):
        return {"type": "Flaky"}


class UpOnly:
    """Direction-only predictor: answers just the current-value threshold."""
    name = "up_only"

    def predict(self, task):
        p = np.full(len(task.thresholds), np.nan)
        p[task.i0] = 0.6
        return Forecast(p)

    def describe(self):
        return {"type": "UpOnly"}


def test_csv_schema_and_values(tmp_path, gbm):
    idxs = decision_points(gbm, 1, max_points=5)
    res = run_forecast_backtest(gbm, [RandomWalkPredictor()], 1, tmp_path / "r.csv", idxs, verbose=False)
    assert list(pd.read_csv(tmp_path / "r.csv").columns) == COLUMNS
    i = idxs[0]
    first = res[res["t"] == gbm.df["t"].iloc[i]].sort_values("threshold")
    np.testing.assert_allclose(first["threshold"], make_task(gbm, i, 1).thresholds)
    assert (first["future"] == gbm.df["close"].iloc[i + 1]).all()
    assert (first["outcome"] == (first["future"] > first["threshold"])).all()


def test_resume_skips_done_and_retries_failures(tmp_path, gbm):
    idxs = decision_points(gbm, 1, max_points=10)
    path = tmp_path / "r.csv"
    run_forecast_backtest(gbm, [Flaky(fail_on=idxs[:3])], 1, path, idxs, verbose=False)
    assert pd.read_csv(path)["t"].nunique() == 7
    res = run_forecast_backtest(gbm, [Flaky(fail_on=[])], 1, path, idxs, verbose=False)
    assert res["t"].nunique() == 10
    assert not res.duplicated(["predictor", "t", "threshold"]).any()
    assert len(load_raw(path)) == 10


def test_multiple_series_share_a_csv(tmp_path):
    a, b = make_gbm(key="a", seed=1), make_gbm(key="b", seed=2)   # same timestamps
    path = tmp_path / "r.csv"
    idxs = decision_points(a, 1, max_points=5)
    run_forecast_backtest(a, [RandomWalkPredictor()], 1, path, idxs, verbose=False)
    res_b = run_forecast_backtest(b, [RandomWalkPredictor()], 1, path, idxs, verbose=False)
    assert res_b["t"].nunique() == 5
    assert pd.read_csv(path).groupby("series")["t"].nunique().to_dict() == {"a": 5, "b": 5}


def test_changed_configuration_is_rejected(tmp_path, gbm):
    path = tmp_path / "r.csv"
    idxs = decision_points(gbm, 1, max_points=3)
    run_forecast_backtest(gbm, [RandomWalkPredictor(window=30)], 1, path, idxs, verbose=False)
    with pytest.raises(ValueError, match="predictor 'random_walk'"):
        run_forecast_backtest(gbm, [RandomWalkPredictor(window=60)], 1, path, idxs, verbose=False)
    with pytest.raises(ValueError, match="sigmas"):
        run_forecast_backtest(gbm, [RandomWalkPredictor()], 1, path, idxs, sigmas=(-1, 0, 1), verbose=False)
    edited = Series(gbm.key, gbm.description, gbm.df.assign(close=gbm.df["close"] * 1.01))
    with pytest.raises(ValueError, match="series data differs"):
        run_forecast_backtest(edited, [RandomWalkPredictor()], 1, path, idxs, verbose=False)


def test_appended_data_keeps_run_valid(tmp_path):
    short, long = make_gbm(n=200), make_gbm(n=260)       # same seed: long extends short
    path = tmp_path / "r.csv"
    idxs = decision_points(short, 1, max_points=5)
    run_forecast_backtest(short, [RandomWalkPredictor()], 1, path, idxs, verbose=False)
    run_forecast_backtest(long, [RandomWalkPredictor()], 1, path, idxs, verbose=False)
    meta = json.loads((tmp_path / "r.meta.json").read_text())
    assert meta["gbm|h=1"]["n_rows"] == 200


def test_fingerprint_from_another_time_resolution_is_accepted_and_upgraded(tmp_path, gbm):
    path, meta_path = tmp_path / "r.csv", tmp_path / "r.meta.json"
    idxs = decision_points(gbm, 1, max_points=3)
    run_forecast_backtest(gbm, [RandomWalkPredictor()], 1, path, idxs, verbose=False)
    meta = json.loads(meta_path.read_text())
    current = meta["gbm|h=1"]["fingerprint"]
    meta["gbm|h=1"]["fingerprint"] = _fingerprint(gbm, len(gbm), unit="s")   # as stored before ns timestamps
    meta_path.write_text(json.dumps(meta))
    run_forecast_backtest(gbm, [RandomWalkPredictor()], 1, path, idxs, verbose=False)
    assert json.loads(meta_path.read_text())["gbm|h=1"]["fingerprint"] == current


def test_random_walk_is_calibrated_on_a_random_walk(tmp_path):
    s = make_gbm(n=600)
    res = run_forecast_backtest(s, [RandomWalkPredictor(), ConstantPredictor()], 1, tmp_path / "r.csv",
                                decision_points(s, 1), verbose=False)
    tab = summary(per_forecast(res), ci=False).loc[("gbm", 1)]
    assert 0.72 <= tab.loc["random_walk", "cover_80"] <= 0.88
    assert tab.loc["constant", "crps_n_skill"] < 0
    assert abs(tab.loc["random_walk", "brier_up"] - 0.25) < 0.01


def test_pooled_summary_pairs_by_series_and_time(tmp_path):
    a, b = make_gbm(key="a", seed=1), make_gbm(key="b", seed=2)   # same timestamps
    path, idxs = tmp_path / "r.csv", decision_points(a, 1, max_points=10)
    preds = [RandomWalkPredictor(), ConstantPredictor()]
    res = pd.concat([run_forecast_backtest(s, preds, 1, path, idxs, verbose=False) for s in (a, b)])
    per = per_forecast(res)
    pooled = summary(per, ci=False, by=("horizon",)).loc[1]
    assert pooled.loc["constant", "N"] == 20
    c, rw = (per[per["predictor"] == p]["crps_n"].mean() for p in ("constant", "random_walk"))
    assert np.isclose(pooled.loc["constant", "crps_n_skill"], 1 - c / rw, atol=1e-4)
    sk = skill_table(per).set_index("predictor").loc["constant"]
    assert np.isclose(sk["skill"], 1 - c / rw) and sk["lo"] <= sk["skill"] <= sk["hi"] and sk["N"] == 20


def test_direction_only_predictor_is_scored(tmp_path, gbm):
    res = run_forecast_backtest(gbm, [RandomWalkPredictor(), UpOnly()], 1, tmp_path / "r.csv",
                                decision_points(gbm, 1, max_points=20), verbose=False)
    per = per_forecast(res)
    up = per[per["predictor"] == "up_only"]
    assert up["brier_up"].notna().all() and up["crps_n"].isna().all()
    assert "brier_up_skill" in summary(per, ci=False).columns


def test_flat_outcomes_count_as_not_up(tmp_path, market):
    res = run_forecast_backtest(market, [RandomWalkPredictor()], 1, tmp_path / "r.csv",
                                decision_points(market, 1), verbose=False)
    per = per_forecast(res)
    assert per["flat"].any()
    assert not per.loc[per["flat"], "up"].any()
