import json
import re
from functools import partial

import numpy as np
import pytest

from sbench import predictors
from sbench.client import Response
from sbench.data import Series
from sbench.elicitation import BinDistribution, ThresholdNoul
from sbench.forecast import blind_task, make_task
from sbench.predictors import DecisionModelPredictor, EmpiricalPredictor, RandomWalkPredictor
from sbench.states import build_price_state
from sbench.tsfm import GaussianForecaster, QuantilePredictor


def fake_ask(model, state, questions, timeout=300, url=None):
    """Uniform answers of the right shape for every question type."""
    answers = {}
    for k, q in questions.items():
        if q["type"] == "noul":
            answers[k] = {"type": "noul", "noul": 0.6}
        elif q["type"] == "score":
            n = len(q["criteria"])
            answers[k] = {"type": "score", "probabilities": {str(i): 1 / n for i in range(n)}}
        else:
            n = len(q["criteria"])
            answers[k] = {"type": "choice", "probabilities": {c: 1 / n for c in q["criteria"]}}
    return Response(answers=answers, input_tokens=len(state), raw={})


@pytest.mark.parametrize("elicitation", [ThresholdNoul(), BinDistribution("score"), BinDistribution("choice")])
def test_decision_model_predictor_any_elicitation(monkeypatch, gbm, elicitation):
    monkeypatch.setattr(predictors, "ask", fake_ask)
    task = make_task(gbm, 100, 1)
    pred = DecisionModelPredictor("fake-model", elicitation=elicitation)
    fc = pred.predict(task)
    assert fc.p_above.shape == task.thresholds.shape
    assert fc.raw["state"] == build_price_state(task)
    assert pred.name == f"fake-model/{elicitation.name}"
    assert pred.describe()["elicitation"]["type"] == type(elicitation).__name__


def test_state_contains_no_future_values(gbm):
    task = make_task(gbm, 100, 1)
    state = build_price_state(task)
    assert gbm.df["t"].iloc[101].strftime("%Y-%m-%d") not in state
    assert gbm.fmt(gbm.df["close"].iloc[100]) in state


def test_gaussian_quantile_predictor_matches_random_walk(gbm):
    """The quantile -> threshold mapping must reproduce the analytic random walk."""
    task = make_task(gbm, 150, 1)
    rw = RandomWalkPredictor().predict(task).p_above
    q = QuantilePredictor(GaussianForecaster(), log_space=True).predict(task).p_above
    np.testing.assert_allclose(q, rw, atol=0.03)


def test_quantile_predictor_passes_aligned_covariates(gbm):
    seen = {}

    class Spy(GaussianForecaster):
        def forecast(self, context, horizon, covariates=None):
            seen["n"], seen["cov"] = len(context), covariates
            return super().forecast(context, horizon)

    other = Series("other", "Other", gbm.df.assign(close=gbm.df["close"] * 2))
    task = make_task(gbm, 100, 1, related=(other,))
    QuantilePredictor(Spy(), max_context=50, use_related=True).predict(task)
    assert seen["n"] == 50 and len(seen["cov"]["other"]) == 50
    np.testing.assert_allclose(seen["cov"]["other"][-1], 2 * task.current)


def test_empirical_returns_nan_without_enough_history(gbm):
    task = make_task(gbm, 5, 1)
    assert np.isnan(EmpiricalPredictor().predict(task).p_above).all()


def test_blind_task_rescales_and_keeps_answers_on_the_original_grid(gbm):
    task = make_task(gbm, 150, 5)
    b = blind_task(task, window=60)
    k = 100 / gbm.df["close"].iloc[91]
    assert len(b.series) == 60 and b.idx == 59 and b.series.df["close"].iloc[0] == pytest.approx(100)
    assert b.current == pytest.approx(task.current * k)
    np.testing.assert_allclose(b.thresholds, task.thresholds * k)
    np.testing.assert_array_equal(b.sigmas, task.sigmas)
    with pytest.raises(ValueError):
        blind_task(make_task(gbm, 150, 5, related=(gbm,)))


def test_blind_predictor_shows_no_name_dates_or_levels(monkeypatch, gbm):
    monkeypatch.setattr(predictors, "ask", fake_ask)
    task = make_task(gbm, 150, 5)
    pred = DecisionModelPredictor("fake-model", ThresholdNoul(), partial(build_price_state, lookback=60), blind=True)
    fc = pred.predict(task)
    text = fc.raw["state"] + json.dumps(fc.raw["questions"])
    assert pred.name == "fake-model/threshold_noul/blind" and fc.p_above.shape == task.thresholds.shape
    assert "Synthetic asset" not in text and "USD" not in text and "Decision time" not in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    assert "day -59: 100.00" in fc.raw["state"] and "day 0:" in fc.raw["state"]
    assert gbm.fmt(task.thresholds[0]).split()[0] not in text


def test_describe_is_unchanged_for_named_predictors():
    d = DecisionModelPredictor("m").describe()
    assert "blind" not in d and DecisionModelPredictor("m", blind=True).describe()["blind_window"] == 60
