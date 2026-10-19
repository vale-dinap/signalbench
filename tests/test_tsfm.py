import numpy as np
import pytest

from sbench import tsfm
from sbench.data import Series
from sbench.forecast import make_task
from sbench.tsfm import QuantilePredictor, TimesFM3, TimesFM25


class StubTimesFM3:
    """Records predict() calls; returns quantiles growing with the horizon step."""

    def __init__(self):
        self.calls = []

    def predict(self, context, horizon, past_only_covariates=None, return_quantiles=False):
        self.calls.append({"context": context, "horizon": horizon, "cov": past_only_covariates})
        q = context[-1] + np.arange(1, horizon + 1)[:, None] * np.linspace(-1, 1, 9)[None, :]
        return type("Out", (), {"quantiles": q})()


@pytest.fixture
def stub(monkeypatch):
    s = StubTimesFM3()
    monkeypatch.setitem(tsfm._MODELS, ("timesfm3", "stub", None), s)
    return s


def test_timesfm3_returns_quantiles_at_the_horizon(stub):
    q = TimesFM3("stub").forecast(np.array([1.0, 2.0, 3.0]), horizon=4)
    np.testing.assert_allclose(q, 3.0 + 4 * np.linspace(-1, 1, 9))
    assert stub.calls[0]["cov"] is None


def test_timesfm3_stacks_covariates_and_drops_empty_ones(stub, gbm):
    late = Series("late", "Starts after the context", gbm.df.iloc[200:])
    other = Series("other", "Other", gbm.df.assign(close=gbm.df["close"] * 2))
    task = make_task(gbm, 150, 1, related=(other, late))
    QuantilePredictor(TimesFM3("stub"), max_context=64, log_space=True, use_related=True).predict(task)
    cov = stub.calls[0]["cov"]
    assert cov.shape == (1, 64)      # `late` has no values before the decision time
    np.testing.assert_allclose(cov[0, -1], np.log(2 * task.current))


def test_timesfm3_describe_omits_device():
    d = TimesFM3(device="cpu").describe()
    assert "device" not in d and d["type"] == "TimesFM3" and "timesfm" in d


def test_timesfm25_rejects_covariates():
    with pytest.raises(ValueError, match="covariates"):
        TimesFM25().forecast(np.ones(10), 1, {"x": np.ones(10)})


# ── Real models (skipped unless timesfm is installed; first run downloads weights) ──

@pytest.mark.parametrize("backend", ["TimesFM3", "TimesFM25"])
def test_real_model_gives_coherent_forecast(backend, gbm):
    pytest.importorskip("timesfm3" if backend == "TimesFM3" else "timesfm")
    task = make_task(gbm, 250, 3)
    p = QuantilePredictor(getattr(tsfm, backend)(), log_space=True).predict(task).p_above
    assert p.shape == task.thresholds.shape
    assert np.all((p >= 0) & (p <= 1)) and np.all(np.diff(p) <= 1e-9)
    assert 0.1 < p[task.i0] < 0.9      # no strong view on a driftless random walk


def test_real_timesfm3_uses_covariates(gbm):
    pytest.importorskip("timesfm3")
    other = Series("other", "Other", gbm.df.assign(close=gbm.df["close"][::-1].to_numpy()))
    task = make_task(gbm, 250, 3, related=(other,))
    base = QuantilePredictor(TimesFM3(), log_space=True).predict(task)
    with_cov = QuantilePredictor(TimesFM3(), log_space=True, use_related=True).predict(task)
    assert not np.allclose(base.raw["quantiles"], with_cov.raw["quantiles"])
