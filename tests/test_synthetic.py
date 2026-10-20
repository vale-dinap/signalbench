import numpy as np
import pytest

from sbench.forecast import make_task
from sbench.synthetic import (
    JumpDiffusion,
    MeanReversion,
    OraclePredictor,
    RandomWalk,
    RegimeVolatility,
    synthetic_series,
)

PROCESSES = [RandomWalk(), RandomWalk(mu=0.004), MeanReversion(), RegimeVolatility(),
             JumpDiffusion(mu=0.002, jump_mean=-0.04, jump_std=0.03)]


@pytest.mark.parametrize("process", PROCESSES, ids=lambda p: type(p).__name__)
def test_oracle_is_calibrated(process):
    """PIT values of realised outcomes under the true distribution must be uniform."""
    n, h = 9000, 3
    y, latent = process.simulate(n, np.log(100.0), np.random.default_rng(7))
    pit = []
    for i in range(50, n - h, h):                      # non-overlapping horizons
        mix = process.conditional(y[:i + 1], None if latent is None else latent[:i + 1], h)
        pit.append(1 - mix.sf(y[i + h])[0])
    pit = np.array(pit)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert abs((pit <= q).mean() - q) < 0.035, q


def test_mixture_quantiles_invert_the_cdf():
    mix = JumpDiffusion(jump_mean=-0.04, jump_std=0.03).conditional(np.array([0.0]), None, 5)
    levels = np.array([0.05, 0.5, 0.95])
    np.testing.assert_allclose(1 - mix.sf(mix.ppf(levels)), levels, atol=1e-6)


def test_regime_mixture_weights_sum_to_one():
    mix = RegimeVolatility().conditional(np.array([0.0]), np.array([1]), 10)
    assert len(mix.weights) == 11 and np.isclose(mix.weights.sum(), 1)


def test_oracle_uses_no_future_data():
    a = synthetic_series("a", RegimeVolatility(), n=300, seed=3)
    b = synthetic_series("b", RegimeVolatility(), n=300, seed=3)
    b.df.loc[201:, "close"] *= 1.5
    b.meta["latent"] = b.meta["latent"].copy()
    b.meta["latent"][201:] = 1 - b.meta["latent"][201:]
    pa = OraclePredictor().predict(make_task(a, 200, 5)).p_above
    pb = OraclePredictor().predict(make_task(b, 200, 5)).p_above
    np.testing.assert_array_equal(pa, pb)


def test_oracle_on_trend_favours_up():
    s = synthetic_series("trend", RandomWalk(mu=0.004, sigma=0.015), seed=1)
    task = make_task(s, 200, 5)
    p = OraclePredictor().predict(task).p_above
    assert p[task.i0] > 0.65 and np.all(np.diff(p) < 0)


def test_oracle_rejects_real_series(gbm):
    with pytest.raises(ValueError, match="not a synthetic"):
        OraclePredictor().predict(make_task(gbm, 100, 1))
