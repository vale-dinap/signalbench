import numpy as np
import pytest

from sbench.hypotheses import paired_bootstrap, sd_difference, slope_per_sd, verdict


@pytest.mark.parametrize(("estimate", "lo", "hi", "side", "expected"), [
    (-0.05, -0.08, -0.02, "below", "holds"),
    (-0.05, -0.08, 0.01, "below", "inconclusive"),
    (0.01, -0.02, 0.04, "below", "fails"),
    (0.02, 0.001, 0.04, "above", "holds"),
    (-0.01, -0.03, 0.01, "above", "fails"),
    (0.0, -0.01, 0.01, "above", "inconclusive"),
])
def test_verdict_follows_the_plan(estimate, lo, hi, side, expected):
    assert verdict(estimate, lo, hi, side) == expected


def test_verdict_rejects_an_unknown_side():
    with pytest.raises(ValueError):
        verdict(0.1, 0.0, 0.2, "up")


def test_slope_per_sd_is_scale_free():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 0.3 * x + rng.normal(size=500)
    assert slope_per_sd(x, y) == pytest.approx(slope_per_sd(10 * x, y))
    assert np.isnan(slope_per_sd(np.ones(5), np.arange(5)))   # no spread in x


def test_bootstrap_covers_the_effect_and_resamples_pairs():
    rng = np.random.default_rng(1)
    x = rng.normal(size=800)
    up = (rng.random(800) < 0.5 - 0.05 * x).astype(float)    # up-rate falls with x
    r = paired_bootstrap(slope_per_sd, x, up)
    assert r["hi"] < 0 and r["lo"] < r["estimate"] < r["hi"] and r["N"] == 800
    p = rng.normal(0.5, 0.05, 800)
    wide = paired_bootstrap(sd_difference, p + rng.normal(0, 0.1, 800), p)
    assert wide["lo"] > 0


def test_bootstrap_drops_rows_with_missing_values_and_needs_three():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert paired_bootstrap(slope_per_sd, x, x, n=50)["N"] == 4
    assert np.isnan(paired_bootstrap(slope_per_sd, x[:2], x[:2], n=50)["estimate"])
