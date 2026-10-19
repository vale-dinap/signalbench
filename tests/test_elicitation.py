import numpy as np
import pytest

from sbench.client import SCORE_MAX_LEVELS, score_question
from sbench.elicitation import BinDistribution, DirectionChoice, DirectionNoul, ThresholdNoul
from sbench.forecast import make_task


@pytest.fixture
def task(gbm):
    return make_task(gbm, 100, horizon=1)


def test_threshold_noul_roundtrip(task):
    e = ThresholdNoul()
    qs = e.questions(task)
    assert len(qs) == len(task.thresholds)
    assert task.series.fmt(task.thresholds[0]) in qs["q0"]["instructions"]
    answers = {k: {"type": "noul", "noul": 0.1 * i} for i, k in enumerate(qs)}
    np.testing.assert_allclose(e.decode(answers, task), 0.1 * np.arange(len(qs)))


@pytest.mark.parametrize("qtype", ["score", "choice"])
def test_bin_distribution_decodes_to_exceedance(task, qtype):
    e = BinDistribution(question_type=qtype)
    q = e.questions(task)["bins"]
    n = len(task.thresholds) + 1
    assert len(q["criteria"]) == n
    mass = np.arange(1, n + 1, dtype=float)
    mass /= mass.sum()
    keys = [str(i) for i in range(n)] if qtype == "score" else [f"b{i}" for i in range(n)]
    p = e.decode({"bins": {"probabilities": dict(zip(keys, mass))}}, task)
    np.testing.assert_allclose(p, [mass[k + 1:].sum() for k in range(n - 1)])
    assert np.all(np.diff(p) <= 0)


def test_bin_score_rejects_too_many_ranges(gbm):
    task = make_task(gbm, 100, 1, sigmas=np.linspace(-3, 3, 13))
    with pytest.raises(ValueError):
        BinDistribution("score").questions(task)
    BinDistribution("choice").questions(task)   # no cap


def test_score_question_level_limits():
    with pytest.raises(ValueError):
        score_question("x", ["a"])
    with pytest.raises(ValueError):
        score_question("x", [str(i) for i in range(SCORE_MAX_LEVELS + 1)])


def test_direction_modes_fill_only_current_threshold(task):
    p = DirectionNoul().decode({"up": {"noul": 0.7}}, task)
    assert p[task.i0] == 0.7 and np.isnan(np.delete(p, task.i0)).all()
    p = DirectionChoice(include_flat=True).decode(
        {"dir": {"probabilities": {"up": 0.3, "down": 0.5, "flat": 0.2}}}, task)
    assert p[task.i0] == pytest.approx(0.3)
    p = DirectionChoice(include_flat=False).decode({"dir": {"probabilities": {"up": 0.3, "down": 0.3}}}, task)
    assert p[task.i0] == pytest.approx(0.5)
