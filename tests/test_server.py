"""Integration tests against a live decision-model server; skipped when it is unreachable.

    pytest -m server
"""
import numpy as np
import pytest
import requests

from sbench.client import DEFAULT_URL
from sbench.elicitation import BinDistribution, DirectionChoice, DirectionNoul, ThresholdNoul
from sbench.forecast import make_task
from sbench.predictors import DecisionModelPredictor

MODEL = "kev-9b"


def server_up() -> bool:
    try:
        requests.post(DEFAULT_URL, json={}, timeout=3)
        return True
    except requests.RequestException:
        return False


pytestmark = [pytest.mark.server,
              pytest.mark.skipif(not server_up(), reason="decision-model server not reachable")]


@pytest.mark.parametrize("elicitation", [ThresholdNoul(), BinDistribution("score"), BinDistribution("choice"),
                                         DirectionNoul(), DirectionChoice()])
def test_every_elicitation_against_server(gbm, elicitation):
    task = make_task(gbm, 120, 1)
    fc = DecisionModelPredictor(MODEL, elicitation=elicitation).predict(task)
    answered = fc.p_above[~np.isnan(fc.p_above)]
    assert fc.p_above.shape == task.thresholds.shape
    assert len(answered) >= 1 and ((answered >= 0) & (answered <= 1)).all()
    assert fc.input_tokens > 0
