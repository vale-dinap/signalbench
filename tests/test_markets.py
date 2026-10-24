import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from sbench import markets
from sbench.elicitation import DirectionChoice, ThresholdNoul
from sbench.predictors import Forecast
from sbench.scoring import skill_table
from sbench.states import build_price_state

END = "2026-12-01T12:00:00Z"


def market(slug, volume, yes_price="1", closed="2026-12-01 13:00:00+00", end=END, outcomes=("Yes", "No")):
    no_price = str(1 - float(yes_price))
    return {"question": slug.replace("-", " ") + "?", "slug": slug, "volume": str(volume), "endDate": end,
            "closedTime": closed, "outcomes": json.dumps(list(outcomes)),
            "outcomePrices": json.dumps([yes_price, no_price]), "clobTokenIds": json.dumps([f"{slug}-y", f"{slug}-n"])}


EVENTS = [
    {"title": "Election", "markets": [
        market("closed-early", 1000, closed="2026-11-20 09:00:00+00"),
        market("short-history", 900),
        market("chosen", 500, yes_price="0"),
        market("smaller", 100),
        market("unresolved", 2000, yes_price="0.5"),
        market("three-way", 3000, outcomes=("A", "B")),
    ]},
    {"title": "Settled early", "markets": [market("only-early", 800, closed="2026-11-01 00:00:00+00")]},
    {"title": "Weather", "markets": [market("rain", 50, yes_price="0")]},
]


def history(token):
    days = 20 if token.startswith("short-history") else 90
    t = pd.date_range(end="2026-12-01", periods=days, freq="D")
    return pd.DataFrame({"t": t, "close": np.linspace(0.3, 0.1, days)})


@pytest.fixture
def questions():
    return markets.select_markets(EVENTS, history)


def test_one_open_resolved_market_with_enough_history_per_event(questions):
    assert [q.key for q in questions] == ["chosen", "rain"]
    q = questions[0]
    assert q.outcome == 0 and q.event == "Election" and q.asked_at == pd.Timestamp("2026-11-24 12:00")
    assert q.series.df["t"].max() <= q.asked_at and len(q.series) >= markets.MIN_HISTORY
    assert q.price == q.series.df["close"].iloc[-1]


def test_the_task_asks_about_0_50_and_shows_nothing_after_the_forecast_time(questions):
    task = questions[0].task()
    assert task.thresholds.tolist() == [0.5] and task.horizon == 8
    assert "0.500" in json.dumps(ThresholdNoul().questions(task))
    state = build_price_state(task)
    assert "Resolves on 2026-12-01" in state and "2026-11-25" not in state
    with pytest.raises(IndexError):    # direction-only formats cannot answer P(Yes)
        DirectionChoice().decode({"dir": {}}, task)


@dataclass
class Constant:
    p: float
    name: str = "const"
    fail: bool = False

    def predict(self, task):
        if self.fail:
            raise RuntimeError("server busy")
        return Forecast(np.array([self.p]), raw={"state": "s"})

    def describe(self):
        return {"type": "Constant", "p": self.p}


def test_answers_resume_and_refuse_a_changed_predictor(questions, tmp_path):
    path = tmp_path / "answers.jsonl"
    assert markets.ask_all([Constant(0.2, fail=True)], questions, path, report=None).empty
    first = markets.ask_all([Constant(0.2)], questions, path, report=None)
    assert first["p_yes"].tolist() == [0.2, 0.2]
    assert len(path.read_text().splitlines()) == 2
    markets.ask_all([Constant(0.2)], questions, path, report=None)
    assert len(path.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="use a new path"):
        markets.ask_all([Constant(0.9)], questions, path, report=None)


def test_skill_is_measured_against_the_market_price(questions, tmp_path):
    answers = markets.ask_all([Constant(0.0, "right"), Constant(1.0, "wrong")], questions, tmp_path / "a.jsonl",
                              report=None)
    per = markets.per_question(answers, questions)
    assert set(per["predictor"]) == {markets.MARKET, "right", "wrong"} and (per["horizon"] == 7).all()
    assert per.loc[per["predictor"] == markets.MARKET, "brier"].iloc[0] == pytest.approx(questions[0].price ** 2)
    sk = skill_table(per, "brier", baseline=markets.MARKET).set_index("predictor")["skill"]
    assert sk["right"] == 1.0 and sk["wrong"] < 0
