import numpy as np
import pandas as pd
import pytest

from sbench import headlines
from sbench.client import Response


@pytest.fixture
def fake_ask(monkeypatch):
    """'up' headlines are rated good news, 'down' bad news, others no effect."""
    calls = []

    def ask(model, state, questions, timeout, url):
        calls.append((model, state, questions))
        lines = [ln.split(". ", 1)[1] for ln in state.splitlines()[1:]]
        answers = {}
        for i, t in enumerate(lines, 1):
            p = [0.0, 0.0, 0.2, 0.2, 0.6] if "up" in t else [0.6, 0.2, 0.2, 0.0, 0.0] if "down" in t \
                else [0.0, 0.0, 1.0, 0.0, 0.0]
            answers[f"h{i}"] = {"score": 2, "probabilities": {str(k): v for k, v in enumerate(p)}}
        return Response(answers, 10, {})

    monkeypatch.setattr(headlines, "ask", ask)
    return calls


def test_reader_rates_each_headline_without_prices_or_dates(fake_ask):
    read = headlines.HeadlineReader("lev").read(["Oil up on supply cut", "Garden show opens"], "oil")
    model, state, questions = fake_ask[0]
    assert model == "lev"
    assert state == "News headlines:\n  1. Oil up on supply cut\n  2. Garden show opens"
    assert set(questions) == {"h1", "h2"} and questions["h2"]["criteria"] == headlines.LEVELS
    assert "price of oil" in questions["h1"]["instructions"]
    assert read["rating"].tolist() == pytest.approx([0.2 * 0.5 + 0.6, 0.0])


def test_news_score_is_the_mean_rating():
    s = headlines.news_score(pd.DataFrame({"rating": [0.7, -0.3, 0.0]}))
    assert s["score"] == pytest.approx(0.4 / 3) and s["intensity"] == pytest.approx(1.0 / 3)
    assert np.isnan(headlines.news_score(pd.DataFrame())["score"])


def test_read_all_stores_resumes_and_returns_only_the_requests(tmp_path, fake_ask):
    titles = {"2026-03-02": ["Oil up", "Oil down", "Oil up again"], "2026-03-03": ["Oil down"], "2026-03-04": []}
    reader = headlines.HeadlineReader("lev")
    path = tmp_path / "reads.jsonl"
    asked = [("USO", pd.Timestamp(d)) for d in titles]
    first = headlines.read_all(reader, asked[:2], lambda k, u: titles[f"{u:%Y-%m-%d}"], {"USO": "oil"}, path,
                               report=None)
    assert len(first) == 2 and first.loc[1, "score"] < 0 < first.loc[0, "score"]
    n = len(fake_ask)
    again = headlines.read_all(reader, asked, lambda k, u: titles[f"{u:%Y-%m-%d}"], {"USO": "oil"}, path,
                               report=None)
    assert len(fake_ask) == n   # stored reads reused; the empty request needs no call
    assert len(again) == 3 and again.loc[2, "n_headlines"] == 0 and np.isnan(again.loc[2, "score"])
    with pytest.raises(ValueError, match="new path"):
        headlines.read_all(headlines.HeadlineReader("clef"), asked, lambda k, u: [], {}, path, report=None)
