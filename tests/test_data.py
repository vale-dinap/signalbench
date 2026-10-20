import json

import pandas as pd
import pytest

from sbench import data


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def market(question, closed, volume, end="2024-11-05T12:00:00Z", outcomes=("Yes", "No")):
    return {"question": question, "slug": question.lower().replace(" ", "-"), "closed": closed,
            "volume": str(volume), "endDate": end, "outcomes": json.dumps(list(outcomes)),
            "clobTokenIds": json.dumps([f"{question}-yes", f"{question}-no"])}


EVENTS = [
    {"title": "Election", "markets": [
        market("Resolved big", True, 900), market("Still open", False, 500),
        market("Three way", True, 800, outcomes=("A", "B", "C")),
    ]},
    {"title": "Other", "markets": [market("Resolved small", True, 100, end=None)]},
]


@pytest.fixture
def fake_events(monkeypatch):
    calls = []

    def get(url, params, timeout):
        calls.append(params)
        return FakeResponse(EVENTS)

    monkeypatch.setattr(data.requests, "get", get)
    return calls


def test_closed_markets_are_binary_resolved_and_sorted_by_volume(fake_events):
    out = data.fetch_polymarket_binary_markets(closed=True, order="volume", end_date_min="2024-01-01")
    assert [m["question"] for m in out] == ["Resolved big", "Resolved small"]
    assert out[0]["yes_token_id"] == "Resolved big-yes"
    assert out[0]["end_date"] == pd.Timestamp("2024-11-05 12:00") and out[1]["end_date"] is None
    assert fake_events[0]["closed"] == "true" and "active" not in fake_events[0]
    assert fake_events[0]["end_date_min"] == "2024-01-01"


def test_active_markets_by_default(fake_events):
    out = data.fetch_polymarket_binary_markets()
    assert [m["question"] for m in out] == ["Still open"]
    assert fake_events[0]["closed"] == "false" and fake_events[0]["active"] == "true"
