import json

import numpy as np
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


def test_snapshot_round_trip_is_exact(tmp_path):
    t = pd.to_datetime([1672531200000, 1672617600000], unit="ms")   # millisecond resolution, as from Binance
    raw = pd.DataFrame({"t": t, "close": [363.23202514648438, 0.1 + 0.2]})
    first = data.cached(tmp_path / "s.csv", lambda: raw)
    again = data.cached(tmp_path / "s.csv", lambda: pytest.fail("snapshot not reused"))
    assert first["t"].dtype == "datetime64[ns]"
    pd.testing.assert_frame_equal(first, again)
    assert again["close"].tolist() == raw["close"].tolist()


def test_close_times_resolve_to_utc_across_daylight_saving():
    labels = pd.Series(pd.to_datetime(["2025-03-07", "2025-03-10", "2025-07-15"]))
    assert data.US_MARKET_CLOSE.resolve(labels).tolist() == [
        pd.Timestamp("2025-03-07 21:00"), pd.Timestamp("2025-03-10 20:00"), pd.Timestamp("2025-07-15 20:00")]
    assert data.ECB_REFERENCE_FIX.resolve(labels).iloc[[0, 2]].tolist() == [
        pd.Timestamp("2025-03-07 13:15"), pd.Timestamp("2025-07-15 12:15")]
    assert data.BINANCE_DAILY_CLOSE.resolve(labels).iloc[0] == pd.Timestamp("2025-03-08 00:00")


def test_series_without_close_time_is_known_at_its_labels(gbm):
    np.testing.assert_array_equal(gbm.available_at(), gbm.df["t"].to_numpy())
    assert data.known_rows(gbm, gbm.df["t"].iloc[9]) == 10


def test_ecb_rates_cross_through_the_euro(monkeypatch):
    csv = ("KEY,CURRENCY,TIME_PERIOD,OBS_VALUE\n"
           "a,JPY,2026-09-29,178.41\na,USD,2026-09-29,1.1355\n"
           "a,JPY,2026-09-30,178.27\na,USD,2026-09-30,1.1355\na,USD,2026-10-01,1.14\n")
    seen = {}

    def get(url, params, timeout):   # like the API: only the currencies in the URL
        seen["url"] = url
        wanted = url.split("/D.")[1].split(".EUR")[0].split("+")
        lines = csv.splitlines()
        text = "\n".join(lines[:1] + [ln for ln in lines[1:] if ln.split(",")[1] in wanted])
        return type("R", (), {"text": text, "raise_for_status": lambda self: None})()

    monkeypatch.setattr(data.requests, "get", get)
    usdjpy = data.load_ecb_rate("USD", "JPY")
    assert seen["url"].endswith("/D.JPY+USD.EUR.SP00.A")
    assert len(usdjpy) == 2   # 2026-10-01 has no JPY fix
    assert usdjpy["close"].iloc[0] == pytest.approx(178.41 / 1.1355)
    assert data.load_ecb_rate("EUR", "USD")["close"].iloc[-1] == pytest.approx(1.14)
