from pathlib import Path

import pandas as pd

from sbench.panel import ASSET_CLASS, load_panel

SNAPSHOTS = Path(__file__).resolve().parents[1] / "data" / "snapshots"


def test_panel_loads_the_committed_snapshots_with_close_times():
    series = {s.key: s for s in load_panel(SNAPSHOTS)}
    assert list(series) == list(ASSET_CLASS)
    assert pd.Timestamp(series["BTC"].available_at()[0]) == pd.Timestamp("2023-01-02 00:00")    # Binance: midnight UTC
    assert pd.Timestamp(series["SPY"].available_at()[0]) == pd.Timestamp("2023-01-03 21:00")    # 16:00 New York
    assert pd.Timestamp(series["EURUSD"].available_at()[0]) == pd.Timestamp("2023-01-02 13:15")  # 14:15 Frankfurt
    assert all(s.df["t"].max() <= pd.Timestamp("2026-09-30") for s in series.values())


def test_download_panel_uses_each_source_and_reuses_saved_files(tmp_path, monkeypatch):
    from sbench import panel
    calls = []

    def fake(name):
        def load(*args, start, end):
            calls.append((name, args, str(start), str(end)))
            return pd.DataFrame({"t": pd.date_range("2026-10-25", "2026-11-05"), "close": 1.0})
        return load

    monkeypatch.setattr(panel, "SOURCES", {k: (fake(f.__name__), a) for k, (f, a) in panel.SOURCES.items()})
    series = panel.download_panel(tmp_path, "2026-11-01", "2026-11-03")
    assert [s.key for s in series] == list(ASSET_CLASS) and all(len(s) == 3 for s in series)
    assert ("load_binance_klines", ("BTCUSDT",), "2026-11-01", "2026-11-03") in calls
    assert (tmp_path / "EURUSD.csv").exists() and len(calls) == 8
    panel.download_panel(tmp_path, "2026-11-01", "2026-11-03")
    assert len(calls) == 8
