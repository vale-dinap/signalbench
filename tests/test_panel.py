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
