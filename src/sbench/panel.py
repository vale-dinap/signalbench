"""The eight-instrument panel of notebooks 03 onward, loaded from committed snapshots.

Crypto from Binance and US-listed ETFs/shares from Yahoo (snapshots of 02), FX
from ECB reference rates (snapshots of 03), each with the time its close became
known. Loading never touches the network.
"""
from __future__ import annotations

from pathlib import Path

from .data import BINANCE_DAILY_CLOSE, ECB_REFERENCE_FIX, US_MARKET_CLOSE, Series, cached, truncate

PANEL = [
    # key, asset class, snapshot (relative to data/snapshots), description shown to models, unit, close time
    ("BTC", "crypto", "02_real_assets/BTC.csv", "Bitcoin (BTC) priced in USDT on Binance: daily close price",
     "USDT", BINANCE_DAILY_CLOSE),
    ("ETH", "crypto", "02_real_assets/ETH.csv", "Ether (ETH) priced in USDT on Binance: daily close price",
     "USDT", BINANCE_DAILY_CLOSE),
    ("SPY", "equity", "02_real_assets/SPY.csv", "SPDR S&P 500 ETF Trust (SPY): daily close price",
     "USD", US_MARKET_CLOSE),
    ("AAPL", "equity", "02_real_assets/AAPL.csv", "Apple Inc. (AAPL) shares: daily close price",
     "USD", US_MARKET_CLOSE),
    ("GLD", "commodity", "02_real_assets/GLD.csv", "SPDR Gold Shares ETF (GLD): daily close price",
     "USD", US_MARKET_CLOSE),
    ("USO", "commodity", "02_real_assets/USO.csv", "United States Oil Fund ETF (USO): daily close price",
     "USD", US_MARKET_CLOSE),
    ("EURUSD", "fx", "03_enrichment/EURUSD_ECB.csv", "Euro / US dollar ECB reference rate (EUR/USD): daily fixing",
     "USD", ECB_REFERENCE_FIX),
    ("USDJPY", "fx", "03_enrichment/USDJPY_ECB.csv",
     "US dollar / Japanese yen from ECB reference rates (USD/JPY): daily fixing", "JPY", ECB_REFERENCE_FIX),
]
ASSET_CLASS = {key: cls for key, cls, *_ in PANEL}


def load_panel(snapshots: str | Path, start: str = "2023-01-01", end: str = "2026-09-30") -> list[Series]:
    """The panel's Series from the committed snapshots under `snapshots` (data/snapshots)."""
    snapshots = Path(snapshots)
    return [Series(key, description, truncate(cached(snapshots / path, None), start, end), unit=unit,
                   close_time=close, meta={"asset_class": cls})
            for key, cls, path, description, unit, close in PANEL]
