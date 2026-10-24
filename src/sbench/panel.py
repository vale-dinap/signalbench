"""The eight-instrument panel of notebooks 03 onward.

Crypto from Binance, US-listed ETFs/shares from Yahoo, FX from ECB reference
rates, each with the time its close became known. load_panel reads the committed
snapshots of 02 and 03 and never touches the network; download_panel fetches the
same instruments from the same sources for a new window (the forward test).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import (
    BINANCE_DAILY_CLOSE,
    ECB_REFERENCE_FIX,
    US_MARKET_CLOSE,
    Series,
    cached,
    load_binance_klines,
    load_ecb_rate,
    load_yahoo,
    truncate,
)

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
SOURCES = {   # loader and arguments of each instrument, as in 02 and 03
    "BTC": (load_binance_klines, ("BTCUSDT",)), "ETH": (load_binance_klines, ("ETHUSDT",)),
    "SPY": (load_yahoo, ("SPY",)), "AAPL": (load_yahoo, ("AAPL",)),
    "GLD": (load_yahoo, ("GLD",)), "USO": (load_yahoo, ("USO",)),
    "EURUSD": (load_ecb_rate, ("EUR", "USD")), "USDJPY": (load_ecb_rate, ("USD", "JPY")),
}


def load_panel(snapshots: str | Path, start: str = "2023-01-01", end: str = "2026-09-30") -> list[Series]:
    """The panel's Series from the committed snapshots under `snapshots` (data/snapshots)."""
    snapshots = Path(snapshots)
    return [Series(key, description, truncate(cached(snapshots / path, None), start, end), unit=unit,
                   close_time=close, meta={"asset_class": cls})
            for key, cls, path, description, unit, close in PANEL]


def download_panel(directory: str | Path, start: str, end: str) -> list[Series]:
    """The panel over [start, end] from fresh downloads, saved as <directory>/<key>.csv and reused if present."""
    directory = Path(directory)
    out = []
    for key, cls, _, description, unit, close in PANEL:
        loader, args = SOURCES[key]
        stop = pd.Timestamp(end) + pd.Timedelta(days=1) if loader is load_yahoo else end   # Yahoo's end is exclusive
        df = cached(directory / f"{key}.csv", loader, *args, start=start, end=stop)
        out.append(Series(key, description, truncate(df, start, end), unit=unit, close_time=close,
                          meta={"asset_class": cls}))
    return out
