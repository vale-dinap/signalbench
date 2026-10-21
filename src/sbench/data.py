"""Time-series container, loaders and caching.

Every loader returns a DataFrame with exactly two columns:
    t     : pandas Timestamp (tz-naive, UTC), sorted, unique
    close : float
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests

POLYMARKET_EVENTS_URL = "https://gamma-api.polymarket.com/events"
POLYMARKET_HISTORY_URL = "https://clob.polymarket.com/prices-history"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


@dataclass
class Series:
    """A single instrument's history plus the metadata needed to describe it.

    kind        : "price" (positive, unbounded; thresholds in log-return space)
                  or "probability" (bounded 0-1, e.g. a prediction-market Yes price;
                  thresholds in absolute-change space)
    description : what the series is, shown at the top of the state
                  (e.g. the market question, or "Bitcoin / USDT spot")
    value_label : how questions refer to the value ("Yes price", "close price")
    unit        : appended to formatted values ("USD", "" for probabilities)
    decimals    : display precision; None = choose from the price level
    """
    key: str
    description: str
    df: pd.DataFrame
    kind: Literal["price", "probability"] = "price"
    value_label: str = "close price"
    unit: str = ""
    decimals: int | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.df = normalize_series(self.df)
        self._precision = self.decimals if self.decimals is not None else self._auto_precision()

    def __len__(self) -> int:
        return len(self.df)

    @property
    def freq(self) -> pd.Timedelta:
        """Median spacing between observations."""
        return infer_freq(self.df)

    def precision(self) -> int:
        return self._precision

    def _auto_precision(self) -> int:
        """Display decimals from the price level (fixed at construction)."""
        if self.kind == "probability":
            return 3
        level = float(self.df["close"].abs().median()) if len(self.df) else 1.0
        if level >= 1000:
            return 0
        if level >= 10:
            return 2
        if level >= 1:
            return 4
        return 6

    def fmt(self, x: float) -> str:
        s = f"{x:,.{self.precision()}f}"
        return f"{s} {self.unit}".rstrip()

    def round(self, x):
        return np.round(x, self.precision())


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalize_series(df: pd.DataFrame, t_col: str = "t", value_col: str | None = None) -> pd.DataFrame:
    """Return a clean (t, close) frame: datetime, sorted, deduplicated, no NaNs."""
    if value_col is None:
        value_col = "close" if "close" in df.columns else next(c for c in df.columns if c != t_col)
    out = pd.DataFrame({
        "t": pd.to_datetime(df[t_col]),
        "close": pd.to_numeric(df[value_col], errors="coerce"),
    })
    if out["t"].dt.tz is not None:
        out["t"] = out["t"].dt.tz_convert("UTC").dt.tz_localize(None)
    out["t"] = out["t"].astype("datetime64[ns]")   # one resolution whatever the source (pandas 3 keeps s/ms/us)
    out = out.dropna().sort_values("t").drop_duplicates("t", keep="last")
    return out.reset_index(drop=True)


def _utc_naive(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def infer_freq(df: pd.DataFrame) -> pd.Timedelta:
    if len(df) < 2:
        return pd.Timedelta(days=1)
    return df["t"].diff().median()


def daily_last(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one observation per calendar day (the last one).

    Polymarket's daily history often ends with an extra intraday point on the same
    date as the previous bar; without this, a "1-day" horizon can be a few hours.
    """
    d = normalize_series(df)
    d["_day"] = d["t"].dt.normalize()
    d = d.groupby("_day", as_index=False).last()
    return d[["t", "close"]].reset_index(drop=True)


def truncate(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Keep start <= t <= end (either bound optional). Use to freeze a test window."""
    m = pd.Series(True, index=df.index)
    if start is not None:
        m &= df["t"] >= pd.Timestamp(start)
    if end is not None:
        m &= df["t"] <= pd.Timestamp(end)
    return df[m].reset_index(drop=True)


# ── Caching ────────────────────────────────────────────────────────────────────

def cached(path: str | Path, loader: Callable[..., pd.DataFrame], *args, refresh: bool = False, **kwargs) -> pd.DataFrame:
    """Load a (t, close) CSV snapshot if present, otherwise call loader and save it.

    Snapshots keep reruns reproducible: live endpoints keep moving the end date.
    """
    path = Path(path)
    if not path.exists() or refresh:
        df = normalize_series(loader(*args, **kwargs))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    # the default parser can be off in the last bit; round_trip reads back exactly what was written
    return normalize_series(pd.read_csv(path, float_precision="round_trip"))


# ── Polymarket ─────────────────────────────────────────────────────────────────

def is_binary_market(market: dict) -> bool:
    try:
        outcomes = json.loads(market.get("outcomes", "[]"))
        return len(outcomes) == 2 and set(outcomes) <= {"Yes", "No"}
    except (json.JSONDecodeError, TypeError):
        return False


def fetch_polymarket_binary_markets(
    limit: int = 50,
    order: str = "volume24hr",
    exclude_keywords: tuple[str, ...] = (),
    closed: bool = False,
    end_date_min=None,
) -> list[dict]:
    """Binary markets from the top `limit` events, sorted by total volume.

    closed=False returns active markets; closed=True returns resolved ones (order
    by "volume" then: 24-hour volume is zero after resolution). end_date_min keeps
    events ending on or after that date. end_date is tz-naive UTC, or None.
    """
    params = {"closed": str(closed).lower(), "limit": limit, "order": order, "ascending": "false"}
    if not closed:
        params["active"] = "true"
    if end_date_min is not None:
        params["end_date_min"] = pd.Timestamp(end_date_min).strftime("%Y-%m-%d")
    r = requests.get(POLYMARKET_EVENTS_URL, params=params, timeout=30)
    r.raise_for_status()

    markets = []
    for event in r.json():
        for m in event.get("markets", []):
            if not is_binary_market(m) or bool(m.get("closed", False)) != closed:
                continue
            q = m.get("question", "")
            if any(k.lower() in q.lower() for k in exclude_keywords):
                continue
            markets.append({
                "event_title": event.get("title"),
                "question": q,
                "slug": m.get("slug"),
                "volume": float(m.get("volume") or 0),
                "end_date": _utc_naive(m["endDate"]) if m.get("endDate") else None,
                "outcomes": m.get("outcomes"),
                "clobTokenIds": m.get("clobTokenIds"),
                "yes_token_id": json.loads(m["clobTokenIds"])[0] if m.get("clobTokenIds") else None,
            })
    markets.sort(key=lambda x: x["volume"], reverse=True)
    return markets


def load_polymarket_history(token_id: str, fidelity: int = 1440, interval: str = "all") -> pd.DataFrame:
    """Yes-token price history. fidelity is in minutes (1440 = daily)."""
    params = {"market": token_id, "interval": interval, "fidelity": fidelity}
    r = requests.get(POLYMARKET_HISTORY_URL, params=params, timeout=30)
    r.raise_for_status()
    hist = pd.DataFrame(r.json().get("history", []))
    if hist.empty:
        return pd.DataFrame(columns=["t", "close"])
    hist["t"] = pd.to_datetime(hist["t"], unit="s")
    return normalize_series(hist.rename(columns={"p": "close"}))


def polymarket_series(market: dict, key: str, cache_dir: str | Path | None = None,
                      fidelity: int = 1440, end=None, refresh: bool = False) -> Series:
    """Build a probability Series for a market dict from fetch_polymarket_binary_markets."""
    token = market.get("yes_token_id") or json.loads(market["clobTokenIds"])[0]
    if cache_dir is not None:
        df = cached(Path(cache_dir) / f"{key}.csv", load_polymarket_history, token, fidelity, refresh=refresh)
    else:
        df = load_polymarket_history(token, fidelity)
    if fidelity >= 1440:
        df = daily_last(df)
    df = truncate(df, end=end)
    return Series(key=key, description=f"Prediction market: {market['question']}", df=df,
                  kind="probability", value_label="Yes price", meta={"source": "polymarket", **market})


# ── Crypto (Binance public market data, no API key) ───────────────────────────

def load_binance_klines(symbol: str, interval: str = "1d", start=None, end=None) -> pd.DataFrame:
    """Close prices from Binance klines, paginated. symbol like 'BTCUSDT'."""
    start_ms = int(pd.Timestamp(start).timestamp() * 1000) if start is not None else 0
    end_ms = int(pd.Timestamp(end).timestamp() * 1000) if end is not None else None
    rows = []
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1000}
        if end_ms is not None:
            params["endTime"] = end_ms
        r = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        start_ms = batch[-1][0] + 1
    df = pd.DataFrame({"t": [b[0] for b in rows], "close": [float(b[4]) for b in rows]})
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return normalize_series(df)


# ── Stocks / ETFs / commodities / FX (Yahoo via yfinance, optional) ───────────

def load_yahoo(ticker: str, start=None, end=None, interval: str = "1d") -> pd.DataFrame:
    """Close prices via yfinance (pip install yfinance). Futures like 'GC=F', FX like 'EURUSD=X'."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("load_yahoo needs yfinance: pip install yfinance") from e
    raw = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=["t", "close"])
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # multi-index columns in recent yfinance
        close = close.iloc[:, 0]
    return normalize_series(close.rename("close").rename_axis("t").reset_index())


# ── Alignment ──────────────────────────────────────────────────────────────────

def align_to(main: pd.DataFrame, others: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align other series to main's timestamps using only past-or-equal values (as-of join).

    Returns a frame with column t plus one column per key. No look-ahead.
    """
    out = main[["t"]].copy()
    for key, df in others.items():
        o = normalize_series(df).rename(columns={"close": key})
        out = pd.merge_asof(out.sort_values("t"), o, on="t", direction="backward")
    return out.reset_index(drop=True)
