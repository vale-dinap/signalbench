"""Prediction markets as forecasting questions: will a market resolve Yes?

A fixed lead time before a market's end date, each model is asked for the
probability that the market resolves Yes, and is scored against the market's own
price at that time (see docs/forward_test_plan.md). Markets are chosen per event:

- binary Yes/No markets that resolved (one outcome priced at 1),
- still open at the forecast time: a market that closed earlier would ask about a
  known result,
- at least `min_history` daily prices at or before the forecast time, the last of
  them no more than `fresh` before it, so that the market's price is current,
- of these, the highest-volume market, so that each event counts once.

Daily history over a market's whole life can skip days, so the week before the
forecast time is filled in from hourly prices; each day keeps its last price. A
question's series holds only prices at or before the forecast time, so no state
builder can show what followed.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Series, daily_last
from .forecast import ForecastTask, horizon_scale

LEAD = pd.Timedelta(days=7)
MIN_HISTORY = 30
RECENT = pd.Timedelta(days=7)    # window fetched at hourly resolution before the forecast time
FRESH = pd.Timedelta(days=1)     # the last price must be at most this old at the forecast time
MARKET = "market"   # the reference predictor: the market's own price as P(Yes)


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def resolution(market: dict) -> int | None:
    """1 if a binary market resolved Yes, 0 if No, None if it is not binary or not resolved."""
    try:
        outcomes, prices = json.loads(market["outcomes"]), [float(p) for p in json.loads(market["outcomePrices"])]
    except (KeyError, TypeError, ValueError):
        return None
    if sorted(outcomes) != ["No", "Yes"] or len(prices) != 2:
        return None
    yes = prices[outcomes.index("Yes")]
    return 1 if yes >= 0.99 else 0 if yes <= 0.01 else None


@dataclass(frozen=True)
class MarketQuestion:
    """One market asked `lead` before its end date; series ends at the forecast time."""
    key: str
    event: str
    question: str
    volume: float
    end: pd.Timestamp
    asked_at: pd.Timestamp
    outcome: int
    series: Series

    @property
    def price(self) -> float:
        """The market's Yes price at the forecast time."""
        return float(self.series.df["close"].iloc[-1])

    def task(self) -> ForecastTask:
        """P(Yes) as P(Yes price above 0.50 at the end date): one threshold, no sigma grid.

        The sigma is NaN, so direction-only elicitations (which answer P(above the
        current price)) fail instead of being read as P(Yes).
        """
        idx = len(self.series) - 1
        horizon = max(1, math.ceil((self.end - self.asked_at) / pd.Timedelta(days=1)))
        return ForecastTask(self.series, idx, horizon, np.array([0.5]), np.array([np.nan]),
                            horizon_scale(self.series, idx, horizon))


def select_markets(events: Iterable[dict], history: Callable[[str], pd.DataFrame],
                   recent: Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame], lead: pd.Timedelta = LEAD,
                   min_history: int = MIN_HISTORY, fresh: pd.Timedelta = FRESH) -> list[MarketQuestion]:
    """One MarketQuestion per event that has an eligible market (see the module docstring).

    history(yes_token_id) returns the market's daily (t, close) history;
    recent(yes_token_id, start, end) its hourly history between two times.
    """
    out = []
    for event in events:
        candidates = []
        for m in event.get("markets", []):
            outcome = resolution(m)
            if outcome is None or not m.get("endDate") or not m.get("closedTime") or not m.get("clobTokenIds"):
                continue
            end = _utc(m["endDate"])
            if _utc(m["closedTime"]) <= end - lead:
                continue
            candidates.append((float(m.get("volume") or 0), end, outcome, m))
        for volume, end, outcome, m in sorted(candidates, key=lambda c: -c[0]):
            outcomes = json.loads(m["outcomes"])
            token = json.loads(m["clobTokenIds"])[outcomes.index("Yes")]
            asked = end - lead
            daily = history(token)
            if len(daily_last(daily[daily["t"] <= asked])) + RECENT.days < min_history:
                continue   # too short even if the last week fills in; spares the hourly fetch
            known = daily_last(pd.concat([daily, recent(token, asked - RECENT, asked)]))
            known = known[known["t"] <= asked].reset_index(drop=True)
            if len(known) < min_history or asked - known["t"].iloc[-1] > fresh:
                continue
            description = (f"Prediction market: {m['question']}\n"
                           f"Resolves on {end:%Y-%m-%d}: the Yes price goes to 1 if the answer is Yes, to 0 if No.")
            series = Series(m["slug"], description, known, kind="probability", value_label="Yes price",
                            meta={"source": "polymarket", "event": event.get("title"), "yes_token_id": token})
            out.append(MarketQuestion(m["slug"], event.get("title", ""), m["question"], volume, end, asked,
                                      outcome, series))
            break
    return out


def ask_all(predictors: list, questions: list[MarketQuestion], path: str | Path,
            report: Callable[[str], None] | None = print) -> pd.DataFrame:
    """P(Yes) from every predictor for every question; resumes from `path` (JSON lines).

    Each line holds one answer with the predictor's description and the raw audit
    trail. A stored answer from a predictor whose description has changed raises.
    Failed calls are reported and retried on the next run. Returns one row per
    (predictor, key) with p_yes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[(rec["predictor"], rec["key"])] = rec
    for pred in predictors:
        meta = pred.describe()
        for (name, key), rec in done.items():
            if name == pred.name and rec["describe"] != meta:
                raise ValueError(f"{path.name}: {name} was run as {rec['describe']}, not {meta}; use a new path")
        todo = [q for q in questions if (pred.name, q.key) not in done]
        if report:
            report(f"{pred.name}: {len(questions) - len(todo)} answered before, {len(todo)} to ask")
        for q in todo:
            try:
                fc = pred.predict(q.task())
            except Exception as e:  # noqa: BLE001 -- one failed call must not stop a run; retried on resume
                if report:
                    report(f"  {q.key}: {type(e).__name__}: {str(e)[:200]}")
                continue
            rec = {"predictor": pred.name, "key": q.key, "p_yes": float(fc.p_above[0]), "describe": meta,
                   "raw": fc.raw}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            done[(pred.name, q.key)] = rec
    names = {p.name for p in predictors}
    keys = {q.key for q in questions}
    return pd.DataFrame([{"predictor": n, "key": k, "p_yes": r["p_yes"]} for (n, k), r in done.items()
                         if n in names and k in keys])


def per_question(answers: pd.DataFrame, questions: list[MarketQuestion]) -> pd.DataFrame:
    """Brier score of P(Yes) per (predictor, question), with the market price added as predictor MARKET.

    horizon is the lead time in days, the same for every question. Columns follow scoring.per_forecast (series, t, horizon, predictor), so
    scoring.skill_table(per, "brier", baseline=MARKET) gives skill against the market.
    """
    info = pd.DataFrame([{"key": q.key, "series": q.key, "t": q.asked_at, "horizon": (q.end - q.asked_at).days,
                          "outcome": q.outcome, "price": q.price} for q in questions])
    market = info.assign(predictor=MARKET, p_yes=info["price"])
    models = answers.merge(info, on="key")
    per = pd.concat([market, models], ignore_index=True)
    per["brier"] = (per["p_yes"] - per["outcome"]) ** 2
    return per[["series", "t", "horizon", "predictor", "p_yes", "outcome", "price", "brier"]]
