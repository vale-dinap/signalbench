"""Weekly backup of the forward test's inputs, run by .github/workflows/forward-data.yml.

The evaluation (notebooks/05_forward_test) downloads its data once the period has
passed, as docs/forward_test_plan.md states. This script saves the same inputs while
the period runs, with the same loaders, so that a source that breaks or changes by
then can be replaced by a snapshot taken at the time (a deviation, reported as such),
and so that the commits show when each input was captured. It does not change what
the evaluation uses.

    python scripts/collect_forward.py OUT_DIR [--sources prices trends news polymarket]

Each source is collected through yesterday (UTC); a source that fails is reported
in OUT_DIR/manifest.json and does not stop the others. Polymarket files use the
layout notebook 05 reads (events.json, history/, recent/).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from sbench import markets, news
from sbench.data import cached, fetch_polymarket_closed_events, load_google_trends, load_polymarket_history
from sbench.panel import download_panel

START, END = "2026-11-01", "2027-10-31"   # the forward period, through the 12-month update
PRICES_START = "2026-09-01"               # overlaps the committed snapshots of 02 and 03, which end on 30 Sep 2026
NEWS_START = "2026-10-01"                 # as in 05: 30 days of news baseline before the first decision point
TRENDS_START = "2026-09-01"               # as in 05
TRENDS_TERMS = {"BTC": ("btc", 0), "ETH": ("eth", 0), "SPY": ("s&p 500", 0), "AAPL": ("aapl", 0),
                "GLD": ("gold price", 0), "USO": ("crude oil", 0), "EURUSD": ("eur usd", 0), "USDJPY": ("yen", 7)}
SOURCES = ("prices", "trends", "news", "polymarket")


def prices(out: Path, through: pd.Timestamp) -> dict:
    """The eight instruments, downloaded afresh each run (the files are overwritten; git keeps each version)."""
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.csv"):
        f.unlink()
    series = download_panel(out, PRICES_START, f"{through:%Y-%m-%d}")
    return {s.key: f"{len(s)} rows to {s.df['t'].max():%Y-%m-%d}" for s in series}


def trends(out: Path, through: pd.Timestamp) -> dict:
    """Search interest per term; stops at the first refusal, as Google tends to block the rest too."""
    out.mkdir(parents=True, exist_ok=True)
    done = {}
    for key, (term, cat) in TRENDS_TERMS.items():
        df = load_google_trends(term, TRENDS_START, through, cat=cat, pause=30, waits=(60,))
        df.to_csv(out / f"trends_{key}.csv", index=False)
        done[key] = f"{len(df)} rows to {df['t'].max():%Y-%m-%d}"
    return done


def news_(out: Path, through: pd.Timestamp) -> dict:
    """Topic-matched GDELT articles, one file per day; days already collected are skipped."""
    slots = news.collect(NEWS_START, through, out, workers=4, report=None)
    missing = int((slots["file"] == "").sum())
    return {"days": int(slots["hour"].dt.normalize().nunique()), "hours without a file": missing,
            "articles matched": int(slots["matched"].sum())}


def polymarket(out: Path, through: pd.Timestamp) -> dict:
    """Closed events of the period so far, and the price histories notebook 05 would fetch for them.

    The 200 highest-volume events of the whole period that have closed so far are among the
    200 highest-volume of those closed so far, so n=200 here loses none of them.
    """
    if through < pd.Timestamp(START):
        return {"events": 0}
    out.mkdir(parents=True, exist_ok=True)
    events = fetch_polymarket_closed_events(START, through)
    (out / "events.json").write_text(json.dumps(events), encoding="utf-8")
    questions = markets.select_markets(
        events,
        history=lambda tok: cached(out / "history" / f"{tok}.csv", load_polymarket_history, tok),
        recent=lambda tok, a, b: cached(out / "recent" / f"{tok}_{a:%Y%m%d%H}.csv", load_polymarket_history, tok, 60,
                                        start=a, end=b))
    return {"events": len(events), "markets selected": len(questions)}


COLLECT = {"prices": prices, "trends": trends, "news": news_, "polymarket": polymarket}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    through = min(now.normalize() - pd.Timedelta(days=1), pd.Timestamp(END))
    manifest = {"run_at_utc": f"{now:%Y-%m-%d %H:%M:%S}", "through": f"{through:%Y-%m-%d}", "sources": {}}
    for name in args.sources:
        try:
            result = {"status": "ok", **COLLECT[name](args.out_dir / name, through)}
        except Exception as e:  # noqa: BLE001 -- one failing source must not stop the others
            traceback.print_exc()
            result = {"status": "failed", "error": f"{type(e).__name__}: {' '.join(str(e).split())[:300]}"}
        manifest["sources"][name] = result
        print(f"{name}: {result}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if any(r["status"] == "ok" for r in manifest["sources"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
