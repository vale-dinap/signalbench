"""News coverage from GDELT's Global Knowledge Graph (GKG 2.1) raw files.

GDELT publishes a GKG file every 15 minutes: one row per English-language
article it saw in those 15 minutes, with the page title, themes, organisations,
names and tone. The DOC API on top of it is throttled too hard for a multi-year
window, so the raw files are read directly: one file per hour (a quarter of all
coverage), keeping only articles whose title or tags match a topic.

collect() stores the matches as one gzipped CSV per day plus a per-day slot log
(which file was read, how many articles it had), and resumes where it stopped:
    python -m sbench.news 2023-12-01 2026-09-30 data/raw/gkg

A file stamped T holds articles seen in the 15 minutes before T and is
published a few minutes after T; treat its articles as known at T + 15 min.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import re
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

GKG_URL = "http://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
GKG_DELAY = pd.Timedelta(minutes=15)   # file stamp -> when its articles can be considered known

# GKG 2.1 column positions (27 tab-separated columns, no header)
_DATE, _DOMAIN, _URL, _THEMES, _ORGS, _TONE, _NAMES, _EXTRAS = 1, 3, 4, 7, 13, 15, 23, 26


@dataclass(frozen=True)
class Topic:
    """What makes an article relevant to an instrument.

    title  : regex searched in the page title, case-insensitive ("" = none)
    themes : GKG themes (V1THEMES), matched exactly
    names  : lower-case organisation or name strings (V1ORGANIZATIONS, V2.1ALLNAMES), matched exactly
    """
    title: str = ""
    themes: tuple[str, ...] = ()
    names: tuple[str, ...] = ()


# Deliberately broad: collection keeps every candidate, the notebook decides how strict to be.
TOPICS = {
    "BTC": Topic(r"\bbitcoin\b|\bbtc\b", ("ECON_BITCOIN",)),
    "ETH": Topic(r"\bethereum\b|\bether\b|\beth\b"),
    "SPY": Topic(r"s&p ?500|wall street|stock market|\bstocks\b|dow jones|nasdaq", ("ECON_STOCKMARKET",)),
    "AAPL": Topic(r"\bapple\b|\baapl\b|tim cook|\biphone", names=("apple", "apple inc")),
    "GLD": Topic(r"\bgold\b", ("ECON_GOLDPRICE",)),
    "USO": Topic(r"\boil\b|\bcrude\b|\bopec\b|\bbrent\b|\bwti\b", ("ECON_OILPRICE",), ("opec",)),
    "EURUSD": Topic(r"\beuro\b|eur/usd|\becb\b|european central bank|lagarde", ("ECON_WORLDCURRENCIES_EURO",),
                    ("european central bank",)),
    "USDJPY": Topic(r"\byen\b|usd/jpy|bank of japan|\bboj\b",
                    ("ECON_WORLDCURRENCIES_YEN", "ECON_WORLDCURRENCIES_JAPANESE_YEN"), ("bank of japan",)),
    "MACRO": Topic(r"tariff|federal reserve|\bfed\b|powell|interest rate|inflation|\bcpi\b|recession"),
    "POSTS": Topic(r"truth social", names=("truth social",)),   # high-impact posts, through their coverage
}

ARTICLE_COLUMNS = ["seen", "domain", "url", "title", "tone", "words", "title_topics", "tag_topics"]


def stamp(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).strftime("%Y%m%d%H%M%S")


def parse_gkg(raw: bytes) -> pd.DataFrame:
    """Unzipped GKG file -> seen, domain, url, title, tone, words, themes, names (one row per article).

    themes and names are ';'-joined (names lower-case, organisations and all names together).
    """
    df = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")), sep="\t", header=None, dtype=str,
                     quoting=csv.QUOTE_NONE, on_bad_lines="skip", keep_default_na=False,
                     usecols=[_DATE, _DOMAIN, _URL, _THEMES, _ORGS, _TONE, _NAMES, _EXTRAS])
    tone = df[_TONE].str.split(",")
    names = df[_NAMES].str.replace(r",\d+(?=;|$)", "", regex=True)   # drop character offsets
    return pd.DataFrame({
        "seen": pd.to_datetime(df[_DATE], format="%Y%m%d%H%M%S", errors="coerce").astype("datetime64[ns]"),
        "domain": df[_DOMAIN],
        "url": df[_URL],
        "title": df[_EXTRAS].str.extract(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", expand=False)
                            .fillna("").map(lambda s: html.unescape(s).strip()),
        "tone": pd.to_numeric(tone.str[0], errors="coerce"),
        "words": pd.to_numeric(tone.str[-1], errors="coerce"),
        "themes": df[_THEMES].str.strip(";"),
        "names": (df[_ORGS] + ";" + names).str.lower().str.strip(";"),
    })


def _exact(items: tuple[str, ...]) -> str:
    """Regex matching any of `items` as a whole ';'-separated entry."""
    return r"(?:^|;)(?:" + "|".join(re.escape(i) for i in items) + r")(?:;|$)"


def match_topics(articles: pd.DataFrame, topics: dict[str, Topic] = TOPICS) -> pd.DataFrame:
    """Articles matching at least one topic, with ';'-joined topic keys in title_topics and tag_topics."""
    by_title, by_tag = {}, {}
    for key, topic in topics.items():
        if topic.title:
            by_title[key] = articles["title"].str.contains(topic.title, case=False, regex=True)
        tag = pd.Series(False, index=articles.index)
        if topic.themes:
            tag |= articles["themes"].str.contains(_exact(topic.themes), regex=True)
        if topic.names:
            tag |= articles["names"].str.contains(_exact(topic.names), regex=True)
        by_tag[key] = tag

    def joined(flags: dict[str, pd.Series]) -> pd.Series:
        out = pd.Series("", index=articles.index, dtype=object)
        for key, flag in flags.items():
            out = out + flag.map({True: key + ";", False: ""})
        return out.str.rstrip(";")

    out = articles.assign(title_topics=joined(by_title), tag_topics=joined(by_tag))
    return out.loc[(out["title_topics"] != "") | (out["tag_topics"] != ""), ARTICLE_COLUMNS].reset_index(drop=True)


def fetch_gkg(t: pd.Timestamp, get: Callable = requests.get, attempts: int = 3) -> bytes | None:
    """Unzipped GKG file stamped t; None if GDELT has no file for that slot."""
    for attempt in range(attempts):
        try:
            r = get(GKG_URL.format(stamp=stamp(t)), timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                return z.read(z.namelist()[0])
        except (requests.RequestException, zipfile.BadZipFile):
            if attempt == attempts - 1:
                raise
            time.sleep(5 * 4 ** attempt)
    return None


def collect_day(day: pd.Timestamp, out_dir: str | Path, topics: dict[str, Topic] = TOPICS,
                get: Callable = requests.get) -> pd.DataFrame:
    """Read one GKG file per hour of `day` and write {day}.csv.gz (matches) and {day}.slots.csv (log).

    Each hour uses its first available quarter-hour file. The slot log is written last,
    so its presence marks the day as complete. Returns the slot log.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp(day).normalize()
    found, slots = [], []
    for hour in range(24):
        for minute in (0, 15, 30, 45):
            t = day + pd.Timedelta(hours=hour, minutes=minute)
            raw = fetch_gkg(t, get=get)
            if raw is not None:
                articles = parse_gkg(raw)
                found.append(match_topics(articles, topics))
                slots.append({"hour": day + pd.Timedelta(hours=hour), "file": stamp(t),
                              "articles": len(articles), "matched": len(found[-1])})
                break
        else:
            slots.append({"hour": day + pd.Timedelta(hours=hour), "file": "", "articles": 0, "matched": 0})
    name = day.strftime("%Y-%m-%d")
    matches = pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=ARTICLE_COLUMNS)
    matches.to_csv(out_dir / f"{name}.csv.gz", index=False)
    log = pd.DataFrame(slots)
    log.to_csv(out_dir / f"{name}.slots.csv", index=False)
    return log


def collect(start, end, out_dir: str | Path, workers: int = 4, topics: dict[str, Topic] = TOPICS,
            get: Callable = requests.get, report: Callable[[str], None] | None = print) -> pd.DataFrame:
    """collect_day for every day from start to end (inclusive) not yet collected; returns all slot logs."""
    out_dir = Path(out_dir)
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    todo = [d for d in days if not (out_dir / f"{d:%Y-%m-%d}.slots.csv").exists()]
    if report:
        report(f"{len(days) - len(todo)} of {len(days)} days already collected, {len(todo)} to go")
    began = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(collect_day, d, out_dir, topics, get): d for d in todo}
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if report and (i % 20 == 0 or i == len(todo)):
                left = (time.time() - began) / i * (len(todo) - i)
                report(f"{i}/{len(todo)} days, about {left / 60:.0f} min left")
    return load_slots(out_dir, start, end)


def _load(out_dir: str | Path, start, end, suffix: str, parse: str) -> pd.DataFrame:
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    paths = [p for d in days if (p := Path(out_dir) / f"{d:%Y-%m-%d}{suffix}").exists()]
    frames = [pd.read_csv(p, keep_default_na=False, na_values={"tone": [""], "words": [""]},
                          dtype={"file": str, "title_topics": str, "tag_topics": str}) for p in paths]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[parse])
    df[parse] = pd.to_datetime(df[parse]).astype("datetime64[ns]")
    return df


def load_articles(out_dir: str | Path, start, end) -> pd.DataFrame:
    """Matched articles collected for start..end (inclusive days)."""
    return _load(out_dir, start, end, ".csv.gz", "seen")


def load_slots(out_dir: str | Path, start, end) -> pd.DataFrame:
    """Per-hour slot logs collected for start..end: which file was read and how many articles it held."""
    return _load(out_dir, start, end, ".slots.csv", "hour")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect topic-matched GDELT GKG articles, one file per hour.")
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("out_dir")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    slots = collect(args.start, args.end, args.out_dir, workers=args.workers)
    missing = slots.loc[slots["file"] == "", "hour"]
    print(f"{len(slots)} hours, {len(missing)} without a GKG file, {int(slots['matched'].sum()):,} articles matched")


if __name__ == "__main__":
    main()
