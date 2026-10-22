"""External context known at a decision time: news coverage and headlines, search interest.

Each source answers one question for an instrument: what was known at time T?
It returns lines for a decision model's state and numbers for TimesFM covariates
(see covariates()). T is ForecastTask.known_at, so nothing published after the
decision is used.

NewsFeed is built from the GDELT GKG articles collected by sbench.news;
SearchInterest from daily Google Trends series (data.load_google_trends).
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import news
from .data import Series, known_rows

# Exchange-ticker tags mark auto-generated feed headlines (holdings changes, short
# interest, transcripts): "Linkhome Holdings Inc. (NASDAQ:LHAI) Short Interest Up 59.9%".
TICKER_TAG = re.compile(r"\((?:NASDAQ|NYSE|NYSEARCA|NYSEAMERICAN|NYSEMKT|OTCMKTS|OTC|BATS|CBOE|TSE|TSX|TSXV"
                        r"|LON|ASX|ETR|FRA|EPA|AMS|HKG|NSE|BOM)\s*:\s*[A-Z0-9.\-]+\)")
MACRO_TOPICS = ("MACRO", "POSTS")
# Stricter than news.TOPICS["MACRO"], whose bare "fed" also matches the verb ("... Fed More Than ...")
MACRO_TITLE = re.compile(r"tariff|federal reserve|\bthe fed\b|\bfed(?:'s)? (?:chair|rates?|policy|officials?|minutes"
                         r"|governors?|cuts?|hikes?|decision|meeting)|\bfomc\b|jerome powell|chair powell"
                         r"|\bpowell(?:'s)? (?:says|said|signals?|warns?|speech|testimony|remarks)|interest rates?|inflation"
                         r"|\bcpi\b|recession", re.IGNORECASE)


def headline_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """Articles whose title names a topic, without ticker-tagged feed headlines.

    Adds `known` (file stamp + news.GKG_DELAY) and `topics`: the title topics (MACRO
    only where MACRO_TITLE matches), plus POSTS when the article is tagged with Truth Social.
    """
    topics = articles["title_topics"].str.split(";")
    strict_macro = articles["title"].str.contains(MACRO_TITLE)
    topics = pd.Series([[t for t in ts if t and (t != "MACRO" or ok)] for ts, ok in zip(topics, strict_macro)],
                       index=articles.index)
    posts = articles["tag_topics"].str.contains(r"(?:^|;)POSTS(?:;|$)", regex=True)
    topics = pd.Series([ts + ["POSTS"] if p and ts and "POSTS" not in ts else ts for ts, p in zip(topics, posts)],
                       index=articles.index)
    keep = (topics.str.len() > 0) & ~articles["title"].str.contains(TICKER_TAG, regex=True)
    a = articles[keep]
    return pd.DataFrame({
        "seen": a["seen"], "known": a["seen"] + news.GKG_DELAY, "domain": a["domain"], "title": a["title"],
        "tone": a["tone"], "topics": topics[keep].str.join(";"),
    }).reset_index(drop=True)


def _has(topics: pd.Series, keys) -> pd.Series:
    keys = (keys,) if isinstance(keys, str) else tuple(keys)
    return topics.str.contains(r"(?:^|;)(?:" + "|".join(map(re.escape, keys)) + r")(?:;|$)", regex=True)


@dataclass(eq=False)
class NewsFeed:
    """News coverage and headlines per topic, as known at a given time.

    coverage  : one row per hour (news.load_slots): hour, file, known, articles (all
                articles in that hour's file) and per topic n_<topic>, tone_<topic>
                (count and mean tone of headline articles naming it); file "" = no data
    headlines : headline_articles(), at most `per_file` per topic and file
    labels    : topic -> how the state names it ("Bitcoin"); defaults to the topic key
    """
    coverage: pd.DataFrame
    headlines: pd.DataFrame
    labels: dict[str, str] = field(default_factory=dict)
    hours: int = 24
    baseline_days: int = 30
    max_headlines: int = 15
    max_macro: int = 5

    def __post_init__(self):
        self.coverage = self.coverage.sort_values("hour").reset_index(drop=True)
        self.headlines = self.headlines.sort_values("known", kind="stable").reset_index(drop=True)
        self._known = self.headlines["known"].to_numpy(dtype="datetime64[ns]")

    def __repr__(self) -> str:   # recorded in run metadata through the state builder
        return (f"NewsFeed(hours={self.hours}, baseline_days={self.baseline_days}, max_headlines="
                f"{self.max_headlines}, max_macro={self.max_macro}, labels={self.labels})")

    # ── Building and storing ──────────────────────────────────────────────────

    @classmethod
    def from_raw(cls, raw_dir: str | Path, start, end, topics=tuple(news.TOPICS), per_file: int = 15,
                 **kwargs) -> NewsFeed:
        """Build from sbench.news.collect output, one month at a time."""
        cov, heads = [], []
        months = pd.date_range(pd.Timestamp(start).normalize().replace(day=1), pd.Timestamp(end), freq="MS")
        for m in months:
            lo, hi = max(m, pd.Timestamp(start)), min(m + pd.offsets.MonthEnd(0), pd.Timestamp(end))
            slots = news.load_slots(raw_dir, lo, hi)
            h = headline_articles(news.load_articles(raw_dir, lo, hi))
            stamp = pd.to_datetime(slots["file"].where(slots["file"] != ""), format="%Y%m%d%H%M%S")
            c = slots[["hour", "file", "articles"]].assign(known=stamp + news.GKG_DELAY)
            exploded = h.assign(topic=h["topics"].str.split(";")).explode("topic")
            per = exploded.groupby(["seen", "topic"])["tone"].agg(["size", "mean"]).unstack("topic")
            for t in topics:
                n = per["size"][t] if t in per["size"] else pd.Series(dtype=float)
                tone = per["mean"][t] if t in per["mean"] else pd.Series(dtype=float)
                c[f"n_{t}"] = stamp.map(n).fillna(0).astype(int)
                c[f"tone_{t}"] = stamp.map(tone)
            exploded = exploded.drop_duplicates(["seen", "topic", "title"])
            exploded = exploded[exploded.groupby(["seen", "topic"]).cumcount() < per_file]
            heads.append(h.loc[sorted(set(exploded.index))])
            cov.append(c)
        return cls(pd.concat(cov, ignore_index=True), pd.concat(heads, ignore_index=True), **kwargs)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.coverage.to_csv(d / "news_coverage.csv.gz", index=False)
        self.headlines.to_csv(d / "news_headlines.csv.gz", index=False)

    @classmethod
    def load(cls, directory: str | Path, **kwargs) -> NewsFeed:
        d = Path(directory)
        cov = pd.read_csv(d / "news_coverage.csv.gz", keep_default_na=False, na_values={"known": [""]},
                          dtype={"file": str}, parse_dates=["hour", "known"])
        for c in cov.columns:
            if c.startswith("tone_"):
                cov[c] = pd.to_numeric(cov[c], errors="coerce")
        heads = pd.read_csv(d / "news_headlines.csv.gz", keep_default_na=False, na_values={"tone": [""]},
                            dtype={"domain": str, "title": str, "topics": str}, parse_dates=["seen", "known"])
        for frame, cols in ((cov, ("hour", "known")), (heads, ("seen", "known"))):
            for c in cols:
                frame[c] = frame[c].astype("datetime64[ns]")
        return cls(cov, heads, **kwargs)

    # ── What was known at `until` ─────────────────────────────────────────────

    def _hours(self, until: pd.Timestamp, hours: int) -> pd.DataFrame:
        """Hour slots whose file is due to be known in (until - hours, until]."""
        due = self.coverage["hour"] + news.GKG_DELAY
        return self.coverage[(due > until - pd.Timedelta(hours=hours)) & (due <= until)]

    def coverage_at(self, topic: str, until) -> dict | None:
        """Headline count, share of all news against the baseline period, and mean tone; None if data is missing."""
        until = pd.Timestamp(until)
        window = self._hours(until, self.hours)
        if len(window) < self.hours or (window["file"] == "").any() or (window["known"] > until).any():
            return None
        before = self._hours(until - pd.Timedelta(hours=self.hours), 24 * self.baseline_days)
        before = before[before["file"] != ""]
        n, n0 = window[f"n_{topic}"].sum(), before[f"n_{topic}"].sum()
        share, share0 = n / window["articles"].sum(), n0 / before["articles"].sum() if len(before) else np.nan

        def mean_tone(w):
            return float(np.average(w[f"tone_{topic}"].fillna(0), weights=w[f"n_{topic}"])) \
                if w[f"n_{topic}"].sum() else np.nan

        return {"n": int(n), "share_ratio": share / share0 if share0 else np.nan,
                "tone": mean_tone(window), "tone_baseline": mean_tone(before)}

    def headlines_at(self, topics, until, n: int) -> pd.DataFrame:
        """Newest distinct headlines naming any of `topics` known in the last `hours` before until."""
        until = pd.Timestamp(until)
        lo, hi = np.searchsorted(self._known, np.datetime64(until - pd.Timedelta(hours=self.hours), "ns"), "right"), \
            np.searchsorted(self._known, np.datetime64(until, "ns"), "right")
        w = self.headlines.iloc[lo:hi]
        w = w[_has(w["topics"], topics)].iloc[::-1]
        return w[~w["title"].str.casefold().duplicated()].head(n)

    def restrict_to(self, requests) -> NewsFeed:
        """A copy keeping only the headlines lines() shows for these (topic, until) pairs.

        Coverage is kept whole. The copy gives the same lines for those pairs and is
        small enough to commit; at other times it would show fewer headlines.
        """
        keep = set()
        for topic, until in requests:
            keep |= set(self.headlines_at(topic, until, self.max_headlines).index)
            keep |= set(self.headlines_at(MACRO_TOPICS, until, self.max_macro).index)
        return replace(self, headlines=self.headlines.loc[sorted(keep)])

    def features(self, topic: str, until) -> dict[str, float]:
        """Covariate values: log share of coverage against the baseline, and mean tone."""
        c = self.coverage_at(topic, until)
        if c is None:
            return {"coverage": np.nan, "tone": np.nan}
        return {"coverage": float(np.log(c["share_ratio"])) if c["share_ratio"] > 0 else np.nan,
                "tone": c["tone"]}

    def lines(self, topic: str, until) -> list[str]:
        label = self.labels.get(topic, topic)
        c = self.coverage_at(topic, until)
        if c is None:
            return ["News: not available for this period (gap in the news archive)."]
        out = [f"News from English-language sources (GDELT), last {self.hours} hours:"]
        tone = "n/a" if np.isnan(c["tone"]) else f"{c['tone']:+.1f}"
        tone0 = "n/a" if np.isnan(c["tone_baseline"]) else f"{c['tone_baseline']:+.1f}"
        ratio = "n/a" if np.isnan(c["share_ratio"]) else f"{c['share_ratio']:.1f}×"
        out.append(f"  Coverage: {c['n']} articles with {label} in the headline, {ratio} the usual share of all "
                   f"news (previous {self.baseline_days} days); average tone {tone} (previous {self.baseline_days} "
                   f"days {tone0}; below 0 = negative)")

        def block(title: str, rows: pd.DataFrame, empty: str) -> list[str]:
            if rows.empty:
                return [f"  {empty}"]
            return [f"  {title}:"] + [f"    {r.seen:%Y-%m-%d %H:%M} UTC  {r.domain}  {r.title}"
                                      for r in rows.itertuples()]

        out += block(f"Headlines about {label} (newest first)", self.headlines_at(topic, until, self.max_headlines),
                     f"No headlines about {label}.")
        out += block("Macro and policy headlines: tariffs, central banks, inflation, posts on Truth Social",
                     self.headlines_at(MACRO_TOPICS, until, self.max_macro), "No macro or policy headlines.")
        return out


@dataclass(eq=False)
class SearchInterest:
    """Google search interest per instrument, as known at a given time.

    series : instrument key -> daily search interest (data.load_google_trends) with
             close_time data.GOOGLE_TRENDS_DAY; the scale is arbitrary, so only
             ratios are used
    terms  : instrument key -> the search term, for the state text
    """
    series: dict[str, Series]
    terms: dict[str, str]
    recent: int = 7
    baseline: int = 28

    def __repr__(self) -> str:   # recorded in run metadata through the state builder
        return f"SearchInterest(recent={self.recent}, baseline={self.baseline}, terms={self.terms})"

    def ratios(self, key: str, until) -> tuple[float, float] | None:
        """(recent days / previous baseline days, latest day / recent days), from days known at until."""
        s = self.series[key]
        v = s.df["close"].to_numpy()[:int(known_rows(s, pd.Timestamp(until)))]
        if len(v) < self.recent + self.baseline:
            return None
        recent, base = v[-self.recent:].mean(), v[-self.recent - self.baseline:-self.recent].mean()
        if recent <= 0 or base <= 0 or v[-1] <= 0:
            return None
        return float(recent / base), float(v[-1] / recent)

    def features(self, key: str, until) -> dict[str, float]:
        r = self.ratios(key, until)
        return {"search_trend": np.log(r[0]) if r else np.nan, "search_latest": np.log(r[1]) if r else np.nan}

    def lines(self, key: str, until) -> list[str]:
        r = self.ratios(key, until)
        term = self.terms[key]
        if r is None:
            return [f"Google searches for '{term}': not available."]
        line = (f"Google searches for '{term}' (worldwide): last {self.recent} days {r[0]:.2f}× the previous "
                f"{self.baseline} days' average; latest day {r[1]:.2f}× the last {self.recent} days' average.")
        return [line]


def covariates(base: Series, features: Callable[[pd.Timestamp], dict[str, float]], prefix: str = "") -> tuple[Series, ...]:
    """Covariate series on base's own rows: features(known_at) per row, known when that row is.

    The values are computed from what was known at each row's close, so aligning
    them by availability (as TimesFM's covariates are) adds no look-ahead.
    """
    rows = pd.DataFrame([features(pd.Timestamp(t)) for t in base.available_at()])
    return tuple(Series(f"{prefix}{name}", f"{base.key} {name}", pd.DataFrame({"t": base.df["t"], "close": rows[name]}),
                        kind="covariate", close_time=base.close_time) for name in rows.columns)
