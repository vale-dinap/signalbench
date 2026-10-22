from functools import partial

import numpy as np
import pandas as pd
import pytest

from sbench import context, news
from sbench.data import GOOGLE_TRENDS_DAY, Series, known_rows
from sbench.forecast import make_task
from sbench.states import build_news_state, build_search_state
from test_news import ROWS, gkg_file, zipped


def articles(rows):
    """rows: (seen, title, title_topics, tag_topics)"""
    return pd.DataFrame([{"seen": pd.Timestamp(s), "domain": "x.com", "url": "u", "title": t, "tone": -1.0,
                          "words": 100, "title_topics": tt, "tag_topics": tg} for s, t, tt, tg in rows])


def test_headline_articles_drop_feeds_and_bare_fed_and_tag_posts():
    a = context.headline_articles(articles([
        ("2026-03-10 10:00", "Apple Inc. (NASDAQ:AAPL) Shares Sold by Fund", "AAPL", ""),
        ("2026-03-10 10:00", "Delivery firm fed more than RON62M into unit", "MACRO", ""),
        ("2026-03-10 10:00", "Fed rate cut odds rise as inflation cools", "MACRO", ""),
        ("2026-03-10 10:00", "Oil jumps after post on tariffs", "USO;MACRO", "POSTS"),
        ("2026-03-10 10:00", "Garden show opens", "", "POSTS"),
    ]))
    assert a["title"].tolist() == ["Fed rate cut odds rise as inflation cools", "Oil jumps after post on tariffs"]
    assert a["topics"].tolist() == ["MACRO", "USO;MACRO;POSTS"]
    assert (a["known"] - a["seen"] == news.GKG_DELAY).all()


def make_feed(gap_hour=None, **kwargs):
    """Two days of hourly slots, 100 articles each; one BTC headline per hour (two from 06:00 on day 2)."""
    hours = pd.date_range("2026-03-01", periods=48, freq="h")
    cov = pd.DataFrame({"hour": hours, "file": hours.strftime("%Y%m%d%H%M%S"), "articles": 100,
                        "known": hours + news.GKG_DELAY, "n_BTC": 1, "tone_BTC": -1.0,
                        "n_MACRO": 0, "tone_MACRO": np.nan})
    cov.loc[hours >= "2026-03-02 06:00", ["n_BTC", "tone_BTC"]] = [2, 1.0]
    if gap_hour is not None:
        cov.loc[cov["hour"] == pd.Timestamp(gap_hour), ["file", "articles", "known", "n_BTC"]] = ["", 0, pd.NaT, 0]
    heads = context.headline_articles(articles(
        [(h, f"Bitcoin story {i}", "BTC", "") for i, h in enumerate(hours)]
        + [(hours[-1], "Bitcoin story 0", "BTC", ""),                            # a repeated title
           (hours[-2], "Tariffs hit markets", "MACRO", "")]))
    return context.NewsFeed(cov, heads, labels={"BTC": "Bitcoin"}, baseline_days=1, max_headlines=3, **kwargs)


def test_coverage_counts_the_last_day_against_the_day_before():
    feed = make_feed()
    c = feed.coverage_at("BTC", "2026-03-02 23:15")   # the 23:00 file is just known
    assert c["n"] == 6 * 1 + 18 * 2
    assert c["share_ratio"] == pytest.approx((42 / 2400) / (24 / 2400))
    assert c["tone"] == pytest.approx((6 * -1 + 36 * 1) / 42) and c["tone_baseline"] == -1.0


def test_files_not_yet_known_are_not_used():
    feed = make_feed()
    before = feed.coverage_at("BTC", "2026-03-02 23:14")   # the 23:00 file is known at 23:15
    assert before["n"] == feed.coverage_at("BTC", "2026-03-02 22:15")["n"]
    shown = feed.headlines_at("BTC", "2026-03-02 23:14", 3)
    assert shown["seen"].max() == pd.Timestamp("2026-03-02 22:00")


def test_a_gap_in_the_archive_gives_no_news():
    feed = make_feed(gap_hour="2026-03-02 05:00")
    assert feed.coverage_at("BTC", "2026-03-02 23:15") is None
    assert feed.lines("BTC", "2026-03-02 23:15") == ["News: not available for this period (gap in the news archive)."]
    assert np.isnan(feed.features("BTC", "2026-03-02 23:15")["coverage"])


def test_headlines_newest_first_distinct_and_capped():
    feed = make_feed()
    shown = feed.headlines_at("BTC", "2026-03-02 23:15", 3)
    # the 23:00 file holds story 47 and a repeat of story 0's title, which replaces the old one
    assert shown["title"].tolist() == ["Bitcoin story 0", "Bitcoin story 47", "Bitcoin story 46"]
    lines = feed.lines("BTC", "2026-03-02 23:15")
    assert lines[1].startswith("  Coverage: 42 articles with Bitcoin in the headline, 1.8× the usual share")
    assert "Tariffs hit markets" in lines[-1] and len(lines) == 2 + 4 + 2


def test_restricted_feed_and_saved_feed_show_the_same_lines(tmp_path):
    feed = make_feed()
    asked = [("BTC", pd.Timestamp("2026-03-02 23:15")), ("BTC", pd.Timestamp("2026-03-02 12:15"))]
    small = feed.restrict_to(asked)
    assert len(small.headlines) < len(feed.headlines)
    small.save(tmp_path)
    loaded = context.NewsFeed.load(tmp_path, labels={"BTC": "Bitcoin"}, baseline_days=1, max_headlines=3)
    for topic, until in asked:
        assert small.lines(topic, until) == feed.lines(topic, until) == loaded.lines(topic, until)


def test_feed_from_raw_files(tmp_path):
    def get(url, timeout):
        s = url.rsplit("/", 1)[1].split(".")[0]
        if s[10:12] != "00":
            return type("R", (), {"status_code": 404})()
        return type("R", (), {"status_code": 200, "content": zipped(gkg_file(s, ROWS), f"{s}.gkg.csv"),
                              "raise_for_status": lambda self: None})()

    news.collect("2025-01-01", "2025-01-02", tmp_path, workers=1, get=get, report=None)
    feed = context.NewsFeed.from_raw(tmp_path, "2025-01-01", "2025-01-02")
    assert len(feed.coverage) == 48 and (feed.coverage["articles"] == 4).all()
    assert (feed.coverage["n_BTC"] == 1).all() and (feed.coverage["n_AAPL"] == 1).all()
    assert feed.coverage_at("BTC", "2025-01-02 23:15")["share_ratio"] == pytest.approx(1.0)


def trends_interest():
    days = pd.date_range("2025-01-01", periods=60, freq="D")
    values = np.where(days < "2025-02-20", 10.0, 30.0)
    s = Series("BTC", "searches", pd.DataFrame({"t": days, "close": values}), kind="covariate",
               close_time=GOOGLE_TRENDS_DAY)
    return context.SearchInterest({"BTC": s}, {"BTC": "btc"})


def test_search_interest_uses_only_days_already_final():
    si = trends_interest()
    # day 2025-02-20 (first at 30) is final at 2025-02-22 00:00
    assert si.ratios("BTC", "2025-02-21 23:59") == (pytest.approx(1.0), pytest.approx(1.0))
    recent, latest = si.ratios("BTC", "2025-02-22 00:00")
    assert recent == pytest.approx((6 * 10 + 30) / 70) and latest == pytest.approx(30 / ((6 * 10 + 30) / 7))
    assert si.lines("BTC", "2025-01-10") == ["Google searches for 'btc': not available."]
    assert "last 7 days 1.29× the previous 28 days' average" in si.lines("BTC", "2025-02-22")[0]


def test_covariates_are_computed_at_each_row_close_and_known_with_it(gbm):
    seen = []

    def features(until):
        seen.append(until)
        return {"a": until.dayofyear, "b": np.nan if until.day == 1 else 1.0}

    a, b = context.covariates(gbm, features, prefix="news_")
    assert a.key == "news_a" and a.kind == "covariate"
    assert seen == list(map(pd.Timestamp, gbm.available_at()))
    np.testing.assert_array_equal(a.available_at(), gbm.available_at())
    assert len(b) < len(gbm)   # rows without a value are dropped; availability alignment still holds
    assert known_rows(b, gbm.available_at()[40]) == known_rows(gbm, gbm.available_at()[40]) - 2


def test_news_and_search_states_add_their_lines_and_hide_covariates(gbm):
    feed = make_feed()
    covs = context.covariates(gbm, lambda t: {"x": 1.0})
    task = make_task(gbm, 100, 5, related=covs)
    state = partial(build_news_state, feed=feed, lookback=5)(task)
    assert "Related:" not in state and state.endswith(feed.lines("gbm", task.known_at)[-1])
    si = trends_interest()
    si.terms["gbm"], si.series["gbm"] = "x", si.series["BTC"]
    assert build_search_state(task, si, lookback=5).splitlines()[-1].startswith("Google searches for 'x'")
    assert "NewsFeed(hours=24" in repr(feed) and "Bitcoin story" not in repr(feed)
