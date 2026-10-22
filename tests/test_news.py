import io
import zipfile

import pandas as pd
import pytest

from sbench import news


def gkg_row(stamp, title, themes="", orgs="", names="", tone="-2.5,1,3.5,4.5,20,0,420", url="https://x.com/a"):
    cols = [""] * 27
    cols[0], cols[1], cols[3], cols[4] = f"{stamp}-1", stamp, url.split("/")[2], url
    cols[7], cols[13], cols[15], cols[23] = themes, orgs, tone, names
    cols[26] = f"<PAGE_LINKS>https://y.com</PAGE_LINKS><PAGE_TITLE>{title}</PAGE_TITLE>"
    return "\t".join(cols)


def gkg_file(stamp, rows) -> bytes:
    return ("\n".join(gkg_row(stamp, *r) for r in rows) + "\n").encode("utf-8")


ROWS = [
    ("Bitcoin slides below $112,000 &#x2014; analysts",),
    ("Markets rally", "ECON_STOCKMARKET;ECON_BITCOIN;", "", "Wall Street,10;Truth Social,55"),
    ("Apple unveils a phone", "", "apple inc;samsung"),
    ("Local garden show opens", "EDUCATION;"),
]


def test_parse_gkg_extracts_title_tone_and_tags():
    a = news.parse_gkg(gkg_file("20250101000000", ROWS))
    assert a["seen"].dtype == "datetime64[ns]" and a["seen"].iloc[0] == pd.Timestamp("2025-01-01")
    assert a["title"].iloc[0] == "Bitcoin slides below $112,000 — analysts"
    assert a["tone"].iloc[0] == -2.5 and a["words"].iloc[0] == 420
    assert a["names"].iloc[1] == "wall street;truth social"
    assert a["domain"].iloc[0] == "x.com"


def test_topics_match_titles_and_exact_tags_only():
    m = news.match_topics(news.parse_gkg(gkg_file("20250101000000", ROWS)))
    assert len(m) == 3   # the garden show matches nothing
    assert list(m.columns) == news.ARTICLE_COLUMNS
    assert m["title_topics"].tolist() == ["BTC", "", "AAPL"]
    assert m["tag_topics"].tolist() == ["", "BTC;SPY;POSTS", "AAPL"]
    only = news.match_topics(news.parse_gkg(gkg_file("20250101000000", [("x", "", "pineapple inc")])))
    assert only.empty   # names match whole entries, not substrings


class Response:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise news.requests.HTTPError(str(self.status_code))


def zipped(raw: bytes, name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, raw)
    return buf.getvalue()


@pytest.fixture
def fake_gdelt():
    """Every :00 file exists except at 05:00, where only the :30 file does; 07:00 has none at all."""
    requested = []

    def get(url, timeout):
        s = url.rsplit("/", 1)[1].split(".")[0]
        requested.append(s)
        hour, minute = int(s[8:10]), int(s[10:12])
        if (hour == 7) or (hour == 5 and minute != 30) or (hour != 5 and minute != 0):
            return Response(404)
        return Response(200, zipped(gkg_file(s, ROWS), f"{s}.gkg.csv"))

    get.requested = requested
    return get


def test_collect_day_falls_back_to_later_quarter_hours_and_logs_gaps(tmp_path, fake_gdelt):
    log = news.collect_day("2025-01-01", tmp_path, get=fake_gdelt)
    assert len(log) == 24
    assert log.loc[5, "file"] == "20250101053000"
    assert log.loc[7, "file"] == "" and log.loc[7, "articles"] == 0
    assert log["matched"].sum() == 3 * 23
    articles = news.load_articles(tmp_path, "2025-01-01", "2025-01-01")
    assert len(articles) == 69 and articles["seen"].dtype == "datetime64[ns]"
    assert articles["title_topics"].tolist()[:3] == ["BTC", "", "AAPL"]


def test_collect_resumes_and_skips_finished_days(tmp_path, fake_gdelt):
    news.collect("2025-01-01", "2025-01-02", tmp_path, workers=2, get=fake_gdelt, report=None)
    n = len(fake_gdelt.requested)
    slots = news.collect("2025-01-01", "2025-01-03", tmp_path, workers=2, get=fake_gdelt, report=None)
    assert len(fake_gdelt.requested) - n == n // 2   # only the new day is fetched
    assert len(slots) == 72 and pd.api.types.is_string_dtype(slots["file"])
    assert (slots["file"] == "").sum() == 3
