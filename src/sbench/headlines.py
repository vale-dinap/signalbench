"""Decision models as headline readers: rate headlines, aggregate them into a news score.

Instead of forecasting from headlines (03), a model is asked a narrower question
about each one: how good or bad is it for the instrument's price? One score
question per headline, on five levels whose middle one also covers headlines that
are unrelated, keeps a call short (each question costs about the same). The
ratings are averaged into one number per decision point, whose predictive value
can then be measured (information coefficient) and used as a feature.

The state holds the numbered headlines only: no prices and no dates, since
whether a headline is good or bad news should not depend on either.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .client import DEFAULT_URL, ask, score_probs, score_question

LEVELS = ["bad news for the price", "slightly bad news", "no effect, or unrelated", "slightly good news",
          "good news for the price"]
VALUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])   # rating of each level


@dataclass
class HeadlineReader:
    """One System One call per set of headlines: a five-level rating of each."""
    model: str
    question: str = "Headline {i}: for the price of {label}, how good or bad is this news?"
    levels: list[str] = field(default_factory=lambda: list(LEVELS))
    url: str = DEFAULT_URL
    timeout: int = 300

    def state(self, titles: list[str]) -> str:
        return "\n".join(["News headlines:", *[f"  {i}. {t}" for i, t in enumerate(titles, 1)]])

    def questions(self, n: int, label: str) -> dict:
        return {f"h{i}": score_question(self.question.format(i=i, label=label), self.levels) for i in range(1, n + 1)}

    def read(self, titles: list[str], label: str) -> pd.DataFrame:
        """One row per headline: title, the probability of each level (p0 = worst) and the expected rating."""
        cols = ["title", *[f"p{k}" for k in range(len(self.levels))], "rating"]
        if not titles:
            return pd.DataFrame(columns=cols)
        resp = ask(self.model, self.state(titles), self.questions(len(titles), label),
                   timeout=self.timeout, url=self.url)
        rows = []
        for i, t in enumerate(titles, 1):
            p = np.array(score_probs(resp.answers[f"h{i}"], len(self.levels)))
            p = p / p.sum() if p.sum() > 0 else np.full(len(p), np.nan)
            rows.append({"title": t, **{f"p{k}": v for k, v in enumerate(p)}, "rating": float(p @ VALUES)})
        return pd.DataFrame(rows)[cols]

    def describe(self) -> dict:
        d = {"type": type(self).__name__, **asdict(self)}
        d.pop("url"); d.pop("timeout")
        return d


def news_score(read: pd.DataFrame) -> dict[str, float]:
    """Mean rating (-1 to 1) and mean absolute rating of a set of headlines; NaN without headlines."""
    r = pd.to_numeric(read.get("rating", pd.Series(dtype=float)), errors="coerce").dropna()
    if r.empty:
        return {"score": np.nan, "intensity": np.nan}
    return {"score": float(r.mean()), "intensity": float(r.abs().mean())}


def read_all(reader: HeadlineReader, requests: Iterable[tuple[str, pd.Timestamp]],
             titles: Callable[[str, pd.Timestamp], list[str]], labels: dict[str, str], path: str | Path,
             report: Callable[[str], None] | None = print) -> pd.DataFrame:
    """Read the headlines for every (series, until) pair and store them; resumes from `path` (JSON lines).

    titles(series, until) gives the headlines to read. Each line holds one request's
    titles and ratings, so scores can be recomputed without the model. Returns one
    row per request: series, until, score, intensity, n_headlines.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = reader.describe()
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["reader"] != meta:
                raise ValueError(f"{path.name} was written by {rec['reader']}, not {meta}; use a new path")
            done[(rec["series"], pd.Timestamp(rec["until"]))] = rec
    wanted = list(dict.fromkeys((k, pd.Timestamp(u)) for k, u in requests))
    todo = [r for r in wanted if r not in done]
    if report:
        report(f"{reader.model}: {len(wanted) - len(todo)} read before, {len(todo)} to read")
    for n, (key, until) in enumerate(todo, 1):
        heads = titles(key, until)
        try:
            read = reader.read(heads, labels.get(key, key))
        except Exception as e:  # noqa: BLE001 -- one failed call must not stop a long run; retried on resume
            if report:
                report(f"  {key} {until}: {type(e).__name__}: {str(e)[:200]}")
            continue
        rec = {"reader": meta, "series": key, "until": str(until), "headlines": read.to_dict("records")}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        done[(key, until)] = rec
        if report and n % 100 == 0:
            report(f"  {n}/{len(todo)}")
    rows = []
    for key, until in (r for r in wanted if r in done):
        read = pd.DataFrame(done[(key, until)]["headlines"])
        rows.append({"series": key, "until": until, **news_score(read), "n_headlines": len(read)})
    return pd.DataFrame(rows)
