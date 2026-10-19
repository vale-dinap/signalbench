"""Ways of asking a decision model for a forecast.

Each elicitation turns a ForecastTask into System One questions and decodes the
answers into P(value > threshold) on the task's threshold grid. Thresholds an
elicitation does not cover are NaN (e.g. direction-only modes answer only the
current-value threshold). All modes are therefore scored by the same code.

    ThresholdNoul     one yes/no question per threshold            full grid, N questions
    BinDistribution   one score/choice question over value ranges  full grid, 1 question
    DirectionNoul     "will it be above the current value?"        P(up) only
    DirectionChoice   higher / lower (/ unchanged)                 P(up) only
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

from .client import (
    SCORE_MAX_LEVELS,
    choice_probs,
    choice_question,
    noul_p,
    noul_question,
    score_probs,
    score_question,
)
from .forecast import ForecastTask
from .states import horizon_text


def _fmt(task: ForecastTask, x: float) -> str:
    return task.series.fmt(x)


@dataclass
class ThresholdNoul:
    """P(value > threshold) asked directly, one noul question per threshold."""
    template: str = "Will the {label} {horizon} be strictly above {value}?"
    name: str = "threshold_noul"

    def questions(self, task: ForecastTask) -> dict:
        h = horizon_text(task.horizon, task.series.freq)
        return {f"q{i}": noul_question(self.template.format(
                    label=task.series.value_label, horizon=h, value=_fmt(task, x)))
                for i, x in enumerate(task.thresholds)}

    def decode(self, answers: dict, task: ForecastTask) -> np.ndarray:
        return np.array([noul_p(answers[f"q{i}"]) for i in range(len(task.thresholds))])

    def describe(self) -> dict:
        return {"type": type(self).__name__, **asdict(self)}


@dataclass
class BinDistribution:
    """One question whose options are the value ranges between thresholds.

    question_type="score" keeps the options ordered (max 10 ranges, i.e. 9 thresholds);
    "choice" has no cap but treats ranges as unordered labels.
    P(> threshold_k) = sum of probabilities of the ranges above threshold_k, so the
    result is coherent by construction.
    """
    question_type: str = "score"
    template: str = "Where will the {label} be {horizon}?"
    name: str = "bin_score"

    def __post_init__(self):
        if self.question_type not in ("score", "choice"):
            raise ValueError("question_type must be 'score' or 'choice'")

    @staticmethod
    def bin_labels(task: ForecastTask) -> list[str]:
        th = task.thresholds
        labels = [f"at or below {_fmt(task, th[0])}"]
        labels += [f"above {_fmt(task, a)} and at most {_fmt(task, b)}" for a, b in pairwise(th)]
        labels.append(f"above {_fmt(task, th[-1])}")
        return labels

    def questions(self, task: ForecastTask) -> dict:
        labels = self.bin_labels(task)
        text = self.template.format(label=task.series.value_label, horizon=horizon_text(task.horizon, task.series.freq))
        if self.question_type == "score":
            if len(labels) > SCORE_MAX_LEVELS:
                raise ValueError(f"score supports {SCORE_MAX_LEVELS} ranges; grid has {len(labels)}. "
                                 "Use fewer sigmas or question_type='choice'.")
            return {"bins": score_question(text, labels)}
        return {"bins": choice_question(text, {f"b{i}": lab for i, lab in enumerate(labels)})}

    def decode(self, answers: dict, task: ForecastTask) -> np.ndarray:
        n = len(task.thresholds) + 1
        a = answers["bins"]
        p = score_probs(a, n) if self.question_type == "score" else choice_probs(a, [f"b{i}" for i in range(n)])
        p = np.asarray(p, dtype=float)
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1 / n)
        # P(> th_k) = mass of bins k+1 .. n-1
        return np.cumsum(p[::-1])[::-1][1:]

    def describe(self) -> dict:
        return {"type": type(self).__name__, **asdict(self)}


@dataclass
class DirectionNoul:
    """Single yes/no: will the value be strictly above the current value?"""
    template: str = "Will the {label} {horizon} be strictly above the current {value}?"
    name: str = "direction_noul"

    def questions(self, task: ForecastTask) -> dict:
        return {"up": noul_question(self.template.format(
            label=task.series.value_label, horizon=horizon_text(task.horizon, task.series.freq),
            value=_fmt(task, task.current)))}

    def decode(self, answers: dict, task: ForecastTask) -> np.ndarray:
        p = np.full(len(task.thresholds), np.nan)
        p[task.i0] = noul_p(answers["up"])
        return p

    def describe(self) -> dict:
        return {"type": type(self).__name__, **asdict(self)}


@dataclass
class DirectionChoice:
    """Higher / lower (/ unchanged) as a choice. P(up) = probability of 'up'.

    With include_flat=False the model must pick a side, so ties are forced into
    'down' or 'up'; with include_flat=True 'unchanged' absorbs them, which matters
    for prediction markets that are often flat day to day.
    """
    include_flat: bool = True
    template: str = "Compared with now, where will the {label} be {horizon}?"
    name: str = "direction_choice"

    def options(self) -> dict:
        o = {"up": "Higher than now", "down": "Lower than now"}
        if self.include_flat:
            o["flat"] = "Unchanged"
        return o

    def questions(self, task: ForecastTask) -> dict:
        return {"dir": choice_question(self.template.format(
            label=task.series.value_label, horizon=horizon_text(task.horizon, task.series.freq)), self.options())}

    def decode(self, answers: dict, task: ForecastTask) -> np.ndarray:
        probs = dict(zip(self.options(), choice_probs(answers["dir"], self.options())))
        total = sum(probs.values())
        p = np.full(len(task.thresholds), np.nan)
        p[task.i0] = probs["up"] / total if total > 0 else 0.5
        return p

    def describe(self) -> dict:
        return {"type": type(self).__name__, **asdict(self)}
