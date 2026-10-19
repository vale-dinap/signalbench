"""Client for decision-model servers exposing the System One API (/v1/systemone).

Request:  {"model", "state", "questions": {key: question}}
Question types:
    noul   : {"type": "noul", "instructions"}                       -> {"noul": p_yes}
    choice : {"type": "choice", "instructions", "criteria": {k: v}}  -> {"choice", "probabilities", "confidence"}
    score  : {"type": "score", "instructions", "criteria": [levels]} -> {"score", "legend", "probabilities", "confidence"}
             2-10 levels, lowest first; probabilities are keyed by level index ("0", "1", ...).
"""
from __future__ import annotations

from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_URL = "http://localhost:2345/v1/systemone"
SCORE_MIN_LEVELS, SCORE_MAX_LEVELS = 2, 10

_session = requests.Session()
_session.mount("http://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=1.0, status_forcelist=(502, 503, 504), allowed_methods=None)))


# ── Question builders ─────────────────────────────────────────────────────────

def noul_question(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def choice_question(instructions: str, options: dict[str, str]) -> dict:
    if len(options) < 2:
        raise ValueError("choice needs at least 2 options")
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score_question(instructions: str, levels: list[str]) -> dict:
    if not SCORE_MIN_LEVELS <= len(levels) <= SCORE_MAX_LEVELS:
        raise ValueError(f"score needs {SCORE_MIN_LEVELS}-{SCORE_MAX_LEVELS} levels, got {len(levels)}")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


# ── Answer accessors (tolerant of server-side key types) ──────────────────────

def noul_p(answer: dict) -> float:
    """P(yes) of a noul answer."""
    return float(answer["noul"])


def choice_probs(answer: dict, options) -> list[float]:
    """Choice probabilities in the given option-key order (missing keys -> 0)."""
    probs = answer.get("probabilities", {})
    return [float(probs.get(k, 0.0)) for k in options]


def score_probs(answer: dict, n_levels: int) -> list[float]:
    """Score probabilities ordered lowest level first."""
    probs = {int(k): float(v) for k, v in answer.get("probabilities", {}).items()}
    return [probs.get(i, 0.0) for i in range(n_levels)]


# ── Requests ──────────────────────────────────────────────────────────────────

@dataclass
class Response:
    answers: dict
    input_tokens: int
    raw: dict


def query_decision_model(model: str, state: str, questions: dict, timeout: int = 300,
                         url: str = DEFAULT_URL) -> dict:
    """POST one state + questions; returns the raw JSON. Raises on HTTP errors."""
    payload = {"model": model, "state": state, "questions": questions}
    r = _session.post(url, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"{model}: HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def ask(model: str, state: str, questions: dict, timeout: int = 300, url: str = DEFAULT_URL) -> Response:
    """Query and check every question was answered."""
    raw = query_decision_model(model, state, questions, timeout=timeout, url=url)
    answers = raw.get("answers", {})
    missing = set(questions) - set(answers)
    if missing:
        raise RuntimeError(f"{model}: no answer for {sorted(missing)}")
    return Response(answers=answers, input_tokens=int(raw.get("usage", {}).get("input_tokens", 0)), raw=raw)
