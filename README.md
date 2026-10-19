# signalbench

Benchmarking AI models as trading-signal sources: decision models, time-series foundation models and their combinations, scored on calibrated forecasts.

> Work in progress.

## Approach

Every model is reduced to the same output: the probability that an instrument's value at a given horizon will be above each of a grid of thresholds around the current value. That makes very different models directly comparable:

- **Decision models** (Jev-style "System One" models served via llama.cpp `/v1/systemone`) answer one yes/no probability question per threshold.
- **Time-series foundation models** (e.g. Google TimesFM) produce quantile forecasts, which are mapped onto the same thresholds.
- **Baselines** (random walk, empirical distribution of past changes) set the bar to beat.

Forecasts are scored with Brier score, normalised CRPS, direction accuracy, interval coverage and answer coherence, always relative to the random-walk baseline with bootstrap confidence intervals.

## Layout

```
src/signalbench/   package: data loaders, state builders, predictors, backtest, scoring
notebooks/         research notebooks (narrative, experiments, results)
data/              cached price histories and model results (committed for reproducibility)
tests/             pytest suite
outputs/           generated charts (not committed)
```

## Setup

```bash
python -m venv venv
venv/Scripts/activate        # Windows; source venv/bin/activate elsewhere
pip install -e ".[notebook,dev]"
pytest
```

Decision models need a local llama.cpp server (Ollama, LM Studio, etc.) exposing `/v1/systemone`.
