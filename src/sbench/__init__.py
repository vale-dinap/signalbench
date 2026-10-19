"""sbench: core package of signalbench, benchmarking AI models as trading-signal sources.

Modules
-------
data        : Series container, loaders (Polymarket, Binance, Yahoo), caching, alignment
client      : System One decision-model client: noul / choice / score questions
elicitation : ways of asking a decision model for a forecast (thresholds, ranges, direction)
states      : natural-language state builders and wording
forecast    : ForecastTask, threshold grids, monotone CDFs, quantile mapping, CRPS
predictors  : common predictor interface; decision models and baselines
tsfm        : time-series foundation models (TimesFM) via quantile forecasts
backtest    : resumable forecast backtests (long-format CSV + run metadata + raw audit trail)
scoring     : Brier / log loss / CRPS / direction / coverage metrics with bootstrap CIs
"""
