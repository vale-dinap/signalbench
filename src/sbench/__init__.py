"""sbench: core package of signalbench, benchmarking AI models as trading-signal sources.

Modules
-------
data        : Series container, loaders (Polymarket, Binance, Yahoo, ECB, Google Trends), caching, alignment
panel       : the eight-instrument panel of notebooks 03 onward, from committed snapshots
synthetic   : synthetic price series with known conditional distributions, and the oracle
news        : GDELT GKG raw-file collector (news coverage, tone, headlines)
context     : news and search-interest context known at a decision time
client      : System One decision-model client: noul / choice / score questions
elicitation : ways of asking a decision model for a forecast (thresholds, ranges, direction)
states      : natural-language state builders and wording
headlines   : decision models as headline readers; news score and intensity
forecast    : ForecastTask, threshold grids, monotone CDFs, quantile mapping, CRPS
predictors  : common predictor interface; decision models and baselines
tsfm        : time-series foundation models (TimesFM) via quantile forecasts
backtest    : resumable forecast backtests (long-format CSV + run metadata + raw audit trail)
combine     : linear pools and agreement gates over stored forecasts
markets     : prediction markets as questions: will a market resolve Yes? Selection, runner, scoring
scoring     : Brier / log loss / CRPS / direction / coverage metrics with bootstrap CIs; rank IC
plots       : house style and the recurring chart types
"""
