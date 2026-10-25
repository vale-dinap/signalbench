# Forward test: pre-registered plan

This plan fixes, before the forward period begins, what the forward test will measure and how its results will be judged. The commit that adds this file dates it. Notebooks 01 to 04 use data up to 30 September 2026 only, and no later data had been looked at when the plan was written. The section on headline readers was added after notebook 04, on 23 October 2026, and the prediction-market rules were made precise on 24 October 2026, both before the forward period. Mistral's Timestral, released on 15 October 2026, was added on 25 October 2026 (H8 and H9) after notebook 06 had scored it on the data of 01–03; that data ends before the forward period, so the addition, like the plan, cannot reflect it. On the same day, `sbench.scoring.skill_table` was fixed to draw each comparison's bootstrap from its own generator over its own pairs; before, an interval also depended on the other predictors scored in the same call.

## Why a forward test

Notebooks 02 and 03 control look-ahead with blinded states and comparisons before and after the decision models' release, but they test on a period that some models may have seen, and their hypotheses were partly formed on that same period (for example, the Bollinger reversal noted after 02). The forward test removes both problems: the period starts after every model file below was obtained (the latest, Timestral's, on 25 October 2026) and after this plan, so neither the models nor the hypotheses can reflect it.

No daily collection is needed: every source keeps an archive (Binance and ECB histories, Yahoo closes, GDELT's GKG files, Google Trends, Polymarket price histories), so the test is run once the period has passed, with the same code.

## Period and timing

- **Forward period:** 1 November 2026 to 30 April 2027. It starts after the plan and after the latest model file, not on 1 October: a model file published in mid-October could in principle have been trained on early-October data.
- **Evaluation:** once all outcomes in the period are known (early May 2027), in the notebook `05_forward_test`, which is committed before the period without being run, so the analysis is fixed as well as the hypotheses.
- **Update:** the same analysis repeated over 1 November 2026 to 31 October 2027 for the hypotheses flagged as low-power below. The first evaluation is reported regardless of the update.

## What is frozen

**Models** (local GGUF files, identified by SHA-256; a different file means a different model and is not part of this test):

| Model | File | Size (bytes) | SHA-256 |
|---|---|---|---|
| clef | `ggml-org/Clef-GGUF/Clef-Q4_K_M.gguf` | 19,232,219,200 | `9997675e4c9e9016273372b5fdedbdda7ad14285bc722949e4b9daca62bf3822` |
| lev | `ggml-org/lev-GGUF/lev-Q4_K_M.gguf` | 3,011,777,440 | `3f61b27c00a098dbc79ed099cca3985cbe0b63d17ac7fedebf2dca1d84f7f3a8` |
| kev-9b | `ggml-org/Kev-9B-GGUF/Kev-9B-Q4_K_M.gguf` | 6,358,923,744 | `86a6084984a6ef6818eb12c07cdc67ed3c47b00208d41a8816c64301bc6bcbda` |

**TimesFM:** `google/timesfm-3.0-pytorch` at Hugging Face revision `43046b85ec22d584a13f8098c2ed39c889e129c2`, run with the `timesfm` package 3.0.2.

**Timestral:** `mistralai/Timestral-209M` at Hugging Face revision `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`, run with the `timestral` package 1.0.0 at commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` of its GitHub repository, as in 06.

**Code:** the scoring (`sbench.scoring`), forecast tasks and grids (`sbench.forecast`), elicitation formats (`sbench.elicitation`), headline readers (`sbench.headlines`), prediction-market questions (`sbench.markets`), hypothesis verdicts (`sbench.hypotheses`), state builders (`sbench.states`, `sbench.context`) and predictors as of the last commit that amends this plan before the forward period (since the original plan, the modules frozen then only gained additions: `rank_ic`, `sbench.headlines`, `sbench.markets`, `sbench.hypotheses` and the `Timestral` backend in `sbench.tsfm`). Data loaders may be extended to fetch the new period; any other change is a deviation (see below).

**Configuration,** as in 03: the same eight instruments (BTC, ETH from Binance; SPY, AAPL, GLD, USO from Yahoo; EUR/USD and USD/JPY from ECB reference rates), the same close times, the threshold grid at ±0.5 to ±2 recent standard deviations, 60 closes in the decision models' state, the same related series, news parameters (24 hours, 15 instrument headlines, 5 macro headlines, 30-day baseline) and search terms. Decision models use the named state.

## Decision points

- **Horizon 1:** every trading day of each instrument in the period (about 1,000 points).
- **Horizon 5:** every fifth observation of each instrument in the period (about 200 points).
- A decision point needs its outcome inside the period. Points without news (archive gaps) are left out of every comparison that involves the news variant, for all predictors alike.

## Hypotheses

Each comparison is paired on decision points, uses skill = 1 − mean score / mean score of the reference, and a 95% paired bootstrap interval (2,000 resamples, seed 0), as in 02 and 03. "Holds" means the interval lies entirely on the stated side of zero; "fails" means the point estimate is on the other side; anything else is "inconclusive".

| | Hypothesis (from) | Comparison | Metric | Holds if |
|---|---|---|---|---|
| H1 | No predictor beats the random walk on real prices (02, 03) | every predictor of 02 and 03 vs random walk, horizons 1 and 5 (Timestral is tested by H9) | CRPS (distributions), Brier on P(up) (direction) | no predictor's interval lies entirely above 0 |
| H2 | News makes lev worse (03) | `lev/direction_choice+news` vs `lev/direction_choice`, horizon 1 | Brier on P(up) | interval below 0 |
| H3 | News makes clef worse (03) | `clef/threshold_noul+news` vs `clef/threshold_noul`, horizon 1 | CRPS | interval below 0 |
| H4 | News makes the decision models more confident (03) | same pairs as H2 and H3, horizon 1 | standard deviation of P(up), news minus price only | interval above 0 for both models |
| H5 | Search ratios improve clef's direction answers (03) | `clef/threshold_noul+search` vs `clef/threshold_noul`, horizon 1 | Brier on P(up) | interval above 0 |
| H6 | Indicators improve kev-9b (03) | `kev-9b/threshold_noul+indicators` vs `kev-9b/threshold_noul`, horizon 5 | CRPS | interval above 0 |
| H7 | Prices revert from the Bollinger bands (02) | realised up-rate on %B (20 days, 2 sd) at decision points, horizon 5, pooled | slope of a linear fit of up (1/0) on %B, per standard deviation of %B | interval below 0 |
| H8 | Timestral beats TimesFM 3.0 (06) | `timestral` vs `timesfm3`, price only, horizon 1 | CRPS | interval above 0 |
| H9 | Timestral beats the random walk (06) | `timestral` vs `random_walk`, horizon 1 | CRPS | interval above 0 |

TimesFM's and Timestral's covariates (related assets, news, search interest) are reported as estimates with intervals, without a hypothesis: 03 and 06 found no effect in either direction, and an absence of effect cannot be confirmed by an interval. So are H8's and H9's comparisons at horizon 5, which has too few points for effects of their size.

**Power.** At horizon 1 the intervals in 03 were about ±0.01 in CRPS skill and ±0.01 to ±0.035 in Brier skill on about 1,300 points; the forward test has about 1,000, so H2, H3 and H5 have enough points to detect effects of the size found in 03 (H5 only just: 03's effect was +0.012, interval +0.001 to +0.022). H6 and H7 rely on about 200 horizon-5 points: 03's H6 effect (+0.026, interval +0.015 to +0.036 on 1,228 points) should still be visible, but the Bollinger slope found after 02 (−0.029 per standard deviation of %B, interval −0.057 to −0.004 on 1,232 points) would not be distinguishable from zero on 200. H6 and H7 are therefore also evaluated at the 12-month update, and an inconclusive first result for them is expected, not a failure. In 06, on data Timestral may have seen, its CRPS skill at horizon 1 was +0.013 (+0.005 to +0.022) against TimesFM 3.0 and +0.008 (+0.001 to +0.015) against the random walk, on 1,312 points: H8 is detectable on about 1,000 points only if the effect is not smaller out of sample, and H9 is not, so H8 and H9 are also evaluated at the 12-month update, where an inconclusive H9 is again expected rather than a failure.

**Multiple comparisons.** Nine hypotheses are tested, each one-sided at 2.5%: if none were true, about one run in five would still show at least one false "holds". Every hypothesis is reported with its interval, whatever the outcome; none is dropped or reworded after the data is seen.

## Headline readers (exploratory)

Notebook 04 asked lev and clef to rate each headline for an instrument's price on five levels, and found that their news score does not predict the next return, while their intensity (mean absolute rating) rose with the size of the next move. Within each instrument that link was close to zero at horizon 1 from March 2026, and small at horizon 5 over 2024–2026 (IC about +0.07, lower bound about +0.01), a period the models may have seen. The forward period is the first test the models cannot have seen. It has too few horizon-5 points to confirm an effect of that size, so these are reported as estimates with intervals, without a hypothesis:

- **Readers:** lev and clef, with `HeadlineReader`'s question and levels as frozen above, on the decision points and headlines of the news variant (15 instrument headlines from the last 24 hours).
- **News score:** IC with the return over the following horizon, horizons 1 and 5.
- **Intensity:** IC with the size of the next move, using percentile ranks within each instrument, and IC with the part of that size left unexplained by the past move's size and the trailing 60-observation volatility (a regression on ranks, per instrument), horizons 1 and 5.
- **Interval:** 95% bootstrap resampling dates (2,000 resamples, seed 0), as `sbench.scoring.rank_ic`.

## Prediction markets (exploratory)

Prediction markets were held out of 02 because resolved questions can be recalled; markets that resolve in the forward period cannot have been seen by any model above. The test is exploratory (no hypothesis), with its selection fixed here:

- **Events:** the 200 highest-volume closed Polymarket events whose end date falls in the forward period.
- **Markets:** from each event, the highest-volume market that is binary (Yes/No), resolved, still open at the forecast time (a market that closed earlier asks about a known result) and has at least 30 daily prices at or before the forecast time, the last of them at most one day before it (the week before the forecast time is filled in from hourly prices, because the daily history can skip days). One market per event, so that events with many related markets (one per candidate, say) do not dominate; an event without such a market is left out.
- **Question:** whether the market resolves Yes, asked 7 days before the market's end date. The state holds the market question, its resolution date and the last 60 daily Yes prices up to the forecast time; P(Yes) is asked as the probability that the Yes price is above 0.50 at the end date.
- **Predictors:** clef and lev, both with `threshold_noul` at that single threshold. lev's own format (`direction_choice`) answers whether the price will be higher than now, not whether the market resolves Yes; its format was chosen in 01 for price grids, so this is a stated exception, not a deviation. The market price at the forecast time is the reference, as the bar to beat.
- **Metric:** Brier score of P(Yes) against the outcome, as skill relative to the market price, with a 95% paired bootstrap interval over markets (`sbench.markets`, `sbench.scoring.skill_table`).

## Deviations

Anything done differently from this plan (a data source that changed, a model file that could not be used, a bug fix in frozen code) will be listed in `05_forward_test` with its reason and, where possible, the result under both the plan and the deviation.
