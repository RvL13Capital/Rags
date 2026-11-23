# Pipeline verification report

> **PIPELINE VERIFICATION - SYNTHETIC DATA.** These results exercise the code on simulated prices.
> They are not empirical market performance and say nothing about the NASDAQ-100.

Synthetic sample 2012-2026; walk-forward 36m train / 12m validation / 6m OOS; 40 Optuna trials per fold where anything is tuned; 10000 Monte Carlo paths; lots normalised to index level 25,288.25; runtime 427s.

| Data set | OOS Sharpe | OOS PF | OOS max DD | OOS trades | OOS net P&L (EUR) | DSR (gate) | N_eff | DSR (cross-trial mean, info) | CSCV P(OOS loss) | Rank PBO | Gates failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Null (martingale), full geometry tuned | -0.05 | 0.97 | 3.46% | 195 | -27.86 | 0.077 | 24.2 | 0.178 | 0.829 | 0.518 | 7 of 10 |
| Planted edge x2 (after ablation, thresholds only) | 0.83 | 1.57 | 2.14% | 159 | 479.24 | 0.962 | 6.3 | 0.267 | 0.126 | n/a | 0 of 10 |

## Checks

- Null data rejected by the release gates: **PASS** (failed gates: OOS net Sharpe, Deflated Sharpe Ratio (zero-mean null, N_eff), CSCV P(OOS loss of IS-best), CSCV rank PBO, OOS net profit factor, OOS net P&L under 1.5x costs, Profitable walk-forward OOS folds)
- Planted edge clearly visible out of sample (Sharpe at least 0.5 above null, positive net P&L, profit factor above 1.2): **PASS**
- Sizing per vehicle at EUR 5,000 (whole-unit IBUST100 never sizes, the 0.001-lot CFD does): **PASS**

## Operating characteristics of the release gates

On this run the positive control fails 0 of 10 gates. One history says little, so the fixed-prior baseline was also run on independent synthetic histories (different seeds, nothing tuned, K = 1), each through the complete battery:

| Planted strength | Histories | Pass every gate | Median OOS Sharpe | Gates failed most often |
|---:|---:|---:|---:|---|
| 0 | 40 | 0 | -0.22 | OOS net Sharpe (40); Deflated Sharpe Ratio (zero-mean null, N_eff) (40); Profitable walk-forward OOS folds (39) |
| 2 | 40 | 24 | 0.93 | OOS net Sharpe (13); Profitable walk-forward OOS folds (13) |

The first row estimates the false-positive rate of the battery, the second its power against an edge of this size. Thresholds: Sharpe > 0.80, DSR > 0.95, P(OOS loss) < 15%, profit factor > 1.20, positive under 1.5x costs, max drawdown < 10%, Monte Carlo P(DD >= 20%) < 2%, profitable folds >= 65%, trades >= 150.

## How strong must an edge be to be seen?

Fixed-prior baseline, no tuning, stitched walk-forward OOS, same seed. Strength 1 is a drift of 0.12% a day inside trend regimes against roughly 0.9% daily volatility; the positive control above uses strength 2.

| Planted strength | Trend drift / day | OOS Sharpe | OOS PF | Expectancy (R) | Trades | Net P&L (EUR) | Profitable folds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | -0.01 | 0.99 | -0.011 | 204 | -14.80 | 50% |
| 1 | 0.12% | 0.22 | 1.13 | 0.090 | 210 | 183.42 | 48% |
| 2 | 0.24% | 0.62 | 1.40 | 0.256 | 225 | 599.81 | 60% |
| 3 | 0.36% | 1.54 | 2.14 | 0.626 | 249 | 1827.91 | 64% |

## Planted-edge ablation

# Sequential ablation (stitched walk-forward OOS)

Retention floor on trade count: 112. Variants evaluated: 15.

| Step | Module | Variant | Status | OOS Sharpe | d Sharpe | Calmar | d Calmar | PF | d PF | ES99 | d ES99 | Max DD | Trades |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | - | BASELINE | 0.621 | n/a | 0.323 | n/a | 1.404 | n/a | 0.44% | n/a | 3.30% | 225 |
| 1 | multi_stage_exits | - | RETAINED | 0.765 | 0.144 | 0.376 | 0.053 | 1.408 | 0.004 | 0.35% | -0.09% | 2.63% | 239 |
| 2 | trend_flip_exit | - | rejected (no OOS net gain) | 0.765 | 0.000 | 0.376 | 0.000 | 1.408 | 0.000 | 0.35% | 0.00% | 2.63% | 239 |
| 3 | regime_gate | rv20 | rejected (parsimony benchmark) | 0.611 | -0.153 | 0.291 | -0.084 | 1.358 | -0.049 | 0.34% | -0.01% | 2.40% | 186 |
| 3 | regime_gate | hmm2 | RETAINED | 0.835 | 0.070 | 0.402 | 0.027 | 1.574 | 0.166 | 0.30% | -0.04% | 2.14% | 159 |
| 3 | regime_gate | hmm3 | rejected (parsimony benchmark) | 0.520 | -0.245 | 0.297 | -0.079 | 1.530 | 0.122 | 0.24% | -0.11% | 1.16% | 69 |
| 4 | garch_vol | - | rejected (no OOS net gain) | 0.464 | -0.371 | 0.203 | -0.199 | 1.315 | -0.259 | 0.30% | -0.00% | 2.14% | 134 |
| 5 | evt_es_cap | - | rejected (no OOS net gain) | 0.608 | -0.226 | 0.288 | -0.114 | 1.445 | -0.129 | 0.19% | -0.11% | 1.23% | 159 |
| 6 | spread_gate | - | rejected (no OOS net gain) | 0.787 | -0.048 | 0.347 | -0.056 | 1.556 | -0.018 | 0.30% | -0.00% | 2.27% | 146 |
| 7 | vpin_filter | - | rejected (no OOS net gain) | 0.857 | 0.022 | 0.407 | 0.004 | 1.610 | 0.036 | 0.30% | -0.00% | 2.14% | 155 |
| 8 | event_blackout | - | rejected (no OOS net gain) | 0.813 | -0.022 | 0.391 | -0.011 | 1.563 | -0.011 | 0.30% | 0.00% | 2.13% | 157 |
| 9 | score_sizing | - | rejected (no OOS net gain) | 0.832 | -0.003 | 0.402 | 0.000 | 1.577 | 0.003 | 0.30% | -0.00% | 2.09% | 159 |
| 10 | kelly_cap | - | rejected (no OOS net gain) | 0.533 | -0.301 | 0.187 | -0.215 | 1.380 | -0.194 | 0.29% | -0.01% | 2.60% | 135 |
| 11 | hurst_gate | lcb | rejected (trade starvation) | 0.143 | -0.692 | 0.078 | -0.324 | 1.832 | 0.258 | 0.00% | -0.30% | 0.27% | 3 |
| 11 | hurst_gate | point | rejected (trade starvation) | 0.185 | -0.650 | 0.075 | -0.327 | 1.286 | -0.288 | 0.09% | -0.21% | 0.71% | 18 |
| 12 | arma_z_filter | - | BLOCKED (Pesaran-Timmermann not significant) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Retained modules:** multi_stage_exits, regime_gate (regime model: hmm2)

Pesaran-Timmermann test of the ARMA forecast sign (H=3d, n=897): statistic 1.588, p-value 0.0561.


## Vehicles: fixed-prior baseline on the planted-edge data

Full sample, default parameters, no tuning. Spreads are each vehicle's preset, expressed at the reference index level. The ETF track takes long signals only and carries EURUSD on its notional.

Risk per trade 0.25% (EUR 12.50):

| Vehicle | Orders | Rejected: below minimum lot | Rejected: short | Trades | Average leverage | Net P&L (EUR) | Spread, slippage, commission, conversion | of which conversion | Financing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Capital.com EU leveraged, 0.001 lot, EUR account | 621 | 0 | 0 | 320 | 0.23x | 1114.98 | 125.81 | 40.23 | 181.34 |
| Capital.com EU leveraged, 0.001 lot, USD sub-account | 621 | 0 | 0 | 320 | 0.23x | 1163.63 | 85.92 | 0.00 | 182.35 |
| Capital.com EU 1:1, 0.001 lot, EUR account, no funding | 621 | 0 | 0 | 320 | 0.23x | 1324.54 | 126.15 | 39.47 | 0.00 |
| Pepperstone EU cTrader Razor, 0.10 lot | 14 | 1238 | 0 | 8 | 0.46x | 94.35 | 3.32 | 0.00 | 0.28 |
| IBKR IBUST100, 1 unit | 0 | 1280 | 0 | 0 | 0.00x | 0.00 | 0.00 | 0.00 | 0.00 |
| UCITS ETF, long only | 286 | 0 | 707 | 148 | 0.17x | 26.46 | 388.82 | 0.00 | 0.00 |

Risk per trade 0.35% (EUR 17.50):

| Vehicle | Orders | Rejected: below minimum lot | Rejected: short | Trades | Average leverage | Net P&L (EUR) | Spread, slippage, commission, conversion | of which conversion | Financing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Capital.com EU leveraged, 0.001 lot, EUR account | 621 | 0 | 0 | 320 | 0.32x | 1539.78 | 180.47 | 57.64 | 262.90 |
| Capital.com EU leveraged, 0.001 lot, USD sub-account | 621 | 0 | 0 | 320 | 0.32x | 1601.74 | 123.48 | 0.00 | 264.55 |
| Capital.com EU 1:1, 0.001 lot, EUR account, no funding | 621 | 0 | 0 | 320 | 0.32x | 1899.17 | 183.42 | 57.45 | 0.00 |
| Pepperstone EU cTrader Razor, 0.10 lot | 151 | 940 | 0 | 80 | 0.44x | 540.53 | 33.88 | 0.00 | 46.85 |
| IBKR IBUST100, 1 unit | 0 | 1280 | 0 | 0 | 0.00x | 0.00 | 0.00 | 0.00 | 0.00 |
| UCITS ETF, long only | 286 | 0 | 707 | 148 | 0.25x | 108.43 | 428.58 | 0.00 | 0.00 |

Risk per trade 0.50% (EUR 25.00):

| Vehicle | Orders | Rejected: below minimum lot | Rejected: short | Trades | Average leverage | Net P&L (EUR) | Spread, slippage, commission, conversion | of which conversion | Financing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Capital.com EU leveraged, 0.001 lot, EUR account | 621 | 0 | 0 | 320 | 0.44x | 2068.73 | 251.80 | 80.16 | 365.57 |
| Capital.com EU leveraged, 0.001 lot, USD sub-account | 621 | 0 | 0 | 320 | 0.44x | 2197.97 | 174.84 | 0.00 | 373.89 |
| Capital.com EU 1:1, 0.001 lot, EUR account, no funding | 621 | 0 | 0 | 320 | 0.44x | 2515.86 | 258.74 | 80.79 | 0.00 |
| Pepperstone EU cTrader Razor, 0.10 lot | 359 | 514 | 0 | 186 | 0.43x | 1241.68 | 80.23 | 0.00 | 117.74 |
| IBKR IBUST100, 1 unit | 0 | 1280 | 0 | 0 | 0.00x | 0.00 | 0.00 | 0.00 | 0.00 |
| UCITS ETF, long only | 286 | 0 | 707 | 148 | 0.18x | -207.23 | 391.09 | 0.00 | 0.00 |

## Entry order: stop-limit (default) against a plain stop

Fixed-prior baseline, full sample, default preset. A plain stop has no limit leg: when a bar opens beyond the limit (trigger + 0.15 ATR) but inside the anti-chase bound (trigger + 0.50 ATR) it is filled at the open, where the stop-limit waits for a pullback or expires. Everything above uses the stop-limit.

No-edge data:

| Entry order | Orders | Cancelled (anti-chase) | Expired unfilled | Opens beyond limit | Trades | Fills beyond limit | Average excess (ATR) | Expectancy (R) | Profit factor | Sharpe | Net P&L (EUR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stop_limit | 549 | 235 | 30 | 174 | 284 | 0 | 0.000 | 0.001 | 1.01 | 0.03 | 21.39 |
| stop | 521 | 191 | 24 | 162 | 306 | 162 | 0.173 | 0.018 | 1.05 | 0.10 | 113.18 |

Planted-edge data:

| Entry order | Orders | Cancelled (anti-chase) | Expired unfilled | Opens beyond limit | Trades | Fills beyond limit | Average excess (ATR) | Expectancy (R) | Profit factor | Sharpe | Net P&L (EUR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stop_limit | 621 | 270 | 32 | 190 | 319 | 0 | 0.000 | 0.317 | 1.51 | 0.79 | 1111.64 |
| stop | 583 | 227 | 22 | 173 | 334 | 173 | 0.180 | 0.344 | 1.58 | 0.91 | 1366.31 |

One history cannot separate an effect from path noise, because the two order types take different trades. Paired over independent histories (plain stop minus stop-limit on the same history, fixed-prior baseline, full sample):

| Planted strength | Histories | Mean expectancy, stop-limit (R) | Mean expectancy, plain stop (R) | Mean difference (R) | Standard error | Histories where the plain stop did better | Mean trades, stop-limit / plain stop |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40 | -0.058 | -0.066 | -0.007 | 0.004 | 17 of 40 | 280 / 295 |
| 2 | 40 | 0.388 | 0.377 | -0.011 | 0.005 | 14 of 40 | 307 / 324 |

Full tearsheets: `null_tearsheet.md`, `planted_tearsheet.md`.
