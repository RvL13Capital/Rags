# Pipeline verification report

> **PIPELINE VERIFICATION - SYNTHETIC DATA.** These results exercise the code on simulated prices.
> They are not empirical market performance and say nothing about the NASDAQ-100.

Synthetic sample 2012-2026; walk-forward 36m train / 12m validation / 6m OOS; 40 Optuna trials per fold where anything is tuned; 10000 Monte Carlo paths; lots normalised to index level 25,288.25; runtime 244s.

| Data set | OOS Sharpe | OOS PF | OOS max DD | OOS trades | OOS net P&L (EUR) | DSR (gate) | N_eff | DSR (cross-trial mean, info) | CSCV P(OOS loss) | Rank PBO | Gates failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Null (martingale), full geometry tuned | -0.30 | 0.85 | 4.29% | 194 | -151.87 | 0.019 | 23.7 | 0.015 | 0.567 | 0.525 | 7 of 10 |
| Planted edge x2 (after ablation, thresholds only) | 0.90 | 1.69 | 2.71% | 190 | 942.92 | 0.959 | 6.0 | 0.111 | 0.071 | n/a | 0 of 10 |

## Checks

- Null data rejected by the release gates: **PASS** (failed gates: OOS net Sharpe, Deflated Sharpe Ratio (zero-mean null, N_eff), CSCV P(OOS loss of IS-best), CSCV rank PBO, OOS net profit factor, OOS net P&L under 1.5x costs, Profitable walk-forward OOS folds)
- Planted edge clearly visible out of sample (Sharpe at least 0.5 above null, positive net P&L, profit factor above 1.2): **PASS**
- Sizing per vehicle at EUR 5,000 (whole-unit IBUST100 never sizes, the 0.001-lot CFD does): **PASS**

## Operating characteristics of the release gates

On this run the positive control fails 0 of 10 gates. One history says little, so the fixed-prior baseline was also run on independent synthetic histories (different seeds, nothing tuned, K = 1), each through the complete battery:

| Planted strength | Histories | Pass every gate | Median OOS Sharpe | Gates failed most often |
|---:|---:|---:|---:|---|
| 0 | 40 | 0 | -0.25 | OOS net Sharpe (40); Profitable walk-forward OOS folds (40); Deflated Sharpe Ratio (zero-mean null, N_eff) (38) |
| 2 | 40 | 17 | 0.86 | Profitable walk-forward OOS folds (19); OOS net Sharpe (16); Deflated Sharpe Ratio (zero-mean null, N_eff) (7) |

The first row estimates the false-positive rate of the battery, the second its power against an edge of this size. Thresholds: Sharpe > 0.80, DSR > 0.95, P(OOS loss) < 15%, profit factor > 1.20, positive under 1.5x costs, max drawdown < 10%, Monte Carlo P(DD >= 20%) < 2%, profitable folds >= 65%, trades >= 150.

## How strong must an edge be to be seen?

Fixed-prior baseline, no tuning, stitched walk-forward OOS, same seed. Strength 1 is a drift of 0.12% a day inside trend regimes against roughly 0.9% daily volatility; the positive control above uses strength 2.

| Planted strength | Trend drift / day | OOS Sharpe | OOS PF | Expectancy (R) | Trades | Net P&L (EUR) | Profitable folds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | -0.08 | 0.95 | -0.067 | 179 | -71.36 | 45% |
| 1 | 0.12% | 0.01 | 1.00 | 0.012 | 188 | 1.90 | 48% |
| 2 | 0.24% | 0.90 | 1.69 | 0.452 | 190 | 942.92 | 69% |
| 3 | 0.36% | 1.53 | 2.27 | 0.828 | 206 | 2029.19 | 69% |

## Planted-edge ablation

# Sequential ablation (stitched walk-forward OOS)

Retention floor on trade count: 95. Variants evaluated: 15.

| Step | Module | Variant | Status | OOS Sharpe | d Sharpe | Calmar | d Calmar | PF | d PF | ES99 | d ES99 | Max DD | Trades |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | - | BASELINE | 0.903 | n/a | 0.602 | n/a | 1.691 | n/a | 0.44% | n/a | 2.71% | 190 |
| 1 | multi_stage_exits | - | rejected (no OOS net gain) | 0.840 | -0.062 | 0.406 | -0.196 | 1.503 | -0.188 | 0.39% | -0.06% | 2.77% | 207 |
| 2 | trend_flip_exit | - | rejected (no OOS net gain) | 0.903 | 0.000 | 0.602 | 0.000 | 1.691 | 0.000 | 0.44% | 0.00% | 2.71% | 190 |
| 3 | regime_gate | rv20 | rejected (parsimony benchmark) | 0.822 | -0.080 | 0.448 | -0.155 | 1.727 | 0.037 | 0.45% | 0.01% | 3.01% | 143 |
| 3 | regime_gate | hmm2 | rejected | 1.009 | 0.106 | 0.442 | -0.161 | 2.040 | 0.350 | 0.40% | -0.04% | 3.35% | 124 |
| 3 | regime_gate | hmm3 | rejected (parsimony benchmark) | 0.250 | -0.652 | 0.076 | -0.526 | 1.292 | -0.398 | 0.37% | -0.08% | 3.04% | 64 |
| 4 | garch_vol | - | rejected (no OOS net gain) | 0.761 | -0.142 | 0.468 | -0.134 | 1.586 | -0.105 | 0.43% | -0.01% | 2.71% | 168 |
| 5 | evt_es_cap | - | rejected (no OOS net gain) | 0.755 | -0.148 | 0.549 | -0.054 | 1.647 | -0.044 | 0.30% | -0.14% | 1.47% | 190 |
| 6 | spread_gate | - | rejected (no OOS net gain) | 0.851 | -0.052 | 0.431 | -0.171 | 1.696 | 0.006 | 0.44% | -0.00% | 3.36% | 167 |
| 7 | vpin_filter | - | rejected (no OOS net gain) | 0.944 | 0.041 | 0.628 | 0.025 | 1.757 | 0.067 | 0.44% | -0.00% | 2.67% | 184 |
| 8 | event_blackout | - | rejected (no OOS net gain) | 0.896 | -0.007 | 0.594 | -0.008 | 1.681 | -0.009 | 0.44% | -0.01% | 2.71% | 190 |
| 9 | score_sizing | - | rejected (no OOS net gain) | 0.903 | 0.000 | 0.602 | 0.000 | 1.691 | 0.000 | 0.44% | 0.00% | 2.71% | 190 |
| 10 | kelly_cap | - | rejected (no OOS net gain) | 0.940 | 0.037 | 0.560 | -0.042 | 1.964 | 0.273 | 0.43% | -0.01% | 2.53% | 121 |
| 11 | hurst_gate | lcb | rejected (trade starvation) | 0.132 | -0.770 | 0.053 | -0.549 | 1.569 | -0.122 | 0.00% | -0.44% | 0.69% | 3 |
| 11 | hurst_gate | point | rejected (trade starvation) | 0.152 | -0.751 | 0.081 | -0.522 | 1.277 | -0.413 | 0.23% | -0.22% | 1.18% | 27 |
| 12 | arma_z_filter | - | BLOCKED (Pesaran-Timmermann not significant) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Retained modules:** none (baseline only)

Pesaran-Timmermann test of the ARMA forecast sign (H=3d, n=897): statistic 1.588, p-value 0.0561.


## Vehicles: fixed-prior baseline on the planted-edge data

Full sample, default parameters, no tuning. Spreads are each vehicle's preset, expressed at the reference index level. The ETF track takes long signals only and carries EURUSD on its notional.

Risk per trade 0.25% (EUR 12.50):

| Vehicle | Orders | Rejected: below minimum lot | Rejected: short | Trades | Average leverage | Net P&L (EUR) | Spread, slippage, commission, conversion | of which conversion | Financing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Capital.com EU leveraged, 0.001 lot, EUR account | 644 | 0 | 0 | 263 | 0.28x | 1640.68 | 128.46 | 40.38 | 179.13 |
| Capital.com EU leveraged, 0.001 lot, USD sub-account | 644 | 0 | 0 | 263 | 0.28x | 1686.83 | 88.39 | 0.00 | 179.96 |
| Capital.com EU 1:1, 0.001 lot, EUR account, no funding | 644 | 0 | 0 | 263 | 0.28x | 1841.68 | 128.58 | 39.52 | 0.00 |
| Pepperstone EU cTrader Razor, 0.10 lot | 290 | 645 | 0 | 121 | 0.30x | 746.10 | 36.06 | 0.00 | 53.00 |
| IBKR IBUST100, 1 unit | 0 | 1204 | 0 | 0 | 0.00x | 0.00 | 0.00 | 0.00 | 0.00 |
| UCITS ETF, long only | 300 | 0 | 684 | 119 | 0.21x | 155.96 | 324.85 | 0.00 | 0.00 |

Risk per trade 0.35% (EUR 17.50):

| Vehicle | Orders | Rejected: below minimum lot | Rejected: short | Trades | Average leverage | Net P&L (EUR) | Spread, slippage, commission, conversion | of which conversion | Financing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Capital.com EU leveraged, 0.001 lot, EUR account | 644 | 0 | 0 | 263 | 0.39x | 2389.45 | 188.31 | 59.24 | 266.84 |
| Capital.com EU leveraged, 0.001 lot, USD sub-account | 644 | 0 | 0 | 263 | 0.39x | 2462.89 | 129.73 | 0.00 | 268.70 |
| Capital.com EU 1:1, 0.001 lot, EUR account, no funding | 644 | 0 | 0 | 263 | 0.39x | 2712.63 | 189.34 | 58.28 | 0.00 |
| Pepperstone EU cTrader Razor, 0.10 lot | 519 | 231 | 0 | 213 | 0.33x | 1693.59 | 75.21 | 0.00 | 111.05 |
| IBKR IBUST100, 1 unit | 0 | 1204 | 0 | 0 | 0.00x | 0.00 | 0.00 | 0.00 | 0.00 |
| UCITS ETF, long only | 300 | 0 | 684 | 119 | 0.31x | 345.77 | 368.29 | 0.00 | 0.00 |

Risk per trade 0.50% (EUR 25.00):

| Vehicle | Orders | Rejected: below minimum lot | Rejected: short | Trades | Average leverage | Net P&L (EUR) | Spread, slippage, commission, conversion | of which conversion | Financing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Capital.com EU leveraged, 0.001 lot, EUR account | 644 | 0 | 0 | 263 | 0.52x | 3419.95 | 270.05 | 85.05 | 387.72 |
| Capital.com EU leveraged, 0.001 lot, USD sub-account | 644 | 0 | 0 | 263 | 0.53x | 3557.86 | 187.13 | 0.00 | 393.82 |
| Capital.com EU 1:1, 0.001 lot, EUR account, no funding | 644 | 0 | 0 | 263 | 0.52x | 3876.05 | 271.59 | 83.93 | 0.00 |
| Pepperstone EU cTrader Razor, 0.10 lot | 457 | 359 | 0 | 186 | 0.44x | 1348.82 | 89.11 | 0.00 | 99.63 |
| IBKR IBUST100, 1 unit | 0 | 1204 | 0 | 0 | 0.00x | 0.00 | 0.00 | 0.00 | 0.00 |
| UCITS ETF, long only | 300 | 0 | 684 | 119 | 0.41x | 525.62 | 418.11 | 0.00 | 0.00 |

Full tearsheets: `null_tearsheet.md`, `planted_tearsheet.md`.
