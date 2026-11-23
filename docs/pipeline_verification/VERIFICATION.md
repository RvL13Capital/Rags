# Pipeline verification report

> **PIPELINE VERIFICATION - SYNTHETIC DATA.** These results exercise the code on simulated prices.
> They are not empirical market performance and say nothing about the NASDAQ-100.

Synthetic sample 2012-2026; walk-forward 36m train / 12m validation / 6m OOS; 40 Optuna trials per fold where anything is tuned; 10000 Monte Carlo paths; lots normalised to index level 25,288.25; runtime 164s.

| Data set | OOS Sharpe | OOS PF | OOS max DD | OOS trades | OOS net P&L (EUR) | DSR (spec) | DSR (zero-mean null) | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Null (martingale), full geometry tuned | -0.26 | 0.87 | 3.81% | 199 | -129.49 | 0.014 | 0.014 | 0.566 |
| Planted edge x2 (after ablation, thresholds only) | 0.96 | 1.75 | 2.65% | 189 | 963.38 | 0.099 | 0.913 | 0.727 |

## Checks

- Null data rejected by the release gates: **PASS** (failed gates: OOS net Sharpe, Deflated Sharpe Ratio, PBO (CSCV), OOS net profit factor, OOS net P&L under 1.5x costs, Profitable walk-forward OOS folds, Independent OOS trades)
- Planted edge clearly visible out of sample (Sharpe at least 0.5 above null, positive net P&L, profit factor above 1.2): **PASS**
- Sizing per vehicle at EUR 5,000 (whole-unit IBUST100 never sizes, 0.01-lot CFD does): **PASS**

## Power of the release gates

The positive control is a real edge by construction, yet it fails 4 of 9 gates: Deflated Sharpe Ratio, PBO (CSCV), Profitable walk-forward OOS folds, Independent OOS trades. With about 18 trades a year the specified battery cannot confirm an edge of this size on roughly ten OOS years. Two of the failures are structural: the specified DSR adds the cross-trial mean to its benchmark, so sibling variants that all share the edge raise the bar to their own level; and CSCV-PBO measures whether the in-sample ranking of the compared variants persists, which is close to a coin flip when the variants are near-identical.

## How strong must an edge be to be seen?

Fixed-prior baseline, no tuning, stitched walk-forward OOS, same seed. Strength 1 is a drift of 0.12% a day inside trend regimes against roughly 0.9% daily volatility; the positive control above uses strength 2.

| Planted strength | Trend drift / day | OOS Sharpe | OOS PF | Expectancy (R) | Trades | Net P&L (EUR) | Profitable folds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | -0.05 | 0.96 | -0.051 | 178 | -47.12 | 45% |
| 1 | 0.12% | 0.05 | 1.02 | 0.029 | 187 | 33.65 | 48% |
| 2 | 0.24% | 0.96 | 1.75 | 0.477 | 189 | 963.38 | 69% |
| 3 | 0.36% | 1.62 | 2.40 | 0.865 | 206 | 2078.50 | 74% |

## Planted-edge ablation

# Sequential ablation (stitched walk-forward OOS)

Retention floor on trade count: 94. Variants evaluated: 15.

| Step | Module | Variant | Status | OOS Sharpe | d Sharpe | Calmar | d Calmar | PF | d PF | ES99 | d ES99 | Max DD | Trades |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | - | BASELINE | 0.963 | n/a | 0.627 | n/a | 1.755 | n/a | 0.43% | n/a | 2.65% | 189 |
| 1 | multi_stage_exits | - | rejected (no OOS net gain) | 0.901 | -0.061 | 0.418 | -0.209 | 1.553 | -0.202 | 0.36% | -0.07% | 2.75% | 206 |
| 2 | trend_flip_exit | - | rejected (no OOS net gain) | 0.963 | 0.000 | 0.627 | 0.000 | 1.755 | 0.000 | 0.43% | 0.00% | 2.65% | 189 |
| 3 | regime_gate | rv20 | rejected (parsimony benchmark) | 0.884 | -0.079 | 0.507 | -0.121 | 1.802 | 0.047 | 0.44% | 0.01% | 2.76% | 143 |
| 3 | regime_gate | hmm2 | rejected | 1.081 | 0.119 | 0.488 | -0.140 | 2.154 | 0.399 | 0.39% | -0.04% | 3.15% | 124 |
| 3 | regime_gate | hmm3 | rejected (parsimony benchmark) | 0.280 | -0.682 | 0.086 | -0.541 | 1.334 | -0.421 | 0.35% | -0.08% | 2.89% | 64 |
| 4 | garch_vol | - | rejected (no OOS net gain) | 0.810 | -0.153 | 0.483 | -0.145 | 1.633 | -0.122 | 0.41% | -0.02% | 2.65% | 167 |
| 5 | evt_es_cap | - | rejected (no OOS net gain) | 0.791 | -0.172 | 0.578 | -0.050 | 1.705 | -0.050 | 0.27% | -0.16% | 1.32% | 173 |
| 6 | spread_gate | - | rejected (no OOS net gain) | 0.922 | -0.040 | 0.487 | -0.140 | 1.774 | 0.019 | 0.41% | -0.01% | 3.07% | 166 |
| 7 | vpin_filter | - | rejected (no OOS net gain) | 1.011 | 0.049 | 0.691 | 0.064 | 1.834 | 0.079 | 0.42% | -0.01% | 2.48% | 183 |
| 8 | event_blackout | - | rejected (no OOS net gain) | 0.959 | -0.004 | 0.620 | -0.007 | 1.747 | -0.008 | 0.42% | -0.01% | 2.65% | 189 |
| 9 | score_sizing | - | rejected (no OOS net gain) | 0.963 | 0.000 | 0.627 | 0.000 | 1.755 | 0.000 | 0.43% | 0.00% | 2.65% | 189 |
| 10 | kelly_cap | - | rejected (no OOS net gain) | 0.920 | -0.043 | 0.514 | -0.113 | 1.997 | 0.242 | 0.40% | -0.02% | 2.48% | 116 |
| 11 | hurst_gate | lcb | rejected (trade starvation) | 0.138 | -0.825 | 0.056 | -0.571 | 1.599 | -0.156 | 0.00% | -0.43% | 0.65% | 3 |
| 11 | hurst_gate | point | rejected (trade starvation) | 0.148 | -0.815 | 0.072 | -0.556 | 1.292 | -0.462 | 0.21% | -0.22% | 1.20% | 26 |
| 12 | arma_z_filter | - | BLOCKED (Pesaran-Timmermann not significant) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Retained modules:** none (baseline only)

Pesaran-Timmermann test of the ARMA forecast sign (H=3d, n=897): statistic 1.588, p-value 0.0561.


## Vehicles: fixed-prior baseline on the planted-edge data

Full sample, default parameters, no tuning. Spreads are each vehicle's preset, expressed at the reference index level. The ETF track takes long signals only and carries EURUSD on its notional.

Risk per trade 0.25% (EUR 12.50):

| Vehicle | Signals | Orders | Rejected: below minimum lot | Rejected: short | Trades | Net P&L (EUR) | Transaction costs | Financing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Micro CFD, 0.01 lot | 1204 | 638 | 2 | 0 | 265 | 1722.36 | 77.91 | 110.49 |
| Micro CFD, 0.1 lot | 1204 | 290 | 645 | 0 | 121 | 744.91 | 37.25 | 53.00 |
| IBKR IBUST100, 1 unit | 1204 | 0 | 1204 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| UCITS ETF, long only | 1204 | 300 | 0 | 684 | 119 | 155.96 | 324.85 | 0.00 |

Risk per trade 0.50% (EUR 25.00):

| Vehicle | Signals | Orders | Rejected: below minimum lot | Rejected: short | Trades | Net P&L (EUR) | Transaction costs | Financing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Micro CFD, 0.01 lot | 1204 | 640 | 0 | 0 | 265 | 3684.37 | 169.35 | 251.40 |
| Micro CFD, 0.1 lot | 1204 | 457 | 359 | 0 | 186 | 1304.44 | 91.08 | 95.24 |
| IBKR IBUST100, 1 unit | 1204 | 0 | 1204 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| UCITS ETF, long only | 1204 | 300 | 0 | 684 | 119 | 421.46 | 398.34 | 0.00 |

Full tearsheets: `null_tearsheet.md`, `planted_tearsheet.md`.
