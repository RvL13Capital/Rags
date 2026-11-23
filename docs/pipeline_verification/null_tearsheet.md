# PIPELINE VERIFICATION - null (no-edge) synthetic data - tearsheet

> **PIPELINE VERIFICATION ON SYNTHETIC DATA.** Every number below comes from a simulated
> price process and is not empirical market performance.

**Verdict:** PIPELINE VERIFICATION ONLY - synthetic data. No empirical conclusion about the NASDAQ-100 can be drawn. NOT READY FOR LIVE DEPLOYMENT.

Data sources: SYNTHETIC-PIPELINE-VERIFICATION

| Run setting | Value |
|---|---|
| modules | multi_stage_exits, trend_flip_exit |
| regime model | n/a |
| walk-forward | 36m train / 12m val / 6m OOS, step 3m, purge 5d + embargo 5d |
| folds | 42 |
| tuned parameters | donchian_window, atr_stop_mult, max_hold_days |
| Optuna trials per fold | 40 |
| backtests evaluated in search | 2583 |
| configurations compared (K) | 64 |
| instrument | Capital.com-EU-US100-0.001 |
| entry order | stop_limit |
| initial equity | 5,000 EUR |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | -0.01% | 1.05% | -0.05% | n/a | n/a |
| 2. Annualised volatility | 0.96% | 0.93% | 0.91% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | -0.00 | 1.13 | -0.05 | n/a | n/a |
| 4. Sortino ratio | -0.01 | 1.94 | -0.08 | n/a | n/a |
| 5. Calmar ratio | -0.00 | 0.85 | -0.02 | n/a | n/a |
| 6. Maximum drawdown | 1.85% | 1.24% | 3.46% | n/a | n/a |
| 7. Profit factor (net) | 1.00 | 1.80 | 0.97 | n/a | n/a |
| 8. Win rate | 43.03% | 56.94% | 44.62% | n/a | n/a |
| 9. Avg win / avg loss | 1.32 | 1.36 | 1.20 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | -0.02 | 2.96 | -0.14 | n/a | n/a |
| 10b. Net expectancy / trade (R) | -0.001 | 0.279 | -0.015 | n/a | n/a |
| 11. Independent trades | 244 | 209 | 195 | n/a | n/a |
| 12. Annualised turnover (x equity) | 7.9 | 7.6 | 7.2 | n/a | n/a |
| 13a. Transaction costs (EUR) | 73.52 | 66.31 | 56.37 | n/a | n/a |
| 13b. Financing costs (EUR) | 38.62 | 37.36 | 27.81 | n/a | n/a |
| 14a. ES 97.5% daily | 0.22% | 0.19% | 0.22% | n/a | n/a |
| 14b. ES 99% daily | 0.28% | 0.25% | 0.28% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.22% | 0.18% | 0.20% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.02% | 0.01% | 0.02% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.01% | 0.01% | 0.01% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.01% | 0.01% | 0.01% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.995 | 1.005 | 0.996 | n/a | n/a |
| 15f. MC worst 99% drawdown | 5.70% | 4.71% | 5.32% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 181 | 124 | 172 | n/a | n/a |
| Net P&L (EUR) | -5.43 | 619.25 | -27.86 | n/a | n/a |
| Sample length (years) | 13.18 | 11.18 | 10.68 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | -0.0532 | > 0.80 | FAIL |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.0774 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | 0.8291 | < 15% | FAIL |
| CSCV rank PBO | 0.5183 | < 0.25 when tuned and IS Sharpe IQR > 0.25 | FAIL |
| OOS net profit factor | 0.9701 | > 1.20 | FAIL |
| OOS net P&L under 1.5x costs | -73.5149 | > 0 | FAIL |
| OOS maximum drawdown (true high-water mark) | 0.0346 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0002 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.4048 | >= 65% | FAIL |
| Independent OOS trades | 195 | >= 150 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** DSR, zero-mean null: **0.0774** (per-period SR -0.0033 vs SR0 0.0241; K = 64 configurations compared, average pairwise correlation 0.63, N_eff = 24.2; cross-trial Sharpe std 0.0121)
- Informational. DSR with the cross-trial mean in the benchmark and all K trials: 0.1777 (SR0 0.0145, cross-trial mean -0.0143). Probabilistic Sharpe ratio against zero (N_eff = 1): 0.4310.
- CSCV over 12870 splits of the 64 configurations compared: P(OOS loss of the in-sample winner) **0.829**; IS->OOS Sharpe slope -0.914
- Rank PBO over 64 parameter draws of this configuration: 0.518; in-sample Sharpe IQR across the draws 0.36

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L -73.51 EUR, Sharpe -0.15, profit factor 0.92, max drawdown 3.53%

## Walk-forward folds (42 folds, 40.5% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2016-02-03 .. 2016-08-02 | 0.470 | 0.71 | 0.84 | 23.27 | 6 |
| 1 | 2016-05-02 .. 2016-11-01 | 1.663 | 2.08 | -0.16 | -3.43 | 6 |
| 2 | 2016-08-03 .. 2017-02-02 | 1.985 | 3.04 | 0.34 | 4.18 | 4 |
| 3 | 2016-11-03 .. 2017-05-02 | 2.163 | 3.48 | -0.22 | -5.16 | 8 |
| 4 | 2017-02-02 .. 2017-08-01 | 1.736 | 2.40 | -0.19 | -3.88 | 10 |
| 5 | 2017-05-02 .. 2017-11-01 | 1.086 | 1.14 | -2.59 | -30.48 | 7 |
| 6 | 2017-08-02 .. 2018-02-01 | 0.000 | 0.14 | -0.99 | -24.77 | 14 |
| 7 | 2017-11-01 .. 2018-04-30 | 1.089 | 1.17 | -0.74 | -17.70 | 10 |
| 8 | 2018-02-01 .. 2018-07-31 | 0.518 | 0.61 | 0.39 | 10.46 | 9 |
| 9 | 2018-05-01 .. 2018-10-31 | 0.104 | 0.24 | -0.34 | -8.30 | 9 |
| 10 | 2018-08-01 .. 2019-01-31 | 0.951 | 1.08 | -1.87 | -26.83 | 3 |
| 11 | 2018-11-01 .. 2019-04-30 | 1.147 | 1.31 | 0.53 | 12.89 | 7 |
| 12 | 2019-02-04 .. 2019-08-02 | 1.083 | 1.40 | 0.97 | 24.46 | 7 |
| 13 | 2019-05-01 .. 2019-10-31 | 1.380 | 1.60 | -0.86 | -20.33 | 6 |
| 14 | 2019-08-01 .. 2020-01-31 | 1.295 | 1.52 | -0.50 | -15.52 | 7 |
| 15 | 2019-11-01 .. 2020-04-30 | 0.919 | 1.11 | 0.53 | 15.85 | 13 |
| 16 | 2020-02-03 .. 2020-07-31 | 1.603 | 1.95 | -0.16 | -3.34 | 17 |
| 17 | 2020-05-01 .. 2020-10-30 | 1.784 | 2.66 | -1.21 | -21.50 | 16 |
| 18 | 2020-08-03 .. 2021-02-02 | 1.246 | 1.25 | -0.52 | -7.44 | 6 |
| 19 | 2020-11-02 .. 2021-04-30 | 1.110 | 1.36 | -0.84 | -13.39 | 19 |
| 20 | 2021-02-02 .. 2021-07-30 | 0.836 | 0.97 | 0.40 | 9.87 | 9 |
| 21 | 2021-05-03 .. 2021-11-02 | 0.128 | 0.22 | 2.86 | 85.86 | 11 |
| 22 | 2021-08-03 .. 2022-02-02 | 0.452 | 0.73 | 2.45 | 72.20 | 8 |
| 23 | 2021-11-03 .. 2022-05-02 | 0.488 | 0.72 | 0.86 | 24.84 | 11 |
| 24 | 2022-02-02 .. 2022-08-01 | 1.733 | 2.30 | -0.75 | -12.79 | 5 |
| 25 | 2022-05-03 .. 2022-11-02 | 1.615 | 1.93 | -0.10 | -2.70 | 11 |
| 26 | 2022-08-03 .. 2023-02-02 | 1.471 | 1.56 | 0.54 | 15.03 | 10 |
| 27 | 2022-11-02 .. 2023-05-01 | 1.506 | 1.93 | 3.10 | 67.79 | 7 |
| 28 | 2023-02-01 .. 2023-07-31 | 0.595 | 0.88 | 2.80 | 62.81 | 9 |
| 29 | 2023-05-02 .. 2023-11-01 | 1.525 | 1.83 | 0.49 | 9.60 | 6 |
| 30 | 2023-08-02 .. 2024-02-01 | 1.848 | 2.79 | -0.48 | -10.36 | 12 |
| 31 | 2023-11-01 .. 2024-04-30 | 1.831 | 2.53 | 0.87 | 15.18 | 3 |
| 32 | 2024-02-01 .. 2024-07-31 | 1.688 | 2.17 | -0.55 | -11.14 | 11 |
| 33 | 2024-05-01 .. 2024-10-31 | 0.994 | 1.19 | -2.08 | -39.39 | 7 |
| 34 | 2024-08-01 .. 2025-01-31 | 0.523 | 0.68 | -3.15 | -72.21 | 6 |
| 35 | 2024-11-01 .. 2025-04-30 | 0.057 | 0.22 | -1.94 | -47.78 | 6 |
| 36 | 2025-02-04 .. 2025-08-01 | -0.174 | -0.15 | -1.14 | -29.83 | 11 |
| 37 | 2025-05-01 .. 2025-10-31 | -0.189 | -0.01 | -1.70 | -23.63 | 11 |
| 38 | 2025-08-01 .. 2026-01-30 | 0.479 | 0.80 | -1.83 | -29.74 | 12 |
| 39 | 2025-11-03 .. 2026-05-01 | 0.142 | 0.26 | -0.22 | -3.56 | 15 |
| 40 | 2026-02-03 .. 2026-07-31 | -0.211 | -0.15 | 0.28 | 3.33 | 12 |
| 41 | 2026-05-01 .. 2026-10-30 | 0.093 | 0.25 | 0.57 | 15.64 | 7 |
