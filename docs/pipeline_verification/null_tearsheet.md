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
| backtests evaluated in search | 2602 |
| configurations compared (K) | 64 |
| instrument | Capital.com-EU-US100-0.001 |
| initial equity | 5,000 EUR |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | -0.07% | 1.01% | -0.29% | n/a | n/a |
| 2. Annualised volatility | 1.07% | 0.97% | 0.93% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | -0.06 | 1.04 | -0.30 | n/a | n/a |
| 4. Sortino ratio | -0.08 | 1.88 | -0.44 | n/a | n/a |
| 5. Calmar ratio | -0.03 | 0.67 | -0.07 | n/a | n/a |
| 6. Maximum drawdown | 2.25% | 1.50% | 4.29% | n/a | n/a |
| 7. Profit factor (net) | 0.97 | 1.75 | 0.85 | n/a | n/a |
| 8. Win rate | 45.39% | 54.50% | 40.72% | n/a | n/a |
| 9. Avg win / avg loss | 1.16 | 1.46 | 1.23 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | -0.17 | 2.97 | -0.78 | n/a | n/a |
| 10b. Net expectancy / trade (R) | -0.014 | 0.283 | -0.074 | n/a | n/a |
| 11. Independent trades | 271 | 200 | 194 | n/a | n/a |
| 12. Annualised turnover (x equity) | 9.4 | 8.2 | 8.5 | n/a | n/a |
| 13a. Transaction costs (EUR) | 87.66 | 68.17 | 62.32 | n/a | n/a |
| 13b. Financing costs (EUR) | 43.38 | 31.94 | 22.13 | n/a | n/a |
| 14a. ES 97.5% daily | 0.25% | 0.19% | 0.22% | n/a | n/a |
| 14b. ES 99% daily | 0.34% | 0.25% | 0.29% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.21% | 0.19% | 0.17% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.03% | 0.02% | 0.02% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.01% | 0.01% | 0.01% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.01% | 0.01% | 0.01% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.995 | 1.002 | 0.991 | n/a | n/a |
| 15f. MC worst 99% drawdown | 5.84% | 5.09% | 5.51% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 181 | 141 | 201 | n/a | n/a |
| Net P&L (EUR) | -45.41 | 594.70 | -151.87 | n/a | n/a |
| Sample length (years) | 13.18 | 11.18 | 10.68 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | -0.3043 | > 0.80 | FAIL |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.0190 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | 0.5669 | < 15% | FAIL |
| CSCV rank PBO | 0.5246 | < 0.25 when tuned and IS Sharpe IQR > 0.25 | FAIL |
| OOS net profit factor | 0.8471 | > 1.20 | FAIL |
| OOS net P&L under 1.5x costs | -221.1343 | > 0 | FAIL |
| OOS maximum drawdown (true high-water mark) | 0.0429 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0002 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.4286 | >= 65% | FAIL |
| Independent OOS trades | 194 | >= 150 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** DSR, zero-mean null: **0.0190** (per-period SR -0.0192 vs SR0 0.0211; K = 64 configurations compared, average pairwise correlation 0.64, N_eff = 23.7; cross-trial Sharpe std 0.0107)
- Informational. DSR with the cross-trial mean in the benchmark and all K trials: 0.0154 (SR0 0.0227, cross-trial mean -0.0026). Probabilistic Sharpe ratio against zero (N_eff = 1): 0.1615.
- CSCV over 12870 splits of the 64 configurations compared: P(OOS loss of the in-sample winner) **0.567**; IS->OOS Sharpe slope -1.002
- Rank PBO over 64 parameter draws of this configuration: 0.525; in-sample Sharpe IQR across the draws 0.31

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L -221.13 EUR, Sharpe -0.46, profit factor 0.78, max drawdown 4.74%

## Walk-forward folds (42 folds, 42.9% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2016-02-03 .. 2016-08-02 | 0.339 | 0.59 | -0.49 | -15.31 | 8 |
| 1 | 2016-05-02 .. 2016-11-01 | 0.774 | 1.29 | -1.49 | -39.65 | 10 |
| 2 | 2016-08-03 .. 2017-02-02 | 1.141 | 1.25 | -0.66 | -9.36 | 14 |
| 3 | 2016-11-03 .. 2017-05-02 | 1.511 | 1.72 | 0.01 | 0.05 | 19 |
| 4 | 2017-02-02 .. 2017-08-01 | 0.727 | 0.85 | 0.27 | 8.51 | 9 |
| 5 | 2017-05-02 .. 2017-11-01 | 0.015 | 0.15 | -0.01 | -0.17 | 13 |
| 6 | 2017-08-02 .. 2018-02-01 | 0.282 | 0.39 | -0.51 | -15.91 | 6 |
| 7 | 2017-11-01 .. 2018-04-30 | 1.194 | 1.30 | -0.95 | -29.29 | 12 |
| 8 | 2018-02-01 .. 2018-07-31 | 0.546 | 0.63 | -2.01 | -25.04 | 4 |
| 9 | 2018-05-01 .. 2018-10-31 | 0.243 | 0.31 | 1.16 | 22.13 | 16 |
| 10 | 2018-08-01 .. 2019-01-31 | 1.318 | 1.30 | 0.56 | 13.92 | 9 |
| 11 | 2018-11-01 .. 2019-04-30 | 1.331 | 1.31 | -0.92 | -21.61 | 8 |
| 12 | 2019-02-04 .. 2019-08-02 | 0.997 | 1.35 | 1.80 | 48.88 | 4 |
| 13 | 2019-05-01 .. 2019-10-31 | 1.551 | 1.77 | -0.14 | -4.22 | 8 |
| 14 | 2019-08-01 .. 2020-01-31 | 1.698 | 2.21 | -0.89 | -24.65 | 8 |
| 15 | 2019-11-01 .. 2020-04-30 | 1.131 | 1.06 | 0.34 | 10.10 | 9 |
| 16 | 2020-02-03 .. 2020-07-31 | 0.790 | 0.88 | 2.22 | 75.15 | 10 |
| 17 | 2020-05-01 .. 2020-10-30 | 0.875 | 1.43 | -0.10 | -2.34 | 18 |
| 18 | 2020-08-03 .. 2021-02-02 | 1.106 | 1.32 | -1.89 | -39.95 | 7 |
| 19 | 2020-11-02 .. 2021-04-30 | 0.626 | 0.75 | 0.67 | 16.81 | 8 |
| 20 | 2021-02-02 .. 2021-07-30 | 0.909 | 0.95 | 1.53 | 31.53 | 6 |
| 21 | 2021-05-03 .. 2021-11-02 | 0.688 | 0.94 | 1.31 | 33.65 | 13 |
| 22 | 2021-08-03 .. 2022-02-02 | 1.677 | 2.10 | 1.60 | 32.43 | 4 |
| 23 | 2021-11-03 .. 2022-05-02 | 1.559 | 1.73 | -1.35 | -22.94 | 5 |
| 24 | 2022-02-02 .. 2022-08-01 | 1.966 | 2.79 | -1.25 | -29.12 | 8 |
| 25 | 2022-05-03 .. 2022-11-02 | 1.867 | 2.61 | 0.51 | 13.98 | 8 |
| 26 | 2022-08-03 .. 2023-02-02 | 1.068 | 1.41 | 0.87 | 18.26 | 9 |
| 27 | 2022-11-02 .. 2023-05-01 | 1.587 | 1.96 | 3.08 | 80.14 | 8 |
| 28 | 2023-02-01 .. 2023-07-31 | 1.198 | 1.23 | 1.04 | 27.59 | 6 |
| 29 | 2023-05-02 .. 2023-11-01 | 1.685 | 2.26 | 1.12 | 24.81 | 12 |
| 30 | 2023-08-02 .. 2024-02-01 | 1.958 | 2.81 | 0.50 | 11.54 | 9 |
| 31 | 2023-11-01 .. 2024-04-30 | 1.980 | 2.93 | -0.18 | -3.88 | 8 |
| 32 | 2024-02-01 .. 2024-07-31 | 1.997 | 3.07 | -3.19 | -95.99 | 15 |
| 33 | 2024-05-01 .. 2024-10-31 | 1.381 | 1.45 | -1.27 | -21.18 | 12 |
| 34 | 2024-08-01 .. 2025-01-31 | -0.136 | -0.08 | -1.84 | -28.36 | 4 |
| 35 | 2024-11-01 .. 2025-04-30 | -0.199 | -0.19 | -0.94 | -14.24 | 14 |
| 36 | 2025-02-04 .. 2025-08-01 | -0.308 | -0.26 | -2.14 | -33.52 | 10 |
| 37 | 2025-05-01 .. 2025-10-31 | -0.317 | -0.39 | -1.32 | -28.77 | 7 |
| 38 | 2025-08-01 .. 2026-01-30 | 0.751 | 1.10 | -1.08 | -28.11 | 13 |
| 39 | 2025-11-03 .. 2026-05-01 | -0.022 | 0.16 | -1.56 | -26.36 | 6 |
| 40 | 2026-02-03 .. 2026-07-31 | -0.242 | -0.25 | 1.04 | 24.63 | 4 |
| 41 | 2026-05-01 .. 2026-10-30 | -0.384 | -0.53 | -1.52 | -37.27 | 8 |
