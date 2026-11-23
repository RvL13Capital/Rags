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
| backtests evaluated in search | 2595 |
| configurations compared (DSR / PBO trials) | 64 |
| instrument | NDX-microCFD-0.01 |
| initial equity | 5,000 EUR |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 0.03% | 1.06% | -0.25% | n/a | n/a |
| 2. Annualised volatility | 1.03% | 0.95% | 0.91% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 0.03 | 1.12 | -0.26 | n/a | n/a |
| 4. Sortino ratio | 0.04 | 2.03 | -0.38 | n/a | n/a |
| 5. Calmar ratio | 0.01 | 0.77 | -0.06 | n/a | n/a |
| 6. Maximum drawdown | 2.41% | 1.38% | 3.81% | n/a | n/a |
| 7. Profit factor (net) | 1.01 | 1.82 | 0.87 | n/a | n/a |
| 8. Win rate | 44.32% | 53.85% | 40.20% | n/a | n/a |
| 9. Avg win / avg loss | 1.27 | 1.56 | 1.30 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 0.06 | 3.02 | -0.65 | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.003 | 0.305 | -0.060 | n/a | n/a |
| 11. Independent trades | 273 | 208 | 199 | n/a | n/a |
| 12. Annualised turnover (x equity) | 8.8 | 7.9 | 8.1 | n/a | n/a |
| 13a. Transaction costs (EUR) | 65.02 | 51.24 | 47.96 | n/a | n/a |
| 13b. Financing costs (EUR) | 21.46 | 16.53 | 10.20 | n/a | n/a |
| 14a. ES 97.5% daily | 0.23% | 0.18% | 0.21% | n/a | n/a |
| 14b. ES 99% daily | 0.32% | 0.24% | 0.28% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.00% | 0.18% | 0.16% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.00% | 0.02% | 0.02% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.00% | 0.01% | 0.01% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.00% | 0.01% | 0.01% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.998 | 1.003 | 0.991 | n/a | n/a |
| 15f. MC worst 99% drawdown | 2.85% | 4.81% | 5.38% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 162 | 137 | 202 | n/a | n/a |
| Net P&L (EUR) | 16.79 | 627.29 | -129.49 | n/a | n/a |
| Sample length (years) | 13.18 | 11.18 | 10.68 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | -0.2645 | > 0.80 | FAIL |
| Deflated Sharpe Ratio | 0.0142 | > 0.95 | FAIL |
| PBO (CSCV) | 0.5657 | < 0.25 | FAIL |
| OOS net profit factor | 0.8708 | > 1.20 | FAIL |
| OOS net P&L under 1.5x costs | -176.3008 | > 0 | FAIL |
| OOS maximum drawdown | 0.0381 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0002 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.4524 | >= 70% | FAIL |
| Independent OOS trades | 199 | > 200 | FAIL |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- Deflated Sharpe Ratio: **0.0142** (per-period SR -0.0167 vs expected maximum under the null SR0 0.0259, N = 64 trials, cross-trial mean -0.0002, std 0.0110). With the null centred at zero instead (original Bailey / Lopez de Prado form, SR0 0.0260) it is 0.0139; the gate uses the first figure.
- PBO (CSCV, 12870 splits of 64 configurations): **0.566**; IS->OOS Sharpe slope -1.013; P(OOS loss of IS-best) 0.542

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L -176.30 EUR, Sharpe -0.37, profit factor 0.82, max drawdown 4.21%

## Walk-forward folds (42 folds, 45.2% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2016-02-03 .. 2016-08-02 | 0.422 | 0.69 | -0.36 | -10.18 | 8 |
| 1 | 2016-05-02 .. 2016-11-01 | 0.816 | 1.34 | -1.79 | -40.36 | 8 |
| 2 | 2016-08-03 .. 2017-02-02 | 1.166 | 1.34 | -0.71 | -9.44 | 13 |
| 3 | 2016-11-03 .. 2017-05-02 | 1.531 | 1.77 | 0.01 | 0.12 | 19 |
| 4 | 2017-02-02 .. 2017-08-01 | 0.826 | 0.99 | 0.41 | 12.59 | 9 |
| 5 | 2017-05-02 .. 2017-11-01 | 0.031 | 0.16 | -0.10 | -1.63 | 13 |
| 6 | 2017-08-02 .. 2018-02-01 | 0.307 | 0.42 | -0.43 | -12.73 | 6 |
| 7 | 2017-11-01 .. 2018-04-30 | 1.320 | 1.39 | -0.68 | -19.29 | 12 |
| 8 | 2018-02-01 .. 2018-07-31 | 0.621 | 0.70 | -2.01 | -23.97 | 4 |
| 9 | 2018-05-01 .. 2018-10-31 | 0.375 | 0.56 | 1.15 | 34.34 | 13 |
| 10 | 2018-08-01 .. 2019-01-31 | 1.358 | 1.42 | 1.19 | 28.34 | 10 |
| 11 | 2018-11-01 .. 2019-04-30 | 1.429 | 1.56 | -0.14 | -3.37 | 9 |
| 12 | 2019-02-04 .. 2019-08-02 | 0.944 | 0.95 | 0.61 | 16.08 | 9 |
| 13 | 2019-05-01 .. 2019-10-31 | 1.558 | 1.76 | 0.40 | 10.82 | 7 |
| 14 | 2019-08-01 .. 2020-01-31 | 1.611 | 1.98 | -1.12 | -28.08 | 8 |
| 15 | 2019-11-01 .. 2020-04-30 | 1.329 | 1.25 | 0.90 | 24.24 | 8 |
| 16 | 2020-02-03 .. 2020-07-31 | 0.869 | 0.96 | 1.43 | 46.00 | 11 |
| 17 | 2020-05-01 .. 2020-10-30 | 0.921 | 1.33 | -1.26 | -29.79 | 24 |
| 18 | 2020-08-03 .. 2021-02-02 | 1.102 | 1.32 | -2.99 | -52.32 | 6 |
| 19 | 2020-11-02 .. 2021-04-30 | 0.641 | 0.82 | -1.28 | -36.50 | 9 |
| 20 | 2021-02-02 .. 2021-07-30 | 0.473 | 0.68 | 2.33 | 56.61 | 7 |
| 21 | 2021-05-03 .. 2021-11-02 | 0.370 | 0.67 | 1.82 | 47.14 | 14 |
| 22 | 2021-08-03 .. 2022-02-02 | 1.725 | 2.23 | 1.41 | 26.03 | 4 |
| 23 | 2021-11-03 .. 2022-05-02 | 1.725 | 2.14 | -1.42 | -22.97 | 5 |
| 24 | 2022-02-02 .. 2022-08-01 | 1.979 | 2.83 | -1.14 | -25.51 | 8 |
| 25 | 2022-05-03 .. 2022-11-02 | 1.892 | 2.66 | 0.67 | 17.78 | 8 |
| 26 | 2022-08-03 .. 2023-02-02 | 1.309 | 1.49 | 1.82 | 50.99 | 8 |
| 27 | 2022-11-02 .. 2023-05-01 | 1.613 | 1.99 | 2.81 | 67.09 | 7 |
| 28 | 2023-02-01 .. 2023-07-31 | 1.377 | 1.36 | 1.04 | 25.93 | 6 |
| 29 | 2023-05-02 .. 2023-11-01 | 1.731 | 2.32 | 1.20 | 25.35 | 12 |
| 30 | 2023-08-02 .. 2024-02-01 | 1.981 | 2.84 | 1.15 | 20.22 | 13 |
| 31 | 2023-11-01 .. 2024-04-30 | 1.963 | 2.98 | 0.43 | 8.31 | 7 |
| 32 | 2024-02-01 .. 2024-07-31 | 1.888 | 2.85 | -2.35 | -73.62 | 17 |
| 33 | 2024-05-01 .. 2024-10-31 | 1.405 | 1.50 | -1.32 | -20.85 | 12 |
| 34 | 2024-08-01 .. 2025-01-31 | -0.051 | 0.04 | -1.42 | -22.98 | 5 |
| 35 | 2024-11-01 .. 2025-04-30 | -0.176 | -0.15 | -0.78 | -11.37 | 14 |
| 36 | 2025-02-04 .. 2025-08-01 | -0.299 | -0.25 | -0.94 | -17.46 | 11 |
| 37 | 2025-05-01 .. 2025-10-31 | -0.283 | -0.33 | -1.28 | -26.35 | 7 |
| 38 | 2025-08-01 .. 2026-01-30 | 0.776 | 1.11 | -0.90 | -22.30 | 13 |
| 39 | 2025-11-03 .. 2026-05-01 | 0.023 | 0.22 | -1.60 | -26.08 | 6 |
| 40 | 2026-02-03 .. 2026-07-31 | -0.125 | -0.04 | 1.30 | 15.85 | 1 |
| 41 | 2026-05-01 .. 2026-10-30 | -0.196 | -0.17 | -0.77 | -16.42 | 8 |
