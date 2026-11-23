# PIPELINE VERIFICATION - null (no-edge) synthetic data - tearsheet

> **PIPELINE VERIFICATION ON SYNTHETIC DATA.** Every number below comes from a simulated
> price process and is not empirical market performance.

**Verdict:** PIPELINE VERIFICATION ONLY - synthetic data. No empirical conclusion about the NASDAQ-100 can be drawn. NOT READY FOR LIVE DEPLOYMENT.

Data sources: SYNTHETIC-PIPELINE-VERIFICATION

| Run setting | Value |
|---|---|
| modules | multi_stage_exits, trend_flip_exit |
| regime model | n/a |
| walk-forward | 36m train / 6m val / 6m OOS, step 3m, purge 5d + embargo 5d |
| folds | 44 |
| Optuna trials per fold | 40 |
| backtests evaluated in search | 2718 |
| DSR trial count N | 64 |
| PBO / DSR configurations | 64 |
| instrument | NDX-CFD |
| initial equity | 5,000 EUR |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | -0.20% | 1.64% | -0.36% | n/a | n/a |
| 2. Annualised volatility | 0.99% | 1.02% | 1.02% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | -0.20 | 1.60 | -0.35 | n/a | n/a |
| 4. Sortino ratio | -0.29 | 3.10 | -0.51 | n/a | n/a |
| 5. Calmar ratio | -0.04 | 1.59 | -0.07 | n/a | n/a |
| 6. Maximum drawdown | 4.83% | 1.03% | 5.25% | n/a | n/a |
| 7. Profit factor (net) | 0.90 | 2.40 | 0.84 | n/a | n/a |
| 8. Win rate | 45.26% | 62.26% | 38.37% | n/a | n/a |
| 9. Avg win / avg loss | 1.09 | 1.45 | 1.35 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | -0.48 | 4.71 | -0.81 | n/a | n/a |
| 10b. Net expectancy / trade (R) | -0.033 | 0.426 | -0.088 | n/a | n/a |
| 11. Independent trades | 285 | 212 | 245 | n/a | n/a |
| 12. Annualised turnover (x equity) | 9.3 | 8.2 | 10.2 | n/a | n/a |
| 13a. Transaction costs (EUR) | 68.23 | 53.97 | 62.78 | n/a | n/a |
| 13b. Financing costs (EUR) | 30.18 | 24.35 | 18.08 | n/a | n/a |
| 14a. ES 97.5% daily | 0.23% | 0.19% | 0.23% | n/a | n/a |
| 14b. ES 99% daily | 0.30% | 0.25% | 0.30% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.22% | 0.18% | 0.25% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.03% | 0.02% | 0.04% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.02% | 0.01% | 0.02% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.01% | 0.01% | 0.01% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.993 | 1.009 | 0.989 | n/a | n/a |
| 15f. MC worst 99% drawdown | 5.77% | 4.90% | 6.25% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 191 | 113 | 205 | n/a | n/a |
| Net P&L (EUR) | -137.30 | 999.30 | -199.16 | n/a | n/a |
| Sample length (years) | 13.68 | 11.18 | 11.18 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | -0.3503 | > 0.80 | FAIL |
| Deflated Sharpe Ratio | 0.0044 | > 0.95 | FAIL |
| PBO (CSCV) | 0.5677 | < 0.25 | FAIL |
| OOS net profit factor | 0.8377 | > 1.20 | FAIL |
| OOS net P&L under 1.5x costs | -248.1907 | > 0 | FAIL |
| OOS maximum drawdown | 0.0525 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0004 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.5000 | >= 70% | FAIL |
| Independent OOS trades | 245 | > 200 | PASS |

## Overfitting diagnostics

- Deflated Sharpe Ratio: **0.0044** (per-period SR -0.0221 vs expected maximum under the null SR0 0.0277, N = 64 trials, cross-trial mean 0.0023, std 0.0107)
- PBO (CSCV, 12870 splits of 64 configurations): **0.568**; IS->OOS Sharpe slope -0.898; P(OOS loss of IS-best) 0.481

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L -248.19 EUR, Sharpe -0.45, profit factor 0.80, max drawdown 5.91%

## Walk-forward folds (44 folds, 50.0% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2015-08-03 .. 2016-02-02 | 0.172 | 0.33 | 1.24 | 38.22 | 13 |
| 1 | 2015-11-02 .. 2016-04-29 | -0.619 | -0.65 | 0.38 | 12.39 | 6 |
| 2 | 2016-02-03 .. 2016-08-02 | 0.966 | 1.30 | -0.00 | -0.34 | 14 |
| 3 | 2016-05-04 .. 2016-11-03 | 2.484 | 4.29 | -2.05 | -64.01 | 12 |
| 4 | 2016-08-02 .. 2017-02-01 | 1.786 | 2.51 | -1.46 | -20.95 | 11 |
| 5 | 2016-11-01 .. 2017-04-28 | 0.245 | 0.43 | 0.06 | 0.98 | 19 |
| 6 | 2017-02-02 .. 2017-08-01 | 1.626 | 2.07 | 0.39 | 12.70 | 9 |
| 7 | 2017-05-02 .. 2017-11-01 | 2.004 | 3.08 | 0.06 | 1.92 | 16 |
| 8 | 2017-08-01 .. 2018-01-31 | 1.326 | 1.31 | -2.42 | -6.73 | 3 |
| 9 | 2017-10-31 .. 2018-04-27 | 1.171 | 1.11 | -0.88 | -13.07 | 17 |
| 10 | 2018-02-01 .. 2018-07-31 | 1.530 | 1.93 | 0.28 | 7.76 | 11 |
| 11 | 2018-05-02 .. 2018-11-01 | 0.624 | 0.65 | 1.50 | 45.97 | 10 |
| 12 | 2018-08-01 .. 2019-01-31 | 1.127 | 1.24 | -0.06 | -1.84 | 12 |
| 13 | 2018-10-31 .. 2019-04-29 | 1.927 | 2.80 | -0.75 | -16.55 | 5 |
| 14 | 2019-02-04 .. 2019-08-02 | 1.843 | 2.63 | 0.57 | 18.35 | 9 |
| 15 | 2019-05-02 .. 2019-11-01 | 1.296 | 1.28 | -0.10 | -3.18 | 12 |
| 16 | 2019-07-31 .. 2020-01-30 | 1.581 | 1.98 | -1.15 | -31.08 | 8 |
| 17 | 2019-10-31 .. 2020-04-29 | 1.590 | 2.29 | 0.48 | 7.02 | 15 |
| 18 | 2020-02-04 .. 2020-08-03 | 0.924 | 0.98 | 0.56 | 18.78 | 14 |
| 19 | 2020-05-04 .. 2020-11-03 | 1.447 | 1.62 | 0.60 | 18.56 | 12 |
| 20 | 2020-07-31 .. 2021-01-29 | 1.542 | 1.91 | -0.92 | -15.24 | 4 |
| 21 | 2020-11-02 .. 2021-04-30 | 0.637 | 0.92 | 0.27 | 5.59 | 12 |
| 22 | 2021-02-03 .. 2021-08-02 | 1.195 | 1.16 | 2.61 | 71.42 | 7 |
| 23 | 2021-05-04 .. 2021-11-03 | 1.391 | 1.40 | 2.05 | 48.75 | 3 |
| 24 | 2021-08-02 .. 2022-02-01 | 2.122 | 3.36 | 1.14 | 26.33 | 27 |
| 25 | 2021-11-02 .. 2022-04-29 | 2.201 | 3.42 | -0.11 | -2.60 | 8 |
| 26 | 2022-02-03 .. 2022-08-02 | 2.076 | 3.22 | 0.08 | 1.39 | 6 |
| 27 | 2022-05-03 .. 2022-11-02 | 2.059 | 3.26 | 0.98 | 25.86 | 7 |
| 28 | 2022-08-01 .. 2023-01-31 | 0.901 | 1.08 | 0.65 | 21.29 | 13 |
| 29 | 2022-11-01 .. 2023-04-28 | 1.369 | 1.34 | 3.51 | 76.28 | 6 |
| 30 | 2023-02-02 .. 2023-08-01 | 1.921 | 2.76 | 2.65 | 43.48 | 15 |
| 31 | 2023-05-02 .. 2023-11-01 | 2.412 | 4.00 | -0.15 | -4.30 | 10 |
| 32 | 2023-08-01 .. 2024-01-31 | 2.284 | 3.75 | 0.52 | 11.84 | 11 |
| 33 | 2023-10-31 .. 2024-04-29 | 1.758 | 2.53 | 0.31 | 7.87 | 9 |
| 34 | 2024-02-01 .. 2024-07-31 | 1.708 | 2.18 | -2.02 | -41.40 | 11 |
| 35 | 2024-05-02 .. 2024-11-01 | 1.715 | 2.24 | -2.70 | -64.31 | 9 |
| 36 | 2024-08-01 .. 2025-01-31 | -0.501 | -0.43 | -2.25 | -42.08 | 8 |
| 37 | 2024-10-31 .. 2025-04-29 | -0.757 | -1.00 | -0.89 | -13.64 | 14 |
| 38 | 2025-02-04 .. 2025-08-01 | 1.168 | 0.97 | -0.96 | -18.49 | 11 |
| 39 | 2025-05-02 .. 2025-10-31 | 1.440 | 1.95 | -2.68 | -39.42 | 7 |
| 40 | 2025-07-31 .. 2026-01-30 | 1.286 | 1.44 | -1.56 | -35.75 | 10 |
| 41 | 2025-10-31 .. 2026-04-29 | -0.582 | -0.75 | -1.55 | -22.84 | 4 |
| 42 | 2026-02-03 .. 2026-07-31 | -0.095 | 0.05 | -1.08 | -37.07 | 12 |
| 43 | 2026-05-04 .. 2026-10-30 | 1.179 | 1.10 | -0.96 | -26.82 | 8 |
