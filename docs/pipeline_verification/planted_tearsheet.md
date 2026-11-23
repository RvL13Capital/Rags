# PIPELINE VERIFICATION - planted-edge synthetic data - tearsheet

> **PIPELINE VERIFICATION ON SYNTHETIC DATA.** Every number below comes from a simulated
> price process and is not empirical market performance.

**Verdict:** PIPELINE VERIFICATION ONLY - synthetic data. No empirical conclusion about the NASDAQ-100 can be drawn. NOT READY FOR LIVE DEPLOYMENT.

Data sources: SYNTHETIC-PIPELINE-VERIFICATION

| Run setting | Value |
|---|---|
| modules | regime_gate |
| regime model | hmm2 |
| walk-forward | 36m train / 6m val / 6m OOS, step 3m, purge 5d + embargo 5d |
| folds | 44 |
| Optuna trials per fold | 40 |
| backtests evaluated in search | 3067 |
| DSR trial count N | 560 |
| PBO / DSR configurations | 64 |
| instrument | NDX-CFD |
| initial equity | 5,000 EUR |
| ablation variants evaluated | 14 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 0.26% | 1.84% | 0.55% | n/a | n/a |
| 2. Annualised volatility | 0.91% | 1.05% | 1.02% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 0.29 | 1.75 | 0.55 | n/a | n/a |
| 4. Sortino ratio | 0.45 | 3.97 | 0.87 | n/a | n/a |
| 5. Calmar ratio | 0.15 | 2.84 | 0.23 | n/a | n/a |
| 6. Maximum drawdown | 1.70% | 0.65% | 2.40% | n/a | n/a |
| 7. Profit factor (net) | 1.26 | 3.58 | 1.42 | n/a | n/a |
| 8. Win rate | 39.60% | 61.15% | 47.34% | n/a | n/a |
| 9. Avg win / avg loss | 1.92 | 2.28 | 1.58 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 1.20 | 8.14 | 1.88 | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.116 | 0.727 | 0.181 | n/a | n/a |
| 11. Independent trades | 149 | 139 | 169 | n/a | n/a |
| 12. Annualised turnover (x equity) | 5.9 | 6.6 | 8.0 | n/a | n/a |
| 13a. Transaction costs (EUR) | 46.40 | 46.79 | 52.95 | n/a | n/a |
| 13b. Financing costs (EUR) | 21.19 | 35.05 | 31.53 | n/a | n/a |
| 14a. ES 97.5% daily | 0.20% | 0.16% | 0.22% | n/a | n/a |
| 14b. ES 99% daily | 0.31% | 0.24% | 0.32% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.13% | 0.15% | 0.21% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.03% | 0.02% | 0.08% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.01% | 0.01% | 0.03% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.01% | 0.01% | 0.01% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.999 | 1.010 | 1.001 | n/a | n/a |
| 15f. MC worst 99% drawdown | 4.56% | 4.27% | 5.20% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 155 | 99 | 148 | n/a | n/a |
| Net P&L (EUR) | 178.95 | 1,132.09 | 317.97 | n/a | n/a |
| Sample length (years) | 13.68 | 11.18 | 11.18 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.5456 | > 0.80 | FAIL |
| Deflated Sharpe Ratio | 0.5282 | > 0.95 | FAIL |
| PBO (CSCV) | 0.8938 | < 0.25 | FAIL |
| OOS net profit factor | 1.4245 | > 1.20 | PASS |
| OOS net P&L under 1.5x costs | 217.7266 | > 0 | PASS |
| OOS maximum drawdown | 0.0240 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0008 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.5682 | >= 70% | FAIL |
| Independent OOS trades | 169 | > 200 | FAIL |

## Overfitting diagnostics

- Deflated Sharpe Ratio: **0.5282** (per-period SR 0.0344 vs expected maximum under the null SR0 0.0331, N = 560 trials, cross-trial mean 0.0102, std 0.0074)
- PBO (CSCV, 12870 splits of 64 configurations): **0.894**; IS->OOS Sharpe slope -1.119; P(OOS loss of IS-best) 0.546

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 217.73 EUR, Sharpe 0.39, profit factor 1.28, max drawdown 2.71%

## Walk-forward folds (44 folds, 56.8% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2015-08-03 .. 2016-02-02 | 0.931 | 0.87 | -0.47 | -8.38 | 13 |
| 1 | 2015-11-02 .. 2016-04-29 | 0.593 | 0.68 | 0.69 | 23.55 | 9 |
| 2 | 2016-02-03 .. 2016-08-02 | 1.561 | 1.92 | 3.17 | 25.91 | 2 |
| 3 | 2016-05-04 .. 2016-11-03 | 2.115 | 3.33 | -0.78 | -25.89 | 10 |
| 4 | 2016-08-02 .. 2017-02-01 | 1.422 | 1.92 | 0.43 | 15.92 | 17 |
| 5 | 2016-11-01 .. 2017-04-28 | 0.337 | 0.46 | 2.28 | 122.84 | 12 |
| 6 | 2017-02-02 .. 2017-08-01 | 1.793 | 2.32 | 2.55 | 122.79 | 13 |
| 7 | 2017-05-02 .. 2017-11-01 | 1.978 | 2.78 | 1.09 | 34.59 | 10 |
| 8 | 2017-08-01 .. 2018-01-31 | 2.265 | 3.64 | 2.22 | 29.88 | 2 |
| 9 | 2017-10-31 .. 2018-04-27 | 2.063 | 3.02 | 0.82 | 16.48 | 6 |
| 10 | 2018-02-01 .. 2018-07-31 | 1.883 | 2.57 | -0.13 | -0.23 | 1 |
| 11 | 2018-05-02 .. 2018-11-01 | 1.635 | 2.20 | 2.40 | 41.33 | 1 |
| 12 | 2018-08-01 .. 2019-01-31 | 2.224 | 3.81 | 0.21 | 4.07 | 4 |
| 13 | 2018-10-31 .. 2019-04-29 | 1.737 | 2.32 | 1.66 | 33.07 | 6 |
| 14 | 2019-02-04 .. 2019-08-02 | 0.476 | 0.49 | -1.41 | -1.22 | 1 |
| 15 | 2019-05-02 .. 2019-11-01 | 1.649 | 2.07 | -1.24 | -25.35 | 4 |
| 16 | 2019-07-31 .. 2020-01-30 | 1.456 | 1.53 | -0.18 | -3.95 | 10 |
| 17 | 2019-10-31 .. 2020-04-29 | 1.274 | 1.23 | -0.33 | -4.35 | 11 |
| 18 | 2020-02-04 .. 2020-08-03 | 0.288 | 0.44 | 1.28 | 18.85 | 8 |
| 19 | 2020-05-04 .. 2020-11-03 | -0.517 | -0.41 | -1.56 | -5.68 | 1 |
| 20 | 2020-07-31 .. 2021-01-29 | 0.968 | 0.79 | -1.43 | -3.40 | 1 |
| 21 | 2020-11-02 .. 2021-04-30 | 0.507 | 0.52 | -0.11 | -1.61 | 9 |
| 22 | 2021-02-03 .. 2021-08-02 | -1.472 | -1.16 | -0.18 | -7.60 | 16 |
| 23 | 2021-05-04 .. 2021-11-03 | 1.300 | 1.12 | 0.90 | 17.05 | 5 |
| 24 | 2021-08-02 .. 2022-02-01 | 1.584 | 1.85 | 0.29 | 8.04 | 7 |
| 25 | 2021-11-02 .. 2022-04-29 | 1.466 | 1.83 | 0.05 | 1.40 | 7 |
| 26 | 2022-02-03 .. 2022-08-02 | 1.751 | 2.38 | -0.71 | -10.82 | 8 |
| 27 | 2022-05-03 .. 2022-11-02 | 1.514 | 1.68 | 0.67 | 24.57 | 13 |
| 28 | 2022-08-01 .. 2023-01-31 | 1.237 | 1.18 | 1.59 | 75.06 | 13 |
| 29 | 2022-11-01 .. 2023-04-28 | 1.669 | 2.03 | 1.68 | 71.83 | 8 |
| 30 | 2023-02-02 .. 2023-08-01 | 2.173 | 3.42 | -1.11 | -35.75 | 5 |
| 31 | 2023-05-02 .. 2023-11-01 | 2.271 | 3.72 | -1.21 | -27.27 | 3 |
| 32 | 2023-08-01 .. 2024-01-31 | 2.044 | 3.19 | 0.00 | -0.10 | 9 |
| 33 | 2023-10-31 .. 2024-04-29 | 0.570 | 0.51 | -2.59 | -55.57 | 8 |
| 34 | 2024-02-01 .. 2024-07-31 | 1.150 | 1.12 | -2.10 | -23.08 | 5 |
| 35 | 2024-05-02 .. 2024-11-01 | 0.829 | 0.91 | -2.25 | -26.99 | 6 |
| 36 | 2024-08-01 .. 2025-01-31 | -1.186 | -1.52 | 0.23 | 2.86 | 19 |
| 37 | 2024-10-31 .. 2025-04-29 | -0.613 | -0.56 | 1.05 | 15.33 | 17 |
| 38 | 2025-02-04 .. 2025-08-01 | 1.664 | 2.31 | 1.03 | 25.66 | 6 |
| 39 | 2025-05-02 .. 2025-10-31 | 2.083 | 3.09 | 0.01 | 0.06 | 7 |
| 40 | 2025-07-31 .. 2026-01-30 | 1.542 | 1.86 | 1.01 | 14.73 | 9 |
| 41 | 2025-10-31 .. 2026-04-29 | 1.319 | 1.20 | 2.95 | 29.59 | 7 |
| 42 | 2026-02-03 .. 2026-07-31 | 1.269 | 1.21 | 1.69 | 16.02 | 8 |
| 43 | 2026-05-04 .. 2026-10-30 | 1.966 | 3.08 | -1.08 | -8.54 | 5 |
