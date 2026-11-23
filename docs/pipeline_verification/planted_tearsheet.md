# PIPELINE VERIFICATION - planted-edge synthetic data - tearsheet

> **PIPELINE VERIFICATION ON SYNTHETIC DATA.** Every number below comes from a simulated
> price process and is not empirical market performance.

**Verdict:** PIPELINE VERIFICATION ONLY - synthetic data. No empirical conclusion about the NASDAQ-100 can be drawn. NOT READY FOR LIVE DEPLOYMENT.

Data sources: SYNTHETIC-PIPELINE-VERIFICATION

| Run setting | Value |
|---|---|
| modules | baseline only |
| regime model | n/a |
| walk-forward | 36m train / 12m val / 6m OOS, step 3m, purge 5d + embargo 5d |
| folds | 42 |
| tuned parameters | none (fixed canonical priors) |
| Optuna trials per fold | 0 |
| backtests evaluated in search | 42 |
| configurations compared (K) | 15 |
| instrument | Pepperstone-EU-cTrader-NAS100-0.01 |
| initial equity | 5,000 EUR |
| ablation variants evaluated | 15 |
| largest search space in any variant | 1 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 1.97% | 1.58% | 1.66% | n/a | n/a |
| 2. Annualised volatility | 1.75% | 1.72% | 1.73% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 1.13 | 0.92 | 0.96 | n/a | n/a |
| 4. Sortino ratio | 1.94 | 1.53 | 1.59 | n/a | n/a |
| 5. Calmar ratio | 0.72 | 0.58 | 0.63 | n/a | n/a |
| 6. Maximum drawdown | 2.74% | 2.74% | 2.65% | n/a | n/a |
| 7. Profit factor (net) | 1.94 | 1.73 | 1.75 | n/a | n/a |
| 8. Win rate | 43.72% | 39.20% | 39.68% | n/a | n/a |
| 9. Avg win / avg loss | 2.49 | 2.68 | 2.67 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 6.36 | 4.82 | 5.10 | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.586 | 0.453 | 0.477 | n/a | n/a |
| 11. Independent trades | 231 | 199 | 189 | n/a | n/a |
| 12. Annualised turnover (x equity) | 9.4 | 9.2 | 9.2 | n/a | n/a |
| 13a. Transaction costs (EUR) | 82.07 | 63.99 | 61.66 | n/a | n/a |
| 13b. Financing costs (EUR) | 76.60 | 90.01 | 96.03 | n/a | n/a |
| 14a. ES 97.5% daily | 0.33% | 0.32% | 0.33% | n/a | n/a |
| 14b. ES 99% daily | 0.43% | 0.43% | 0.43% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.65% | 0.70% | 0.71% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.14% | 0.15% | 0.16% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.05% | 0.05% | 0.06% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.02% | 0.02% | 0.03% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 1.008 | 1.005 | 1.006 | n/a | n/a |
| 15f. MC worst 99% drawdown | 8.51% | 8.84% | 8.90% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 130 | 138 | 135 | n/a | n/a |
| Net P&L (EUR) | 1,468.11 | 960.01 | 963.38 | n/a | n/a |
| Sample length (years) | 13.18 | 11.18 | 10.68 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.9627 | > 0.80 | PASS |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.9683 | > 0.95 | PASS |
| CSCV P(OOS loss of IS-best) | 0.0633 | < 15% | PASS |
| CSCV rank PBO | n/a | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (nothing tuned) |
| OOS net profit factor | 1.7550 | > 1.20 | PASS |
| OOS net P&L under 1.5x costs | 834.3389 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0265 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0016 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.6905 | >= 65% | PASS |
| Independent OOS trades | 189 | >= 150 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** DSR, zero-mean null: **0.9683** (per-period SR 0.0606 vs SR0 0.0261; K = 15 configurations compared, average pairwise correlation 0.64, N_eff = 6.1; cross-trial Sharpe std 0.0200)
- Informational. DSR with the cross-trial mean in the benchmark and all K trials: 0.0993 (SR0 0.0846, cross-trial mean 0.0493). Probabilistic Sharpe ratio against zero (N_eff = 1): 0.9994.
- CSCV over 12870 splits of the 15 configurations compared: P(OOS loss of the in-sample winner) **0.063**; IS->OOS Sharpe slope -0.779

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 834.34 EUR, Sharpe 0.87, profit factor 1.66, max drawdown 2.72%

## Walk-forward folds (42 folds, 69.0% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2016-02-03 .. 2016-08-02 | 0.009 | 0.07 | -1.66 | -80.99 | 8 |
| 1 | 2016-05-02 .. 2016-11-01 | 0.322 | 0.41 | -1.67 | -85.06 | 10 |
| 2 | 2016-08-03 .. 2017-02-02 | -0.409 | -0.55 | 0.87 | 51.38 | 15 |
| 3 | 2016-11-03 .. 2017-05-02 | -0.184 | -0.18 | 1.92 | 122.63 | 13 |
| 4 | 2017-02-02 .. 2017-08-01 | -0.246 | -0.27 | 2.69 | 126.91 | 7 |
| 5 | 2017-05-02 .. 2017-11-01 | -0.259 | -0.29 | 2.53 | 74.44 | 7 |
| 6 | 2017-08-02 .. 2018-02-01 | 1.518 | 1.64 | 0.28 | 9.89 | 11 |
| 7 | 2017-11-01 .. 2018-04-30 | 1.644 | 1.95 | -1.63 | -46.03 | 10 |
| 8 | 2018-02-01 .. 2018-07-31 | 1.319 | 1.49 | -0.37 | -9.95 | 9 |
| 9 | 2018-05-01 .. 2018-10-31 | 1.101 | 1.28 | 1.12 | 49.78 | 9 |
| 10 | 2018-08-01 .. 2019-01-31 | 0.100 | 0.17 | 0.50 | 19.66 | 6 |
| 11 | 2018-11-01 .. 2019-04-30 | 0.208 | 0.26 | -0.29 | -6.15 | 3 |
| 12 | 2019-02-04 .. 2019-08-02 | 0.118 | 0.17 | 2.33 | 93.01 | 5 |
| 13 | 2019-05-01 .. 2019-10-31 | 0.578 | 0.67 | 0.97 | 41.77 | 10 |
| 14 | 2019-08-01 .. 2020-01-31 | 1.409 | 1.32 | -2.97 | -99.00 | 11 |
| 15 | 2019-11-01 .. 2020-04-30 | 0.624 | 0.71 | -2.06 | -60.39 | 6 |
| 16 | 2020-02-03 .. 2020-07-31 | -0.009 | 0.06 | 0.63 | 17.21 | 7 |
| 17 | 2020-05-01 .. 2020-10-30 | -0.221 | -0.28 | 0.80 | 30.63 | 12 |
| 18 | 2020-08-03 .. 2021-02-02 | -0.809 | -1.36 | -0.21 | -7.78 | 7 |
| 19 | 2020-11-02 .. 2021-04-30 | -0.383 | -0.53 | -1.85 | -52.82 | 4 |
| 20 | 2021-02-02 .. 2021-07-30 | 0.115 | 0.16 | 0.93 | 44.03 | 6 |
| 21 | 2021-05-03 .. 2021-11-02 | -0.076 | -0.06 | 2.33 | 116.75 | 10 |
| 22 | 2021-08-03 .. 2022-02-02 | 0.300 | 0.48 | 2.87 | 104.50 | 10 |
| 23 | 2021-11-03 .. 2022-05-02 | -0.018 | 0.02 | -0.68 | -18.56 | 8 |
| 24 | 2022-02-02 .. 2022-08-01 | 1.129 | 1.42 | 0.53 | 16.39 | 11 |
| 25 | 2022-05-03 .. 2022-11-02 | 1.103 | 1.39 | 1.60 | 69.60 | 14 |
| 26 | 2022-08-03 .. 2023-02-02 | 1.247 | 1.69 | 2.93 | 173.33 | 15 |
| 27 | 2022-11-02 .. 2023-05-01 | 1.602 | 1.97 | 2.90 | 172.85 | 13 |
| 28 | 2023-02-01 .. 2023-07-31 | 1.697 | 2.08 | 0.43 | 17.00 | 9 |
| 29 | 2023-05-02 .. 2023-11-01 | 1.867 | 2.50 | -0.56 | -22.33 | 4 |
| 30 | 2023-08-02 .. 2024-02-01 | 1.503 | 1.58 | -0.05 | -2.34 | 4 |
| 31 | 2023-11-01 .. 2024-04-30 | 1.113 | 1.24 | 0.48 | 15.11 | 7 |
| 32 | 2024-02-01 .. 2024-07-31 | -0.007 | 0.04 | 1.48 | 63.50 | 5 |
| 33 | 2024-05-01 .. 2024-10-31 | -0.229 | -0.25 | 1.93 | 84.35 | 4 |
| 34 | 2024-08-01 .. 2025-01-31 | 0.607 | 0.74 | 4.26 | 220.06 | 7 |
| 35 | 2024-11-01 .. 2025-04-30 | 1.259 | 1.21 | 3.56 | 204.98 | 11 |
| 36 | 2025-02-04 .. 2025-08-01 | 2.059 | 2.95 | 2.24 | 101.21 | 11 |
| 37 | 2025-05-01 .. 2025-10-31 | 2.037 | 2.90 | 3.86 | 163.44 | 8 |
| 38 | 2025-08-01 .. 2026-01-30 | 2.212 | 3.34 | 1.54 | 57.91 | 7 |
| 39 | 2025-11-03 .. 2026-05-01 | 2.354 | 3.70 | -1.11 | -41.62 | 10 |
| 40 | 2026-02-03 .. 2026-07-31 | 1.585 | 1.77 | 0.03 | 0.91 | 14 |
| 41 | 2026-05-01 .. 2026-10-30 | 1.513 | 1.59 | 0.41 | 19.57 | 11 |
