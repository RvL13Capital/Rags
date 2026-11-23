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
| instrument | Capital.com-EU-US100-0.001 |
| initial equity | 5,000 EUR |
| ablation variants evaluated | 15 |
| largest search space in any variant | 1 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 1.97% | 1.55% | 1.63% | n/a | n/a |
| 2. Annualised volatility | 1.83% | 1.80% | 1.81% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 1.08 | 0.86 | 0.90 | n/a | n/a |
| 4. Sortino ratio | 1.83 | 1.41 | 1.48 | n/a | n/a |
| 5. Calmar ratio | 0.68 | 0.54 | 0.60 | n/a | n/a |
| 6. Maximum drawdown | 2.88% | 2.87% | 2.71% | n/a | n/a |
| 7. Profit factor (net) | 1.87 | 1.67 | 1.69 | n/a | n/a |
| 8. Win rate | 43.72% | 39.50% | 40.00% | n/a | n/a |
| 9. Avg win / avg loss | 2.41 | 2.55 | 2.54 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 6.33 | 4.70 | 4.96 | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.552 | 0.428 | 0.452 | n/a | n/a |
| 11. Independent trades | 231 | 200 | 190 | n/a | n/a |
| 12. Annualised turnover (x equity) | 9.8 | 9.7 | 9.6 | n/a | n/a |
| 13a. Transaction costs (EUR) | 121.24 | 94.86 | 91.71 | n/a | n/a |
| 13b. Financing costs (EUR) | 136.85 | 137.31 | 142.04 | n/a | n/a |
| 14a. ES 97.5% daily | 0.34% | 0.34% | 0.34% | n/a | n/a |
| 14b. ES 99% daily | 0.44% | 0.45% | 0.44% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.76% | 0.82% | 0.82% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.16% | 0.17% | 0.17% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.06% | 0.06% | 0.06% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.03% | 0.03% | 0.03% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 1.007 | 1.005 | 1.006 | n/a | n/a |
| 15f. MC worst 99% drawdown | 9.08% | 9.38% | 9.40% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 134 | 140 | 137 | n/a | n/a |
| Net P&L (EUR) | 1,462.42 | 940.44 | 942.92 | n/a | n/a |
| Sample length (years) | 13.18 | 11.18 | 10.68 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.9028 | > 0.80 | PASS |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.9586 | > 0.95 | PASS |
| CSCV P(OOS loss of IS-best) | 0.0706 | < 15% | PASS |
| CSCV rank PBO | n/a | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (nothing tuned) |
| OOS net profit factor | 1.6906 | > 1.20 | PASS |
| OOS net P&L under 1.5x costs | 765.8598 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0271 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0017 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.6905 | >= 65% | PASS |
| Independent OOS trades | 190 | >= 150 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** DSR, zero-mean null: **0.9586** (per-period SR 0.0569 vs SR0 0.0245; K = 15 configurations compared, average pairwise correlation 0.64, N_eff = 6.0; cross-trial Sharpe std 0.0188)
- Informational. DSR with the cross-trial mean in the benchmark and all K trials: 0.1110 (SR0 0.0797, cross-trial mean 0.0465). Probabilistic Sharpe ratio against zero (N_eff = 1): 0.9988.
- CSCV over 12870 splits of the 15 configurations compared: P(OOS loss of the in-sample winner) **0.071**; IS->OOS Sharpe slope -0.740

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 765.86 EUR, Sharpe 0.77, profit factor 1.56, max drawdown 2.81%

## Walk-forward folds (42 folds, 69.0% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2016-02-03 .. 2016-08-02 | -0.043 | 0.01 | -1.66 | -84.69 | 8 |
| 1 | 2016-05-02 .. 2016-11-01 | 0.244 | 0.33 | -1.72 | -89.09 | 10 |
| 2 | 2016-08-03 .. 2017-02-02 | -0.491 | -0.68 | 0.72 | 43.93 | 15 |
| 3 | 2016-11-03 .. 2017-05-02 | -0.270 | -0.32 | 1.82 | 121.25 | 13 |
| 4 | 2017-02-02 .. 2017-08-01 | -0.284 | -0.32 | 2.60 | 127.20 | 7 |
| 5 | 2017-05-02 .. 2017-11-01 | -0.313 | -0.36 | 2.40 | 72.53 | 7 |
| 6 | 2017-08-02 .. 2018-02-01 | 1.457 | 1.54 | 0.17 | 6.43 | 11 |
| 7 | 2017-11-01 .. 2018-04-30 | 1.606 | 1.86 | -1.67 | -49.95 | 10 |
| 8 | 2018-02-01 .. 2018-07-31 | 1.162 | 1.37 | -0.41 | -12.42 | 9 |
| 9 | 2018-05-01 .. 2018-10-31 | 0.960 | 1.16 | 1.00 | 47.79 | 9 |
| 10 | 2018-08-01 .. 2019-01-31 | 0.027 | 0.08 | 0.54 | 22.93 | 6 |
| 11 | 2018-11-01 .. 2019-04-30 | 0.115 | 0.17 | -0.28 | -6.62 | 3 |
| 12 | 2019-02-04 .. 2019-08-02 | 0.039 | 0.08 | 2.26 | 98.12 | 5 |
| 13 | 2019-05-01 .. 2019-10-31 | 0.463 | 0.57 | 0.89 | 41.60 | 10 |
| 14 | 2019-08-01 .. 2020-01-31 | 1.315 | 1.26 | -2.99 | -106.89 | 11 |
| 15 | 2019-11-01 .. 2020-04-30 | 0.540 | 0.64 | -2.08 | -65.53 | 6 |
| 16 | 2020-02-03 .. 2020-07-31 | -0.020 | 0.05 | 0.48 | 14.04 | 7 |
| 17 | 2020-05-01 .. 2020-10-30 | -0.233 | -0.29 | 0.60 | 24.25 | 12 |
| 18 | 2020-08-03 .. 2021-02-02 | -0.840 | -1.42 | 0.20 | 8.03 | 8 |
| 19 | 2020-11-02 .. 2021-04-30 | -0.415 | -0.57 | -1.05 | -35.49 | 5 |
| 20 | 2021-02-02 .. 2021-07-30 | 0.267 | 0.34 | 0.85 | 41.55 | 6 |
| 21 | 2021-05-03 .. 2021-11-02 | 0.093 | 0.14 | 2.22 | 113.79 | 10 |
| 22 | 2021-08-03 .. 2022-02-02 | 0.414 | 0.56 | 2.81 | 105.42 | 10 |
| 23 | 2021-11-03 .. 2022-05-02 | 0.123 | 0.21 | -0.64 | -19.16 | 8 |
| 24 | 2022-02-02 .. 2022-08-01 | 1.053 | 1.35 | 0.35 | 11.11 | 11 |
| 25 | 2022-05-03 .. 2022-11-02 | 1.023 | 1.32 | 1.53 | 71.04 | 14 |
| 26 | 2022-08-03 .. 2023-02-02 | 1.190 | 1.62 | 2.91 | 183.17 | 15 |
| 27 | 2022-11-02 .. 2023-05-01 | 1.564 | 1.92 | 2.89 | 184.97 | 13 |
| 28 | 2023-02-01 .. 2023-07-31 | 1.655 | 1.98 | 0.45 | 19.43 | 9 |
| 29 | 2023-05-02 .. 2023-11-01 | 1.820 | 2.38 | -0.58 | -24.42 | 4 |
| 30 | 2023-08-02 .. 2024-02-01 | 1.506 | 1.59 | -0.09 | -4.02 | 4 |
| 31 | 2023-11-01 .. 2024-04-30 | 1.084 | 1.23 | 0.36 | 11.54 | 7 |
| 32 | 2024-02-01 .. 2024-07-31 | -0.081 | -0.05 | 1.41 | 62.31 | 5 |
| 33 | 2024-05-01 .. 2024-10-31 | -0.276 | -0.31 | 1.90 | 86.23 | 4 |
| 34 | 2024-08-01 .. 2025-01-31 | 0.542 | 0.67 | 4.18 | 223.44 | 7 |
| 35 | 2024-11-01 .. 2025-04-30 | 1.176 | 1.16 | 3.44 | 204.05 | 11 |
| 36 | 2025-02-04 .. 2025-08-01 | 2.029 | 2.88 | 2.08 | 98.31 | 11 |
| 37 | 2025-05-01 .. 2025-10-31 | 2.005 | 2.82 | 3.79 | 162.40 | 8 |
| 38 | 2025-08-01 .. 2026-01-30 | 2.172 | 3.24 | 1.44 | 57.47 | 7 |
| 39 | 2025-11-03 .. 2026-05-01 | 2.309 | 3.59 | -1.11 | -43.58 | 10 |
| 40 | 2026-02-03 .. 2026-07-31 | 1.535 | 1.65 | -0.00 | -0.72 | 14 |
| 41 | 2026-05-01 .. 2026-10-30 | 1.338 | 1.41 | 0.35 | 17.16 | 11 |
