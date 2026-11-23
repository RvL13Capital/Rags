# PIPELINE VERIFICATION - planted-edge synthetic data - tearsheet

> **PIPELINE VERIFICATION ON SYNTHETIC DATA.** Every number below comes from a simulated
> price process and is not empirical market performance.

**Verdict:** PIPELINE VERIFICATION ONLY - synthetic data. No empirical conclusion about the NASDAQ-100 can be drawn. NOT READY FOR LIVE DEPLOYMENT.

Data sources: SYNTHETIC-PIPELINE-VERIFICATION

| Run setting | Value |
|---|---|
| modules | multi_stage_exits, regime_gate |
| regime model | hmm2 |
| walk-forward | 36m train / 12m val / 6m OOS, step 3m, purge 5d + embargo 5d |
| folds | 42 |
| tuned parameters | none (fixed canonical priors) |
| Optuna trials per fold | 0 |
| backtests evaluated in search | 42 |
| configurations compared (K) | 15 |
| instrument | Capital.com-EU-US100-0.001 |
| entry order | stop_limit |
| initial equity | 5,000 EUR |
| ablation variants evaluated | 15 |
| largest search space in any variant | 1 |
| risk per trade | 0.25% (survival gates also at 0.75%) |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Holdout | Paper | Live |
|---|---:|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 0.83% | 1.00% | 0.86% | n/a | n/a | n/a |
| 2. Annualised volatility | 0.94% | 1.04% | 1.03% | n/a | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 0.89 | 0.96 | 0.83 | n/a | n/a | n/a |
| 4. Sortino ratio | 1.46 | 1.56 | 1.31 | n/a | n/a | n/a |
| 5. Calmar ratio | 0.39 | 0.47 | 0.40 | n/a | n/a | n/a |
| 6. Maximum drawdown | 2.15% | 2.15% | 2.14% | n/a | n/a | n/a |
| 6b. Max drawdown / annualised volatility (info) | 2.29 | 2.06 | 2.07 | n/a | n/a | n/a |
| 7. Profit factor (net) | 1.75 | 1.71 | 1.57 | n/a | n/a | n/a |
| 8. Win rate | 52.60% | 52.69% | 51.57% | n/a | n/a | n/a |
| 9. Avg win / avg loss | 1.58 | 1.53 | 1.48 | n/a | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 3.76 | 3.53 | 3.01 | n/a | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.348 | 0.326 | 0.284 | n/a | n/a | n/a |
| 11. Independent trades | 154 | 167 | 159 | n/a | n/a | n/a |
| 12. Annualised turnover (x equity) | 6.1 | 7.6 | 7.5 | n/a | n/a | n/a |
| 13a. Transaction costs (EUR) | 55.26 | 59.55 | 56.71 | n/a | n/a | n/a |
| 13b. Financing costs (EUR) | 65.84 | 70.26 | 74.86 | n/a | n/a | n/a |
| 14a. ES 97.5% daily | 0.20% | 0.22% | 0.23% | n/a | n/a | n/a |
| 14b. ES 99% daily | 0.27% | 0.30% | 0.30% | n/a | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.15% | 0.21% | 0.21% | n/a | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.02% | 0.03% | 0.03% | n/a | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.01% | 0.02% | 0.02% | n/a | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.01% | 0.01% | 0.01% | n/a | n/a | n/a |
| 15e. MC median terminal wealth (x) | 1.006 | 1.007 | 1.007 | n/a | n/a | n/a |
| 15f. MC worst 99% drawdown | 4.59% | 5.16% | 5.03% | n/a | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 122 | 120 | 120 | n/a | n/a | n/a |
| Net P&L (EUR) | 578.79 | 589.70 | 479.24 | n/a | n/a | n/a |
| Sample length (years) | 13.18 | 11.18 | 10.68 | n/a | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it.

Out-of-Sample is the stitched walk-forward series. Its hyper-parameters were chosen on validation data, but the module retention decisions of the ablation were made ON this series, so it is not selection-free; the DSR and P(OOS loss) gates exist to account for that. No holdout was reserved, so no column of this table is selection-free.

## Release gates (stitched walk-forward OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.8345 | > 0.80 | PASS |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.9618 | > 0.95 | PASS |
| CSCV P(OOS loss of IS-best) | 0.1260 | < 15% | PASS |
| CSCV rank PBO | n/a | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (nothing tuned) |
| OOS net profit factor | 1.5742 | > 1.20 | PASS |
| OOS net P&L under 1.5x costs | 399.9477 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0214 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0003 | < 2% | PASS |
| OOS maximum drawdown at the 0.75% risk ceiling | 0.0628 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling | 0.0000 | < 2% | PASS |
| Independent OOS trades | 159 | >= 100 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** DSR, zero-mean null: **0.9618** (per-period SR 0.0526 vs SR0 0.0189; K = 15 configurations compared, average pairwise correlation 0.62, N_eff = 6.3; cross-trial Sharpe std 0.0142)
- Informational. DSR with the cross-trial mean in the benchmark and all K trials: 0.2673 (SR0 0.0644, cross-trial mean 0.0392). Probabilistic Sharpe ratio against zero (N_eff = 1): 0.9971.
- CSCV over 12870 splits of the 15 configurations compared: P(OOS loss of the in-sample winner) **0.126**; IS->OOS Sharpe slope -0.647

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 399.95 EUR, Sharpe 0.71, profit factor 1.47, max drawdown 2.23%

## Stitched OOS replayed at the 0.75% risk ceiling

Same signals and fold parameters, nothing re-estimated; only the risk budget per trade changes.

- Max drawdown 6.28%, annualised volatility 2.83%, max drawdown / volatility 2.21, Sharpe 0.68, net P&L 1,117.02 EUR
- Monte Carlo P(DD >= 10%) 0.04%, P(DD >= 20%) 0.00%, worst 99% drawdown 6.37%; breaker pauses 0, hard deactivation none

## Stability over time (informational, not a gate)

Non-overlapping 6-month slices of the stitched OOS curve: 13 of 21 profitable (61.9%), 8 losing, 0 with no trade (active slices: 21).

| Slice | Window | Net P&L | Trades | Note |
|---:|---|---:|---:|---|
| 0 | 2016-02-03 .. 2016-08-02 | -19.84 | 6 |  |
| 1 | 2016-08-03 .. 2017-02-02 | 124.56 | 12 |  |
| 2 | 2017-02-03 .. 2017-08-02 | 31.13 | 14 |  |
| 3 | 2017-08-03 .. 2018-02-02 | 16.17 | 6 |  |
| 4 | 2018-02-05 .. 2018-08-02 | 9.16 | 5 |  |
| 5 | 2018-08-03 .. 2019-02-01 | -13.84 | 3 |  |
| 6 | 2019-02-04 .. 2019-08-02 | 35.94 | 4 |  |
| 7 | 2019-08-05 .. 2020-01-31 | -26.75 | 5 |  |
| 8 | 2020-02-03 .. 2020-07-31 | -16.12 | 3 |  |
| 9 | 2020-08-03 .. 2021-02-02 | -3.18 | 2 |  |
| 10 | 2021-02-03 .. 2021-08-02 | 35.14 | 8 |  |
| 11 | 2021-08-03 .. 2022-02-02 | -26.74 | 12 |  |
| 12 | 2022-02-03 .. 2022-08-02 | 8.11 | 11 |  |
| 13 | 2022-08-03 .. 2023-02-02 | 141.67 | 11 |  |
| 14 | 2023-02-03 .. 2023-08-02 | 72.63 | 11 |  |
| 15 | 2023-08-03 .. 2024-02-02 | 104.63 | 9 |  |
| 16 | 2024-02-05 .. 2024-08-02 | -9.82 | 3 |  |
| 17 | 2024-08-05 .. 2025-01-31 | 64.75 | 12 |  |
| 18 | 2025-02-03 .. 2025-08-01 | 36.77 | 3 |  |
| 19 | 2025-08-04 .. 2026-02-02 | -56.18 | 6 |  |
| 20 | 2026-02-03 .. 2026-07-31 | 4.50 | 7 |  |
| 21 | 2026-08-03 .. 2026-10-30 | -33.43 | 6 | incomplete, not counted |

## Walk-forward folds (42 overlapping OOS windows, each run on its own; informational)

66.7% of the windows were profitable. Adjacent windows share half their span, so trades are counted twice across the table.

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2016-02-03 .. 2016-08-02 | 0.317 | 0.47 | -0.80 | -19.84 | 6 |
| 1 | 2016-05-02 .. 2016-11-01 | 0.440 | 0.62 | 1.35 | 43.57 | 8 |
| 2 | 2016-08-03 .. 2017-02-02 | 1.061 | 1.12 | 3.13 | 124.81 | 12 |
| 3 | 2016-11-03 .. 2017-05-02 | 1.614 | 1.86 | 1.62 | 66.82 | 13 |
| 4 | 2017-02-02 .. 2017-08-01 | 1.351 | 1.49 | 0.82 | 29.86 | 14 |
| 5 | 2017-05-02 .. 2017-11-01 | 0.596 | 0.74 | 0.92 | 26.05 | 9 |
| 6 | 2017-08-02 .. 2018-02-01 | 1.426 | 1.67 | 0.60 | 11.68 | 6 |
| 7 | 2017-11-01 .. 2018-04-30 | 0.913 | 1.13 | 1.58 | 24.69 | 6 |
| 8 | 2018-02-01 .. 2018-07-31 | 0.803 | 1.07 | 0.71 | 13.20 | 6 |
| 9 | 2018-05-01 .. 2018-10-31 | 1.408 | 1.32 | 1.37 | 21.73 | 3 |
| 10 | 2018-08-01 .. 2019-01-31 | 1.200 | 1.00 | -0.77 | -11.24 | 2 |
| 11 | 2018-11-01 .. 2019-04-30 | 1.137 | 1.06 | -0.91 | -18.65 | 5 |
| 12 | 2019-02-04 .. 2019-08-02 | -0.140 | -0.18 | 1.88 | 39.74 | 4 |
| 13 | 2019-05-01 .. 2019-10-31 | 0.044 | 0.08 | 1.80 | 37.25 | 4 |
| 14 | 2019-08-01 .. 2020-01-31 | 0.554 | 0.58 | -1.05 | -25.44 | 5 |
| 15 | 2019-11-01 .. 2020-04-30 | 0.452 | 0.44 | -1.32 | -38.06 | 6 |
| 16 | 2020-02-03 .. 2020-07-31 | 0.219 | 0.30 | -0.74 | -15.75 | 3 |
| 17 | 2020-05-01 .. 2020-10-30 | 0.110 | 0.18 | 0.84 | 7.92 | 1 |
| 18 | 2020-08-03 .. 2021-02-02 | -0.645 | -0.91 | -0.21 | -2.90 | 2 |
| 19 | 2020-11-02 .. 2021-04-30 | -0.536 | -0.70 | 0.61 | 13.42 | 4 |
| 20 | 2021-02-02 .. 2021-07-30 | -0.173 | -0.20 | 1.27 | 34.97 | 8 |
| 21 | 2021-05-03 .. 2021-11-02 | 0.331 | 0.31 | 0.04 | 0.94 | 10 |
| 22 | 2021-08-03 .. 2022-02-02 | 0.921 | 1.01 | -0.84 | -26.11 | 12 |
| 23 | 2021-11-03 .. 2022-05-02 | -0.196 | -0.27 | -2.55 | -67.10 | 12 |
| 24 | 2022-02-02 .. 2022-08-01 | 0.065 | 0.16 | 0.07 | 1.65 | 12 |
| 25 | 2022-05-03 .. 2022-11-02 | -0.589 | -0.95 | 3.04 | 73.36 | 11 |
| 26 | 2022-08-03 .. 2023-02-02 | -0.355 | -0.52 | 4.68 | 136.81 | 11 |
| 27 | 2022-11-02 .. 2023-05-01 | 0.581 | 1.06 | 4.42 | 128.73 | 10 |
| 28 | 2023-02-01 .. 2023-07-31 | 1.592 | 2.44 | 2.21 | 65.82 | 10 |
| 29 | 2023-05-02 .. 2023-11-01 | 2.393 | 3.77 | 2.40 | 81.00 | 10 |
| 30 | 2023-08-02 .. 2024-02-01 | 2.280 | 3.48 | 3.21 | 98.51 | 10 |
| 31 | 2023-11-01 .. 2024-04-30 | 2.133 | 3.13 | 1.97 | 45.19 | 7 |
| 32 | 2024-02-01 .. 2024-07-31 | 1.936 | 2.64 | -0.42 | -8.92 | 3 |
| 33 | 2024-05-01 .. 2024-10-31 | 1.748 | 2.17 | -0.41 | -9.93 | 6 |
| 34 | 2024-08-01 .. 2025-01-31 | 1.334 | 1.68 | 1.84 | 59.39 | 12 |
| 35 | 2024-11-01 .. 2025-04-30 | 0.389 | 0.60 | 2.65 | 79.38 | 9 |
| 36 | 2025-02-04 .. 2025-08-01 | 0.751 | 0.92 | 2.47 | 32.84 | 3 |
| 37 | 2025-05-01 .. 2025-10-31 | 1.323 | 1.35 | 0.40 | 6.13 | 4 |
| 38 | 2025-08-01 .. 2026-01-30 | 1.624 | 1.86 | -2.64 | -50.59 | 6 |
| 39 | 2025-11-03 .. 2026-05-01 | 1.776 | 2.23 | -0.14 | -3.11 | 8 |
| 40 | 2026-02-03 .. 2026-07-31 | -0.148 | -0.20 | 0.18 | 3.94 | 7 |
| 41 | 2026-05-01 .. 2026-10-30 | -0.057 | -0.05 | -1.91 | -53.76 | 9 |
