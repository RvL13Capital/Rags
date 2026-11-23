# NASDAQ-100 swing system - tearsheet

**Verdict:** NOT READY FOR LIVE DEPLOYMENT - gates not passed: OOS net Sharpe; Deflated Sharpe Ratio (zero-mean null, N_eff); CSCV P(OOS loss of IS-best); Profitable walk-forward OOS folds; Independent OOS trades

Data sources: NQ futures 1m (HF mdelcristo/NQ-F_1min_OHLCV_Parquet, MIT) less fitted fair-value basis vs FRED NASDAQ100 (resampled from 1-minute bars)

| Run setting | Value |
|---|---|
| modules | multi_stage_exits, regime_gate, vpin_filter |
| regime model | rv20 |
| walk-forward | 36m train / 12m val / 6m OOS, step 3m, purge 5d + embargo 5d |
| folds | 25 |
| tuned parameters | vpin_block_pct |
| Optuna trials per fold | 60 |
| backtests evaluated in search | 1900 |
| configurations compared (K) | 78 |
| instrument | Capital.com-EU-US100-1to1-0.001 |
| entry order | stop_limit |
| initial equity | 5,000 EUR |
| ablation variants evaluated | 14 |
| largest search space in any variant | 2 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 0.47% | 0.49% | 0.72% | n/a | n/a |
| 2. Annualised volatility | 1.15% | 1.20% | 1.21% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 0.41 | 0.41 | 0.60 | n/a | n/a |
| 4. Sortino ratio | 0.62 | 0.62 | 0.91 | n/a | n/a |
| 5. Calmar ratio | 0.24 | 0.25 | 0.35 | n/a | n/a |
| 6. Maximum drawdown | 1.94% | 1.96% | 2.05% | n/a | n/a |
| 7. Profit factor (net) | 1.24 | 1.22 | 1.33 | n/a | n/a |
| 8. Win rate | 46.88% | 45.71% | 47.33% | n/a | n/a |
| 9. Avg win / avg loss | 1.40 | 1.45 | 1.48 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 1.33 | 1.24 | 1.81 | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.130 | 0.126 | 0.173 | n/a | n/a |
| 11. Independent trades | 160 | 140 | 131 | n/a | n/a |
| 12. Annualised turnover (x equity) | 9.8 | 9.7 | 9.6 | n/a | n/a |
| 13a. Transaction costs (EUR) | 74.71 | 53.81 | 48.10 | n/a | n/a |
| 13b. Financing costs (EUR) | 0.00 | 0.00 | 0.00 | n/a | n/a |
| 14a. ES 97.5% daily | 0.25% | 0.25% | 0.26% | n/a | n/a |
| 14b. ES 99% daily | 0.31% | 0.31% | 0.31% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.33% | 0.41% | 0.00% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.04% | 0.06% | 0.00% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.03% | 0.03% | 0.00% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.02% | 0.02% | 0.00% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 1.001 | 1.000 | 1.007 | n/a | n/a |
| 15f. MC worst 99% drawdown | 6.27% | 6.69% | 2.70% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 149 | 155 | 120 | n/a | n/a |
| Net P&L (EUR) | 213.43 | 174.24 | 237.74 | n/a | n/a |
| Sample length (years) | 8.98 | 6.98 | 6.45 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.6003 | > 0.80 | FAIL |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.8454 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | 0.3482 | < 15% | FAIL |
| CSCV rank PBO | 0.8320 | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (IS Sharpe IQR 0.02: homogeneous variants) |
| OOS net profit factor | 1.3322 | > 1.20 | PASS |
| OOS net P&L under 1.5x costs | 195.6386 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0205 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0000 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.6000 | >= 65% | FAIL |
| Independent OOS trades | 131 | >= 150 | FAIL |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** DSR, zero-mean null: **0.8454** (per-period SR 0.0378 vs SR0 0.0127; K = 78 configurations compared, average pairwise correlation 0.92, N_eff = 7.4; cross-trial Sharpe std 0.0090)
- Informational. DSR with the cross-trial mean in the benchmark and all K trials: 0.1961 (SR0 0.0589, cross-trial mean 0.0370). Probabilistic Sharpe ratio against zero (N_eff = 1): 0.9373.
- CSCV over 12870 splits of the 78 configurations compared: P(OOS loss of the in-sample winner) **0.348**; IS->OOS Sharpe slope -0.718
- Rank PBO over 64 parameter draws of this configuration: 0.832; in-sample Sharpe IQR across the draws 0.02

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 195.64 EUR, Sharpe 0.50, profit factor 1.27, max drawdown 2.06%

## Walk-forward folds (25 folds, 60.0% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2019-02-01 .. 2019-07-31 | -0.636 | -0.91 | 1.95 | 53.55 | 9 |
| 1 | 2019-05-01 .. 2019-10-31 | 0.105 | 0.18 | -2.11 | -49.88 | 7 |
| 2 | 2019-07-31 .. 2020-01-30 | 0.704 | 0.97 | -0.18 | -5.87 | 12 |
| 3 | 2019-10-30 .. 2020-04-29 | 0.118 | 0.24 | 1.22 | 41.44 | 10 |
| 4 | 2020-01-31 .. 2020-07-30 | 0.113 | 0.29 | 1.02 | 33.18 | 8 |
| 5 | 2020-04-30 .. 2020-10-29 | 0.065 | 0.20 | 2.82 | 105.07 | 12 |
| 6 | 2020-08-03 .. 2021-02-02 | 0.126 | 0.27 | 2.46 | 79.76 | 12 |
| 7 | 2020-10-30 .. 2021-04-29 | 1.631 | 1.90 | -0.17 | -4.72 | 12 |
| 8 | 2021-02-02 .. 2021-07-30 | 1.640 | 1.90 | 1.14 | 37.90 | 15 |
| 9 | 2021-05-03 .. 2021-11-02 | 1.645 | 1.91 | 0.29 | 9.27 | 14 |
| 10 | 2021-08-02 .. 2022-02-01 | 1.660 | 1.97 | 0.88 | 26.63 | 11 |
| 11 | 2021-11-01 .. 2022-04-29 | 0.049 | 0.10 | 2.41 | 69.15 | 8 |
| 12 | 2022-02-02 .. 2022-08-01 | 0.486 | 0.71 | -1.06 | -23.86 | 4 |
| 13 | 2022-05-03 .. 2022-11-02 | 0.559 | 0.83 | -1.66 | -51.85 | 11 |
| 14 | 2022-08-02 .. 2023-02-01 | 0.021 | 0.08 | -1.44 | -49.85 | 14 |
| 15 | 2022-11-01 .. 2023-04-28 | 0.279 | 0.45 | -1.10 | -32.83 | 11 |
| 16 | 2023-02-01 .. 2023-07-31 | -0.577 | -0.92 | 1.26 | 40.46 | 13 |
| 17 | 2023-05-03 .. 2023-11-02 | -0.688 | -1.07 | 1.83 | 59.80 | 12 |
| 18 | 2023-08-02 .. 2024-02-01 | -0.079 | -0.05 | 0.06 | 1.50 | 11 |
| 19 | 2023-10-31 .. 2024-04-29 | -0.113 | -0.11 | -0.10 | -3.34 | 11 |
| 20 | 2024-02-01 .. 2024-07-31 | 0.943 | 1.02 | 1.13 | 33.71 | 11 |
| 21 | 2024-05-02 .. 2024-11-01 | 0.962 | 0.93 | 2.02 | 47.79 | 8 |
| 22 | 2024-08-01 .. 2025-01-31 | 0.768 | 0.79 | 0.34 | 9.91 | 7 |
| 23 | 2024-10-30 .. 2025-04-29 | 1.369 | 1.43 | -0.51 | -16.44 | 7 |
| 24 | 2025-02-03 .. 2025-07-24 | 0.297 | 0.41 | -0.58 | -12.48 | 5 |
