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
| instrument | Capital.com-EU-US100-0.001 |
| entry order | stop_limit |
| initial equity | 5,000 EUR |
| ablation variants evaluated | 14 |
| largest search space in any variant | 2 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 0.26% | 0.29% | 0.53% | n/a | n/a |
| 2. Annualised volatility | 1.14% | 1.20% | 1.21% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 0.24 | 0.24 | 0.44 | n/a | n/a |
| 4. Sortino ratio | 0.35 | 0.36 | 0.66 | n/a | n/a |
| 5. Calmar ratio | 0.13 | 0.14 | 0.25 | n/a | n/a |
| 6. Maximum drawdown | 2.08% | 2.02% | 2.08% | n/a | n/a |
| 7. Profit factor (net) | 1.13 | 1.12 | 1.23 | n/a | n/a |
| 8. Win rate | 46.58% | 45.71% | 47.73% | n/a | n/a |
| 9. Avg win / avg loss | 1.29 | 1.33 | 1.35 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 0.74 | 0.72 | 1.31 | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.069 | 0.075 | 0.131 | n/a | n/a |
| 11. Independent trades | 161 | 140 | 132 | n/a | n/a |
| 12. Annualised turnover (x equity) | 9.8 | 9.7 | 9.6 | n/a | n/a |
| 13a. Transaction costs (EUR) | 74.75 | 54.02 | 48.45 | n/a | n/a |
| 13b. Financing costs (EUR) | 85.65 | 73.31 | 73.64 | n/a | n/a |
| 14a. ES 97.5% daily | 0.25% | 0.26% | 0.26% | n/a | n/a |
| 14b. ES 99% daily | 0.32% | 0.31% | 0.32% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.33% | 0.41% | 0.40% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.04% | 0.06% | 0.06% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.03% | 0.03% | 0.03% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.02% | 0.02% | 0.02% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.999 | 0.998 | 1.002 | n/a | n/a |
| 15f. MC worst 99% drawdown | 6.35% | 6.84% | 6.62% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 156 | 163 | 148 | n/a | n/a |
| Net P&L (EUR) | 119.73 | 100.47 | 172.90 | n/a | n/a |
| Sample length (years) | 8.98 | 6.98 | 6.45 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.4410 | > 0.80 | FAIL |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.7485 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | 0.4956 | < 15% | FAIL |
| CSCV rank PBO | 0.5165 | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (IS Sharpe IQR 0.02: homogeneous variants) |
| OOS net profit factor | 1.2343 | > 1.20 | PASS |
| OOS net P&L under 1.5x costs | 106.3581 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0208 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0006 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.5600 | >= 65% | FAIL |
| Independent OOS trades | 132 | >= 150 | FAIL |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** DSR, zero-mean null: **0.7485** (per-period SR 0.0278 vs SR0 0.0112; K = 78 configurations compared, average pairwise correlation 0.92, N_eff = 7.5; cross-trial Sharpe std 0.0079)
- Informational. DSR with the cross-trial mean in the benchmark and all K trials: 0.2262 (SR0 0.0464, cross-trial mean 0.0271). Probabilistic Sharpe ratio against zero (N_eff = 1): 0.8693.
- CSCV over 12870 splits of the 78 configurations compared: P(OOS loss of the in-sample winner) **0.496**; IS->OOS Sharpe slope -0.704
- Rank PBO over 64 parameter draws of this configuration: 0.516; in-sample Sharpe IQR across the draws 0.02

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 106.36 EUR, Sharpe 0.28, profit factor 1.14, max drawdown 2.15%

## Walk-forward folds (25 folds, 56.0% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2019-02-01 .. 2019-07-31 | -0.706 | -1.05 | 1.77 | 48.08 | 9 |
| 1 | 2019-05-01 .. 2019-10-31 | -0.026 | 0.01 | -2.15 | -51.22 | 7 |
| 2 | 2019-07-31 .. 2020-01-30 | 0.595 | 0.83 | -0.40 | -12.42 | 12 |
| 3 | 2019-10-30 .. 2020-04-29 | 0.032 | 0.10 | 1.03 | 34.61 | 10 |
| 4 | 2020-01-31 .. 2020-07-30 | 0.008 | 0.11 | 0.94 | 30.19 | 8 |
| 5 | 2020-04-30 .. 2020-10-29 | -0.035 | 0.03 | 2.75 | 101.42 | 12 |
| 6 | 2020-08-03 .. 2021-02-02 | 0.032 | 0.12 | 2.31 | 74.59 | 12 |
| 7 | 2020-10-30 .. 2021-04-29 | 1.576 | 1.76 | -0.37 | -10.24 | 12 |
| 8 | 2021-02-02 .. 2021-07-30 | 1.593 | 1.78 | 0.93 | 30.65 | 15 |
| 9 | 2021-05-03 .. 2021-11-02 | 1.594 | 1.78 | 0.10 | 2.97 | 14 |
| 10 | 2021-08-02 .. 2022-02-01 | 1.509 | 1.79 | 0.77 | 23.38 | 11 |
| 11 | 2021-11-01 .. 2022-04-29 | -0.097 | -0.08 | 2.37 | 67.42 | 8 |
| 12 | 2022-02-02 .. 2022-08-01 | 0.343 | 0.52 | -1.08 | -24.30 | 4 |
| 13 | 2022-05-03 .. 2022-11-02 | 0.443 | 0.68 | -1.70 | -53.41 | 11 |
| 14 | 2022-08-02 .. 2023-02-01 | -0.035 | -0.01 | -1.51 | -52.55 | 14 |
| 15 | 2022-11-01 .. 2023-04-28 | 0.251 | 0.41 | -1.25 | -37.44 | 11 |
| 16 | 2023-02-01 .. 2023-07-31 | -0.592 | -0.94 | 0.97 | 30.68 | 13 |
| 17 | 2023-05-03 .. 2023-11-02 | -0.733 | -1.15 | 1.60 | 51.66 | 12 |
| 18 | 2023-08-02 .. 2024-02-01 | -0.189 | -0.22 | -0.25 | -7.90 | 11 |
| 19 | 2023-10-31 .. 2024-04-29 | -0.263 | -0.32 | -0.46 | -14.61 | 11 |
| 20 | 2024-02-01 .. 2024-07-31 | 0.615 | 0.72 | 0.86 | 25.35 | 11 |
| 21 | 2024-05-02 .. 2024-11-01 | 0.613 | 0.63 | 2.05 | 51.23 | 9 |
| 22 | 2024-08-01 .. 2025-01-31 | 0.434 | 0.51 | 0.09 | 2.55 | 7 |
| 23 | 2024-10-30 .. 2025-04-29 | 0.990 | 1.16 | -0.76 | -24.37 | 7 |
| 24 | 2025-02-03 .. 2025-07-24 | 0.104 | 0.18 | -0.80 | -17.44 | 5 |
