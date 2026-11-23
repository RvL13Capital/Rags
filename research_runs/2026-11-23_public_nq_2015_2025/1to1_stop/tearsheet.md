# NASDAQ-100 swing system - tearsheet

**Verdict:** NOT READY FOR LIVE DEPLOYMENT - gates not passed: OOS net Sharpe; Deflated Sharpe Ratio (zero-mean null, N_eff); CSCV P(OOS loss of IS-best); OOS net profit factor; Profitable walk-forward OOS folds; Independent OOS trades

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
| entry order | stop |
| initial equity | 5,000 EUR |
| ablation variants evaluated | 14 |
| largest search space in any variant | 2 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 0.40% | 0.37% | 0.36% | n/a | n/a |
| 2. Annualised volatility | 1.26% | 1.32% | 1.32% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 0.32 | 0.28 | 0.28 | n/a | n/a |
| 4. Sortino ratio | 0.46 | 0.41 | 0.40 | n/a | n/a |
| 5. Calmar ratio | 0.17 | 0.16 | 0.15 | n/a | n/a |
| 6. Maximum drawdown | 2.34% | 2.34% | 2.46% | n/a | n/a |
| 7. Profit factor (net) | 1.17 | 1.14 | 1.14 | n/a | n/a |
| 8. Win rate | 45.29% | 44.52% | 43.80% | n/a | n/a |
| 9. Avg win / avg loss | 1.42 | 1.43 | 1.46 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 1.06 | 0.89 | 0.86 | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.095 | 0.089 | 0.080 | n/a | n/a |
| 11. Independent trades | 170 | 146 | 137 | n/a | n/a |
| 12. Annualised turnover (x equity) | 10.3 | 10.1 | 9.9 | n/a | n/a |
| 13a. Transaction costs (EUR) | 106.77 | 71.52 | 63.74 | n/a | n/a |
| 13b. Financing costs (EUR) | 0.00 | 0.00 | 0.00 | n/a | n/a |
| 14a. ES 97.5% daily | 0.28% | 0.29% | 0.29% | n/a | n/a |
| 14b. ES 99% daily | 0.35% | 0.35% | 0.35% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.41% | 0.48% | 0.46% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.08% | 0.08% | 0.07% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.03% | 0.03% | 0.03% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.02% | 0.02% | 0.02% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 1.003 | 1.001 | 1.003 | n/a | n/a |
| 15f. MC worst 99% drawdown | 6.79% | 7.10% | 7.00% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 144 | 152 | 144 | n/a | n/a |
| Net P&L (EUR) | 180.52 | 130.00 | 118.18 | n/a | n/a |
| Sample length (years) | 8.98 | 6.98 | 6.45 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.2807 | > 0.80 | FAIL |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.6418 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | 0.6792 | < 15% | FAIL |
| CSCV rank PBO | 0.8995 | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (IS Sharpe IQR 0.02: homogeneous variants) |
| OOS net profit factor | 1.1406 | > 1.20 | FAIL |
| OOS net P&L under 1.5x costs | 75.3552 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0246 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0007 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.5200 | >= 65% | FAIL |
| Independent OOS trades | 137 | >= 150 | FAIL |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** DSR, zero-mean null: **0.6418** (per-period SR 0.0177 vs SR0 0.0087; K = 78 configurations compared, average pairwise correlation 0.92, N_eff = 7.5; cross-trial Sharpe std 0.0061)
- Informational. DSR with the cross-trial mean in the benchmark and all K trials: 0.2508 (SR0 0.0344, cross-trial mean 0.0195). Probabilistic Sharpe ratio against zero (N_eff = 1): 0.7619.
- CSCV over 12870 splits of the 78 configurations compared: P(OOS loss of the in-sample winner) **0.679**; IS->OOS Sharpe slope -0.722
- Rank PBO over 64 parameter draws of this configuration: 0.899; in-sample Sharpe IQR across the draws 0.02

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 75.36 EUR, Sharpe 0.18, profit factor 1.09, max drawdown 2.55%

## Walk-forward folds (25 folds, 52.0% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2019-02-01 .. 2019-07-31 | 0.263 | 0.42 | 1.78 | 60.21 | 10 |
| 1 | 2019-05-01 .. 2019-10-31 | 1.399 | 1.38 | -2.12 | -54.93 | 7 |
| 2 | 2019-07-31 .. 2020-01-30 | 1.506 | 1.55 | -0.78 | -27.57 | 13 |
| 3 | 2019-10-30 .. 2020-04-29 | 0.145 | 0.27 | 0.59 | 22.01 | 11 |
| 4 | 2020-01-31 .. 2020-07-30 | -0.052 | 0.01 | 0.49 | 17.96 | 9 |
| 5 | 2020-04-30 .. 2020-10-29 | -0.163 | -0.18 | 1.70 | 73.63 | 13 |
| 6 | 2020-08-03 .. 2021-02-02 | -0.131 | -0.13 | 2.30 | 81.34 | 12 |
| 7 | 2020-10-30 .. 2021-04-29 | 0.868 | 1.04 | 0.35 | 9.85 | 12 |
| 8 | 2021-02-02 .. 2021-07-30 | 1.213 | 1.52 | 0.71 | 26.68 | 15 |
| 9 | 2021-05-03 .. 2021-11-02 | 1.152 | 1.46 | -0.30 | -11.99 | 15 |
| 10 | 2021-08-02 .. 2022-02-01 | 1.301 | 1.44 | 0.31 | 9.46 | 12 |
| 11 | 2021-11-01 .. 2022-04-29 | -0.030 | 0.02 | 2.41 | 64.26 | 8 |
| 12 | 2022-02-02 .. 2022-08-01 | 0.095 | 0.22 | -1.12 | -24.95 | 4 |
| 13 | 2022-05-03 .. 2022-11-02 | 0.187 | 0.36 | -1.64 | -53.02 | 11 |
| 14 | 2022-08-02 .. 2023-02-01 | -0.207 | -0.28 | -1.67 | -62.37 | 15 |
| 15 | 2022-11-01 .. 2023-04-28 | 0.219 | 0.36 | -1.38 | -47.35 | 12 |
| 16 | 2023-02-01 .. 2023-07-31 | -0.659 | -1.07 | 1.07 | 37.09 | 13 |
| 17 | 2023-05-03 .. 2023-11-02 | -0.756 | -1.19 | 1.78 | 58.71 | 12 |
| 18 | 2023-08-02 .. 2024-02-01 | -0.214 | -0.26 | -0.17 | -5.29 | 11 |
| 19 | 2023-10-31 .. 2024-04-29 | -0.260 | -0.32 | -0.56 | -19.71 | 12 |
| 20 | 2024-02-01 .. 2024-07-31 | 0.714 | 0.84 | -0.02 | -0.98 | 12 |
| 21 | 2024-05-02 .. 2024-11-01 | 0.570 | 0.62 | 0.88 | 22.45 | 8 |
| 22 | 2024-08-01 .. 2025-01-31 | 0.107 | 0.15 | 0.28 | 8.27 | 7 |
| 23 | 2024-10-30 .. 2025-04-29 | 0.599 | 0.72 | -0.56 | -18.50 | 7 |
| 24 | 2025-02-03 .. 2025-07-24 | -0.143 | -0.14 | -0.56 | -12.03 | 5 |
