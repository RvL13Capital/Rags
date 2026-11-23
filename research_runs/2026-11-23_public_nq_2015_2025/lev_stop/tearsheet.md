# NASDAQ-100 swing system - tearsheet

**Verdict:** NOT READY FOR LIVE DEPLOYMENT - gates not passed: OOS net Sharpe; Deflated Sharpe Ratio (zero-mean null, N_eff); CSCV P(OOS loss of IS-best); OOS net profit factor; OOS net P&L under 1.5x costs; Profitable walk-forward OOS folds; Independent OOS trades

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
| entry order | stop |
| initial equity | 5,000 EUR |
| ablation variants evaluated | 14 |
| largest search space in any variant | 2 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |
|---|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 0.18% | 0.13% | 0.11% | n/a | n/a |
| 2. Annualised volatility | 1.25% | 1.32% | 1.32% | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 0.15 | 0.11 | 0.09 | n/a | n/a |
| 4. Sortino ratio | 0.21 | 0.15 | 0.13 | n/a | n/a |
| 5. Calmar ratio | 0.07 | 0.05 | 0.04 | n/a | n/a |
| 6. Maximum drawdown | 2.58% | 2.56% | 2.68% | n/a | n/a |
| 7. Profit factor (net) | 1.07 | 1.05 | 1.04 | n/a | n/a |
| 8. Win rate | 44.71% | 43.84% | 43.07% | n/a | n/a |
| 9. Avg win / avg loss | 1.33 | 1.34 | 1.38 | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 0.47 | 0.32 | 0.26 | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.041 | 0.036 | 0.025 | n/a | n/a |
| 11. Independent trades | 170 | 146 | 137 | n/a | n/a |
| 12. Annualised turnover (x equity) | 10.3 | 10.1 | 10.0 | n/a | n/a |
| 13a. Transaction costs (EUR) | 106.48 | 71.59 | 64.01 | n/a | n/a |
| 13b. Financing costs (EUR) | 98.58 | 84.03 | 80.61 | n/a | n/a |
| 14a. ES 97.5% daily | 0.29% | 0.29% | 0.30% | n/a | n/a |
| 14b. ES 99% daily | 0.35% | 0.35% | 0.35% | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.41% | 0.49% | 0.47% | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.08% | 0.08% | 0.07% | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.03% | 0.03% | 0.03% | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.02% | 0.02% | 0.02% | n/a | n/a |
| 15e. MC median terminal wealth (x) | 1.001 | 0.999 | 1.001 | n/a | n/a |
| 15f. MC worst 99% drawdown | 6.87% | 7.27% | 7.11% | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 151 | 160 | 152 | n/a | n/a |
| Net P&L (EUR) | 79.53 | 46.00 | 36.29 | n/a | n/a |
| Sample length (years) | 8.98 | 6.98 | 6.45 | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it; only Out-of-Sample is untouched by selection.

## Release gates (untouched OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.0914 | > 0.80 | FAIL |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.4687 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | 0.8552 | < 15% | FAIL |
| CSCV rank PBO | 0.7411 | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (IS Sharpe IQR 0.03: homogeneous variants) |
| OOS net profit factor | 1.0417 | > 1.20 | FAIL |
| OOS net P&L under 1.5x costs | -31.8717 | > 0 | FAIL |
| OOS maximum drawdown (true high-water mark) | 0.0268 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0007 | < 2% | PASS |
| Profitable walk-forward OOS folds | 0.5200 | >= 65% | FAIL |
| Independent OOS trades | 137 | >= 150 | FAIL |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** DSR, zero-mean null: **0.4687** (per-period SR 0.0058 vs SR0 0.0077; K = 78 configurations compared, average pairwise correlation 0.91, N_eff = 7.6; cross-trial Sharpe std 0.0054)
- Informational. DSR with the cross-trial mean in the benchmark and all K trials: 0.2492 (SR0 0.0226, cross-trial mean 0.0094). Probabilistic Sharpe ratio against zero (N_eff = 1): 0.5918.
- CSCV over 12870 splits of the 78 configurations compared: P(OOS loss of the in-sample winner) **0.855**; IS->OOS Sharpe slope -0.795
- Rank PBO over 64 parameter draws of this configuration: 0.741; in-sample Sharpe IQR across the draws 0.03

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L -31.87 EUR, Sharpe -0.07, profit factor 0.96, max drawdown 3.38%

## Walk-forward folds (25 folds, 52.0% profitable OOS)

| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2019-02-01 .. 2019-07-31 | 0.114 | 0.21 | 1.60 | 53.74 | 10 |
| 1 | 2019-05-01 .. 2019-10-31 | 1.124 | 1.14 | -2.16 | -56.61 | 7 |
| 2 | 2019-07-31 .. 2020-01-30 | 1.425 | 1.35 | -0.97 | -34.36 | 13 |
| 3 | 2019-10-30 .. 2020-04-29 | 0.048 | 0.13 | 0.40 | 14.94 | 11 |
| 4 | 2020-01-31 .. 2020-07-30 | -0.162 | -0.17 | 0.40 | 14.70 | 9 |
| 5 | 2020-04-30 .. 2020-10-29 | -0.259 | -0.35 | 1.60 | 69.17 | 13 |
| 6 | 2020-08-03 .. 2021-02-02 | -0.230 | -0.29 | 2.13 | 75.15 | 12 |
| 7 | 2020-10-30 .. 2021-04-29 | 0.721 | 0.89 | 0.14 | 3.70 | 12 |
| 8 | 2021-02-02 .. 2021-07-30 | 1.096 | 1.39 | 0.49 | 18.49 | 15 |
| 9 | 2021-05-03 .. 2021-11-02 | 1.041 | 1.33 | -0.49 | -19.55 | 15 |
| 10 | 2021-08-02 .. 2022-02-01 | 1.114 | 1.26 | 0.19 | 5.80 | 12 |
| 11 | 2021-11-01 .. 2022-04-29 | -0.175 | -0.19 | 2.36 | 62.59 | 8 |
| 12 | 2022-02-02 .. 2022-08-01 | -0.018 | 0.05 | -1.14 | -25.39 | 4 |
| 13 | 2022-05-03 .. 2022-11-02 | 0.082 | 0.21 | -1.68 | -54.52 | 11 |
| 14 | 2022-08-02 .. 2023-02-01 | -0.261 | -0.37 | -1.74 | -65.00 | 15 |
| 15 | 2022-11-01 .. 2023-04-28 | 0.186 | 0.32 | -1.52 | -52.32 | 12 |
| 16 | 2023-02-01 .. 2023-07-31 | -0.681 | -1.11 | 0.77 | 26.72 | 13 |
| 17 | 2023-05-03 .. 2023-11-02 | -0.812 | -1.29 | 1.54 | 50.35 | 12 |
| 18 | 2023-08-02 .. 2024-02-01 | -0.334 | -0.46 | -0.48 | -14.87 | 11 |
| 19 | 2023-10-31 .. 2024-04-29 | -0.403 | -0.54 | -0.93 | -32.27 | 12 |
| 20 | 2024-02-01 .. 2024-07-31 | 0.428 | 0.55 | -0.31 | -10.93 | 12 |
| 21 | 2024-05-02 .. 2024-11-01 | 0.235 | 0.30 | 0.63 | 15.76 | 8 |
| 22 | 2024-08-01 .. 2025-01-31 | -0.130 | -0.12 | 0.02 | 0.49 | 7 |
| 23 | 2024-10-30 .. 2025-04-29 | 0.312 | 0.44 | -0.81 | -26.86 | 7 |
| 24 | 2025-02-03 .. 2025-07-24 | -0.332 | -0.38 | -0.80 | -16.99 | 5 |
