# Bot A TWAP - Track 1: Capital.com, 0.25% - schedule II - tearsheet

**Verdict:** NOT READY FOR LIVE DEPLOYMENT - gates not passed: Deflated Sharpe Ratio (zero-mean null, N_eff); OOS net profit factor; Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling

Data sources: NQ futures 1m less fitted basis (cash-equivalent NASDAQ-100)

| Run setting | Value |
|---|---|
| rule | Bot A noise-area momentum, long and short, flat at 15:55 ET |
| anchor | TWAP |
| account | Track 1: Capital.com, 0.25% |
| cost schedule | II (stress: III) |
| prior effective trials | 22.44 |
| in-sample column | 2015-2020, same account settings |
| out-of-sample | 2021-01-04 .. 2025-07-24 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Holdout | Paper | Live |
|---|---:|---:|---:|---:|---:|---:|
| 1. CAGR (net) | -0.86% | n/a | 2.46% | n/a | n/a | n/a |
| 2. Annualised volatility | 1.94% | n/a | 2.93% | n/a | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | -0.44 | n/a | 0.84 | n/a | n/a | n/a |
| 4. Sortino ratio | -0.71 | n/a | 1.49 | n/a | n/a | n/a |
| 5. Calmar ratio | -0.12 | n/a | 0.99 | n/a | n/a | n/a |
| 6. Maximum drawdown | 7.45% | n/a | 2.48% | n/a | n/a | n/a |
| 6b. Max drawdown / annualised volatility (info) | 3.83 | n/a | 0.85 | n/a | n/a | n/a |
| 7. Profit factor (net) | 0.90 | n/a | 1.17 | n/a | n/a | n/a |
| 8. Win rate | 32.93% | n/a | 40.00% | n/a | n/a | n/a |
| 9. Avg win / avg loss | 1.83 | n/a | 1.76 | n/a | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | -0.23 | n/a | 0.65 | n/a | n/a | n/a |
| 10b. Net expectancy / trade (R) | -0.013 | n/a | 0.051 | n/a | n/a | n/a |
| 11. Independent trades | 1081 | n/a | 880 | n/a | n/a | n/a |
| 12. Annualised turnover (x equity) | 110.5 | n/a | 180.4 | n/a | n/a | n/a |
| 13a. Transaction costs (EUR) | 854.56 | n/a | 439.43 | n/a | n/a | n/a |
| 13b. Financing costs (EUR) | 0.00 | n/a | 0.00 | n/a | n/a | n/a |
| 14a. ES 97.5% daily | 0.33% | n/a | 0.38% | n/a | n/a | n/a |
| 14b. ES 99% daily | 0.39% | n/a | 0.46% | n/a | n/a | n/a |
| 15a. MC P(DD >= 10%) | 1.05% | n/a | 19.17% | n/a | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.10% | n/a | 0.48% | n/a | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.02% | n/a | 0.08% | n/a | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.01% | n/a | 0.02% | n/a | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.967 | n/a | 0.947 | n/a | n/a | n/a |
| 15f. MC worst 99% drawdown | 10.14% | n/a | 17.28% | n/a | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 239 | n/a | 232 | n/a | n/a | n/a |
| Net P&L (EUR) | -250.58 | n/a | 576.15 | n/a | n/a | n/a |
| Sample length (years) | 5.92 | n/a | 4.50 | n/a | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it.

Out-of-Sample is the stitched walk-forward series; it was not used to choose anything in this run (earlier runs on the same data are a separate matter: see prior effective trials).

## Release gates (stitched walk-forward OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.8414 | > 0.80 | PASS |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.4242 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | n/a | < 15% | N/A (no selection) |
| CSCV rank PBO | n/a | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (no selection) |
| OOS net profit factor | 1.1720 | > 1.20 | FAIL |
| OOS net P&L under 1.5x costs | 380.0860 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0248 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0048 | < 2% | PASS |
| OOS maximum drawdown at the 0.75% risk ceiling | 0.0644 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling | 0.0207 | < 2% | FAIL |
| Independent OOS trades | 880 | >= 100 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** One fixed configuration in this run, 22.4 effective trials carried from earlier research: N_eff = 23.4, SR0 = 0.0585 per period (variance 1/T), DSR **0.4242** (per-period SR 0.0530; against zero: 0.9670). The CSCV statistics are not applicable.

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 380.09 EUR, Sharpe 0.57, profit factor 1.11, max drawdown 2.77%

## Stitched OOS replayed at the 0.75% risk ceiling

Same signals and fold parameters, nothing re-estimated; only the risk budget per trade changes.

- Max drawdown 6.44%, annualised volatility 7.39%, max drawdown / volatility 0.87, Sharpe 0.61, net P&L 1,056.09 EUR
- Monte Carlo P(DD >= 10%) 46.96%, P(DD >= 20%) 2.07%, worst 99% drawdown 21.66%; breaker pauses 0, hard deactivation none

## Stability over time (informational, not a gate)

Non-overlapping 6-month slices of the stitched OOS curve: 8 of 9 profitable (88.9%), 1 losing, 0 with no trade (active slices: 9).

| Slice | Window | Net P&L | Trades | Note |
|---:|---|---:|---:|---|
| 0 | 2021-01-04 .. 2021-07-02 | 116.11 | 97 |  |
| 1 | 2021-07-06 .. 2022-01-03 | 51.54 | 91 |  |
| 2 | 2022-01-04 .. 2022-07-01 | 142.89 | 111 |  |
| 3 | 2022-07-05 .. 2023-01-03 | -16.40 | 106 |  |
| 4 | 2023-01-04 .. 2023-06-30 | 89.34 | 93 |  |
| 5 | 2023-07-05 .. 2024-01-03 | 16.87 | 97 |  |
| 6 | 2024-01-04 .. 2024-07-02 | 60.19 | 90 |  |
| 7 | 2024-07-05 .. 2025-01-03 | 122.61 | 98 |  |
| 8 | 2025-01-06 .. 2025-07-02 | 23.55 | 85 |  |
| 9 | 2025-07-07 .. 2025-07-24 | -30.54 | 12 | incomplete, not counted |
