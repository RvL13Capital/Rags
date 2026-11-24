# Bot A VWAP - Track 2: Pepperstone, 0.50% - schedule I - tearsheet

**Verdict:** NOT READY FOR LIVE DEPLOYMENT - gates not passed: Deflated Sharpe Ratio (zero-mean null, N_eff); Monte Carlo P(DD >= 20%)

Data sources: NQ futures 1m less fitted basis (cash-equivalent NASDAQ-100)

| Run setting | Value |
|---|---|
| rule | Bot A noise-area momentum, long and short, flat at 15:55 ET |
| anchor | VWAP |
| account | Track 2: Pepperstone, 0.50% |
| cost schedule | I (stress: I x1.5) |
| prior effective trials | 22.44 |
| in-sample column | 2015-2020, same account settings |
| out-of-sample | 2021-01-04 .. 2025-07-24 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Holdout | Paper | Live |
|---|---:|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 4.73% | n/a | 5.30% | n/a | n/a | n/a |
| 2. Annualised volatility | 4.64% | n/a | 4.59% | n/a | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 1.02 | n/a | 1.15 | n/a | n/a | n/a |
| 4. Sortino ratio | 2.10 | n/a | 2.12 | n/a | n/a | n/a |
| 5. Calmar ratio | 0.73 | n/a | 1.47 | n/a | n/a | n/a |
| 6. Maximum drawdown | 6.45% | n/a | 3.61% | n/a | n/a | n/a |
| 6b. Max drawdown / annualised volatility (info) | 1.39 | n/a | 0.79 | n/a | n/a | n/a |
| 7. Profit factor (net) | 1.27 | n/a | 1.26 | n/a | n/a | n/a |
| 8. Win rate | 35.29% | n/a | 39.78% | n/a | n/a | n/a |
| 9. Avg win / avg loss | 2.33 | n/a | 1.90 | n/a | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 1.75 | n/a | 1.59 | n/a | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.084 | n/a | 0.073 | n/a | n/a | n/a |
| 11. Independent trades | 901 | n/a | 822 | n/a | n/a | n/a |
| 12. Annualised turnover (x equity) | 273.5 | n/a | 267.5 | n/a | n/a | n/a |
| 13a. Transaction costs (EUR) | 385.36 | n/a | 274.81 | n/a | n/a | n/a |
| 13b. Financing costs (EUR) | 0.00 | n/a | 0.00 | n/a | n/a | n/a |
| 14a. ES 97.5% daily | 0.57% | n/a | 0.61% | n/a | n/a | n/a |
| 14b. ES 99% daily | 0.66% | n/a | 0.71% | n/a | n/a | n/a |
| 15a. MC P(DD >= 10%) | 39.46% | n/a | 45.07% | n/a | n/a | n/a |
| 15b. MC P(DD >= 20%) | 1.50% | n/a | 2.55% | n/a | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.24% | n/a | 0.38% | n/a | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.04% | n/a | 0.07% | n/a | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.932 | n/a | 0.933 | n/a | n/a | n/a |
| 15f. MC worst 99% drawdown | 21.43% | n/a | 23.90% | n/a | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 232 | n/a | 225 | n/a | n/a | n/a |
| Net P&L (EUR) | 1,575.65 | n/a | 1,306.76 | n/a | n/a | n/a |
| Sample length (years) | 5.92 | n/a | 4.50 | n/a | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it.

Out-of-Sample is the stitched walk-forward series; it was not used to choose anything in this run (earlier runs on the same data are a separate matter: see prior effective trials).

## Release gates (stitched walk-forward OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 1.1470 | > 0.80 | PASS |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.6862 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | n/a | < 15% | N/A (no selection) |
| CSCV rank PBO | n/a | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (no selection) |
| OOS net profit factor | 1.2560 | > 1.20 | PASS |
| OOS net P&L under 1.5x costs | 1166.9886 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0361 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0255 | < 2% | FAIL |
| OOS maximum drawdown at the 0.75% risk ceiling | 0.0515 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling | 0.0045 | < 2% | PASS |
| Independent OOS trades | 822 | >= 100 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** One fixed configuration in this run, 22.4 effective trials carried from earlier research: N_eff = 23.4, SR0 = 0.0585 per period (variance 1/T), DSR **0.6862** (per-period SR 0.0723; against zero: 0.9946). The CSCV statistics are not applicable.

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 1,166.99 EUR, Sharpe 1.04, profit factor 1.23, max drawdown 3.93%

## Stitched OOS replayed at the 0.75% risk ceiling

Same signals and fold parameters, nothing re-estimated; only the risk budget per trade changes.

- Max drawdown 5.15%, annualised volatility 6.68%, max drawdown / volatility 0.77, Sharpe 1.05, net P&L 1,793.21 EUR
- Monte Carlo P(DD >= 10%) 29.98%, P(DD >= 20%) 0.45%, worst 99% drawdown 18.73%; breaker pauses 0, hard deactivation none

## Stability over time (informational, not a gate)

Non-overlapping 6-month slices of the stitched OOS curve: 8 of 9 profitable (88.9%), 1 losing, 0 with no trade (active slices: 9).

| Slice | Window | Net P&L | Trades | Note |
|---:|---|---:|---:|---|
| 0 | 2021-01-04 .. 2021-07-02 | 212.99 | 95 |  |
| 1 | 2021-07-06 .. 2022-01-03 | 89.03 | 89 |  |
| 2 | 2022-01-04 .. 2022-07-01 | 294.73 | 86 |  |
| 3 | 2022-07-05 .. 2023-01-03 | -35.06 | 98 |  |
| 4 | 2023-01-04 .. 2023-06-30 | 224.24 | 90 |  |
| 5 | 2023-07-05 .. 2024-01-03 | 63.34 | 96 |  |
| 6 | 2024-01-04 .. 2024-07-02 | 97.31 | 86 |  |
| 7 | 2024-07-05 .. 2025-01-03 | 308.67 | 93 |  |
| 8 | 2025-01-06 .. 2025-07-02 | 101.54 | 78 |  |
| 9 | 2025-07-07 .. 2025-07-24 | -50.02 | 11 | incomplete, not counted |
