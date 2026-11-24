# Bot A VWAP - Track 2: Capital.com, 0.50% - schedule II - tearsheet

**Verdict:** NOT READY FOR LIVE DEPLOYMENT - gates not passed: OOS net Sharpe; Deflated Sharpe Ratio (zero-mean null, N_eff); OOS net profit factor; Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling

Data sources: NQ futures 1m less fitted basis (cash-equivalent NASDAQ-100)

| Run setting | Value |
|---|---|
| rule | Bot A noise-area momentum, long and short, flat at 15:55 ET |
| anchor | VWAP |
| account | Track 2: Capital.com, 0.50% |
| cost schedule | II (stress: III) |
| prior effective trials | 22.44 |
| in-sample column | 2015-2020, same account settings |
| out-of-sample | 2021-01-04 .. 2025-07-24 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Holdout | Paper | Live |
|---|---:|---:|---:|---:|---:|---:|
| 1. CAGR (net) | -1.35% | n/a | 3.79% | n/a | n/a | n/a |
| 2. Annualised volatility | 2.23% | n/a | 5.67% | n/a | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | -0.60 | n/a | 0.68 | n/a | n/a | n/a |
| 4. Sortino ratio | -0.96 | n/a | 1.19 | n/a | n/a | n/a |
| 5. Calmar ratio | -0.13 | n/a | 0.74 | n/a | n/a | n/a |
| 6. Maximum drawdown | 10.03% | n/a | 5.13% | n/a | n/a | n/a |
| 6b. Max drawdown / annualised volatility (info) | 4.50 | n/a | 0.90 | n/a | n/a | n/a |
| 7. Profit factor (net) | 0.78 | n/a | 1.13 | n/a | n/a | n/a |
| 8. Win rate | 28.57% | n/a | 38.96% | n/a | n/a | n/a |
| 9. Avg win / avg loss | 1.96 | n/a | 1.78 | n/a | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | -1.06 | n/a | 1.05 | n/a | n/a | n/a |
| 10b. Net expectancy / trade (R) | -0.103 | n/a | 0.046 | n/a | n/a | n/a |
| 11. Independent trades | 364 | n/a | 865 | n/a | n/a | n/a |
| 12. Annualised turnover (x equity) | 78.5 | n/a | 331.6 | n/a | n/a | n/a |
| 13a. Transaction costs (EUR) | 698.56 | n/a | 849.65 | n/a | n/a | n/a |
| 13b. Financing costs (EUR) | 0.00 | n/a | 0.00 | n/a | n/a | n/a |
| 14a. ES 97.5% daily | 0.46% | n/a | 0.76% | n/a | n/a | n/a |
| 14b. ES 99% daily | 0.59% | n/a | 0.91% | n/a | n/a | n/a |
| 15a. MC P(DD >= 10%) | 0.99% | n/a | 13.01% | n/a | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.04% | n/a | 0.02% | n/a | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.01% | n/a | 0.00% | n/a | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.00% | n/a | 0.00% | n/a | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.970 | n/a | 0.991 | n/a | n/a | n/a |
| 15f. MC worst 99% drawdown | 9.97% | n/a | 15.17% | n/a | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 233 | n/a | 167 | n/a | n/a | n/a |
| Net P&L (EUR) | -386.03 | n/a | 911.32 | n/a | n/a | n/a |
| Sample length (years) | 5.92 | n/a | 4.50 | n/a | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it.

Out-of-Sample is the stitched walk-forward series; it was not used to choose anything in this run (earlier runs on the same data are a separate matter: see prior effective trials).

## Release gates (stitched walk-forward OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.6846 | > 0.80 | FAIL |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.2980 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | n/a | < 15% | N/A (no selection) |
| CSCV rank PBO | n/a | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (no selection) |
| OOS net profit factor | 1.1331 | > 1.20 | FAIL |
| OOS net P&L under 1.5x costs | 578.2400 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0513 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0002 | < 2% | PASS |
| OOS maximum drawdown at the 0.75% risk ceiling | 0.0680 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling | 0.0250 | < 2% | FAIL |
| Independent OOS trades | 865 | >= 100 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** One fixed configuration in this run, 22.4 effective trials carried from earlier research: N_eff = 23.4, SR0 = 0.0585 per period (variance 1/T), DSR **0.2980** (per-period SR 0.0431; against zero: 0.9313). The CSCV statistics are not applicable.

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 578.24 EUR, Sharpe 0.46, profit factor 1.09, max drawdown 5.69%

## Stitched OOS replayed at the 0.75% risk ceiling

Same signals and fold parameters, nothing re-estimated; only the risk budget per trade changes.

- Max drawdown 6.80%, annualised volatility 6.79%, max drawdown / volatility 1.00, Sharpe 0.30, net P&L 422.26 EUR
- Monte Carlo P(DD >= 10%) 51.48%, P(DD >= 20%) 2.50%, worst 99% drawdown 22.18%; breaker pauses 0, hard deactivation none

## Stability over time (informational, not a gate)

Non-overlapping 6-month slices of the stitched OOS curve: 8 of 9 profitable (88.9%), 1 losing, 0 with no trade (active slices: 9).

| Slice | Window | Net P&L | Trades | Note |
|---:|---|---:|---:|---|
| 0 | 2021-01-04 .. 2021-07-02 | 237.46 | 96 |  |
| 1 | 2021-07-06 .. 2022-01-03 | 49.28 | 90 |  |
| 2 | 2022-01-04 .. 2022-07-01 | 238.26 | 110 |  |
| 3 | 2022-07-05 .. 2023-01-03 | -41.17 | 105 |  |
| 4 | 2023-01-04 .. 2023-06-30 | 137.36 | 92 |  |
| 5 | 2023-07-05 .. 2024-01-03 | 9.81 | 96 |  |
| 6 | 2024-01-04 .. 2024-07-02 | 36.10 | 87 |  |
| 7 | 2024-07-05 .. 2025-01-03 | 278.88 | 94 |  |
| 8 | 2025-01-06 .. 2025-07-02 | 24.96 | 84 |  |
| 9 | 2025-07-07 .. 2025-07-24 | -59.61 | 11 | incomplete, not counted |
