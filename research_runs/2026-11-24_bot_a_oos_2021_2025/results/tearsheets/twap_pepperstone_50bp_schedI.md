# Bot A TWAP - Track 2: Pepperstone, 0.50% - schedule I - tearsheet

**Verdict:** NOT READY FOR LIVE DEPLOYMENT - gates not passed: Deflated Sharpe Ratio (zero-mean null, N_eff); Monte Carlo P(DD >= 20%)

Data sources: NQ futures 1m less fitted basis (cash-equivalent NASDAQ-100)

| Run setting | Value |
|---|---|
| rule | Bot A noise-area momentum, long and short, flat at 15:55 ET |
| anchor | TWAP |
| account | Track 2: Pepperstone, 0.50% |
| cost schedule | I (stress: I x1.5) |
| prior effective trials | 22.44 |
| in-sample column | 2015-2020, same account settings |
| out-of-sample | 2021-01-04 .. 2025-07-24 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Holdout | Paper | Live |
|---|---:|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 5.83% | n/a | 5.43% | n/a | n/a | n/a |
| 2. Annualised volatility | 4.88% | n/a | 4.59% | n/a | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 1.18 | n/a | 1.18 | n/a | n/a | n/a |
| 4. Sortino ratio | 2.46 | n/a | 2.18 | n/a | n/a | n/a |
| 5. Calmar ratio | 0.98 | n/a | 1.44 | n/a | n/a | n/a |
| 6. Maximum drawdown | 5.95% | n/a | 3.77% | n/a | n/a | n/a |
| 6b. Max drawdown / annualised volatility (info) | 1.22 | n/a | 0.82 | n/a | n/a | n/a |
| 7. Profit factor (net) | 1.29 | n/a | 1.26 | n/a | n/a | n/a |
| 8. Win rate | 36.65% | n/a | 40.67% | n/a | n/a | n/a |
| 9. Avg win / avg loss | 2.23 | n/a | 1.84 | n/a | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 1.89 | n/a | 1.60 | n/a | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.089 | n/a | 0.076 | n/a | n/a | n/a |
| 11. Independent trades | 1056 | n/a | 841 | n/a | n/a | n/a |
| 12. Annualised turnover (x equity) | 315.5 | n/a | 279.0 | n/a | n/a | n/a |
| 13a. Transaction costs (EUR) | 455.93 | n/a | 287.10 | n/a | n/a | n/a |
| 13b. Financing costs (EUR) | 0.00 | n/a | 0.00 | n/a | n/a | n/a |
| 14a. ES 97.5% daily | 0.57% | n/a | 0.60% | n/a | n/a | n/a |
| 14b. ES 99% daily | 0.66% | n/a | 0.71% | n/a | n/a | n/a |
| 15a. MC P(DD >= 10%) | 48.18% | n/a | 44.26% | n/a | n/a | n/a |
| 15b. MC P(DD >= 20%) | 2.20% | n/a | 2.76% | n/a | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.30% | n/a | 0.37% | n/a | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.09% | n/a | 0.07% | n/a | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.924 | n/a | 0.936 | n/a | n/a | n/a |
| 15f. MC worst 99% drawdown | 22.97% | n/a | 23.65% | n/a | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 234 | n/a | 225 | n/a | n/a | n/a |
| Net P&L (EUR) | 1,993.15 | n/a | 1,343.17 | n/a | n/a | n/a |
| Sample length (years) | 5.92 | n/a | 4.50 | n/a | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it.

Out-of-Sample is the stitched walk-forward series; it was not used to choose anything in this run (earlier runs on the same data are a separate matter: see prior effective trials).

## Release gates (stitched walk-forward OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 1.1765 | > 0.80 | PASS |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.7093 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | n/a | < 15% | N/A (no selection) |
| CSCV rank PBO | n/a | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (no selection) |
| OOS net profit factor | 1.2615 | > 1.20 | PASS |
| OOS net P&L under 1.5x costs | 1265.0744 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0377 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0276 | < 2% | FAIL |
| OOS maximum drawdown at the 0.75% risk ceiling | 0.0421 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling | 0.0015 | < 2% | PASS |
| Independent OOS trades | 841 | >= 100 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** One fixed configuration in this run, 22.4 effective trials carried from earlier research: N_eff = 23.4, SR0 = 0.0585 per period (variance 1/T), DSR **0.7093** (per-period SR 0.0741; against zero: 0.9956). The CSCV statistics are not applicable.

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 1,265.07 EUR, Sharpe 1.11, profit factor 1.25, max drawdown 3.84%

## Stitched OOS replayed at the 0.75% risk ceiling

Same signals and fold parameters, nothing re-estimated; only the risk budget per trade changes.

- Max drawdown 4.21%, annualised volatility 6.65%, max drawdown / volatility 0.63, Sharpe 1.16, net P&L 2,003.89 EUR
- Monte Carlo P(DD >= 10%) 19.32%, P(DD >= 20%) 0.15%, worst 99% drawdown 16.63%; breaker pauses 0, hard deactivation none

## Stability over time (informational, not a gate)

Non-overlapping 6-month slices of the stitched OOS curve: 8 of 9 profitable (88.9%), 1 losing, 0 with no trade (active slices: 9).

| Slice | Window | Net P&L | Trades | Note |
|---:|---|---:|---:|---|
| 0 | 2021-01-04 .. 2021-07-02 | 203.01 | 96 |  |
| 1 | 2021-07-06 .. 2022-01-03 | 90.67 | 91 |  |
| 2 | 2022-01-04 .. 2022-07-01 | 300.84 | 91 |  |
| 3 | 2022-07-05 .. 2023-01-03 | -19.53 | 98 |  |
| 4 | 2023-01-04 .. 2023-06-30 | 229.69 | 91 |  |
| 5 | 2023-07-05 .. 2024-01-03 | 60.00 | 97 |  |
| 6 | 2024-01-04 .. 2024-07-02 | 111.69 | 89 |  |
| 7 | 2024-07-05 .. 2025-01-03 | 277.77 | 97 |  |
| 8 | 2025-01-06 .. 2025-07-02 | 139.67 | 79 |  |
| 9 | 2025-07-07 .. 2025-07-24 | -50.65 | 12 | incomplete, not counted |
