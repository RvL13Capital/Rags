# Bot A TWAP - Track 2: Pepperstone, 0.50% - schedule II - tearsheet

**Verdict:** NOT READY FOR LIVE DEPLOYMENT - gates not passed: Deflated Sharpe Ratio (zero-mean null, N_eff); Monte Carlo P(DD >= 20%)

Data sources: NQ futures 1m less fitted basis (cash-equivalent NASDAQ-100)

| Run setting | Value |
|---|---|
| rule | Bot A noise-area momentum, long and short, flat at 15:55 ET |
| anchor | TWAP |
| account | Track 2: Pepperstone, 0.50% |
| cost schedule | II (stress: III) |
| prior effective trials | 22.44 |
| in-sample column | 2015-2020, same account settings |
| out-of-sample | 2021-01-04 .. 2025-07-24 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Holdout | Paper | Live |
|---|---:|---:|---:|---:|---:|---:|
| 1. CAGR (net) | -0.76% | n/a | 5.11% | n/a | n/a | n/a |
| 2. Annualised volatility | 2.50% | n/a | 4.59% | n/a | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | -0.29 | n/a | 1.11 | n/a | n/a | n/a |
| 4. Sortino ratio | -0.50 | n/a | 2.04 | n/a | n/a | n/a |
| 5. Calmar ratio | -0.10 | n/a | 1.34 | n/a | n/a | n/a |
| 6. Maximum drawdown | 7.81% | n/a | 3.82% | n/a | n/a | n/a |
| 6b. Max drawdown / annualised volatility (info) | 3.13 | n/a | 0.83 | n/a | n/a | n/a |
| 7. Profit factor (net) | 0.90 | n/a | 1.24 | n/a | n/a | n/a |
| 8. Win rate | 32.01% | n/a | 40.10% | n/a | n/a | n/a |
| 9. Avg win / avg loss | 1.90 | n/a | 1.85 | n/a | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | -0.55 | n/a | 1.50 | n/a | n/a | n/a |
| 10b. Net expectancy / trade (R) | -0.044 | n/a | 0.069 | n/a | n/a | n/a |
| 11. Independent trades | 403 | n/a | 838 | n/a | n/a | n/a |
| 12. Annualised turnover (x equity) | 105.1 | n/a | 277.9 | n/a | n/a | n/a |
| 13a. Transaction costs (EUR) | 646.13 | n/a | 447.46 | n/a | n/a | n/a |
| 13b. Financing costs (EUR) | 0.00 | n/a | 0.00 | n/a | n/a | n/a |
| 14a. ES 97.5% daily | 0.47% | n/a | 0.61% | n/a | n/a | n/a |
| 14b. ES 99% daily | 0.57% | n/a | 0.72% | n/a | n/a | n/a |
| 15a. MC P(DD >= 10%) | 2.55% | n/a | 46.41% | n/a | n/a | n/a |
| 15b. MC P(DD >= 20%) | 0.12% | n/a | 2.84% | n/a | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.05% | n/a | 0.32% | n/a | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.01% | n/a | 0.07% | n/a | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.966 | n/a | 0.932 | n/a | n/a | n/a |
| 15f. MC worst 99% drawdown | 11.94% | n/a | 23.40% | n/a | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 231 | n/a | 226 | n/a | n/a | n/a |
| Net P&L (EUR) | -220.46 | n/a | 1,254.96 | n/a | n/a | n/a |
| Sample length (years) | 5.92 | n/a | 4.50 | n/a | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it.

Out-of-Sample is the stitched walk-forward series; it was not used to choose anything in this run (earlier runs on the same data are a separate matter: see prior effective trials).

## Release gates (stitched walk-forward OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 1.1080 | > 0.80 | PASS |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.6546 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | n/a | < 15% | N/A (no selection) |
| CSCV rank PBO | n/a | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (no selection) |
| OOS net profit factor | 1.2448 | > 1.20 | PASS |
| OOS net P&L under 1.5x costs | 1040.4236 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0382 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0284 | < 2% | FAIL |
| OOS maximum drawdown at the 0.75% risk ceiling | 0.0457 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling | 0.0017 | < 2% | PASS |
| Independent OOS trades | 838 | >= 100 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** One fixed configuration in this run, 22.4 effective trials carried from earlier research: N_eff = 23.4, SR0 = 0.0585 per period (variance 1/T), DSR **0.6546** (per-period SR 0.0698; against zero: 0.9931). The CSCV statistics are not applicable.

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 1,040.42 EUR, Sharpe 0.93, profit factor 1.20, max drawdown 3.95%

## Stitched OOS replayed at the 0.75% risk ceiling

Same signals and fold parameters, nothing re-estimated; only the risk budget per trade changes.

- Max drawdown 4.57%, annualised volatility 6.61%, max drawdown / volatility 0.69, Sharpe 0.99, net P&L 1,648.24 EUR
- Monte Carlo P(DD >= 10%) 25.98%, P(DD >= 20%) 0.17%, worst 99% drawdown 17.32%; breaker pauses 0, hard deactivation none

## Stability over time (informational, not a gate)

Non-overlapping 6-month slices of the stitched OOS curve: 8 of 9 profitable (88.9%), 1 losing, 0 with no trade (active slices: 9).

| Slice | Window | Net P&L | Trades | Note |
|---:|---|---:|---:|---|
| 0 | 2021-01-04 .. 2021-07-02 | 179.14 | 96 |  |
| 1 | 2021-07-06 .. 2022-01-03 | 73.45 | 91 |  |
| 2 | 2022-01-04 .. 2022-07-01 | 330.02 | 88 |  |
| 3 | 2022-07-05 .. 2023-01-03 | -42.89 | 98 |  |
| 4 | 2023-01-04 .. 2023-06-30 | 208.55 | 91 |  |
| 5 | 2023-07-05 .. 2024-01-03 | 34.61 | 97 |  |
| 6 | 2024-01-04 .. 2024-07-02 | 108.92 | 89 |  |
| 7 | 2024-07-05 .. 2025-01-03 | 279.55 | 97 |  |
| 8 | 2025-01-06 .. 2025-07-02 | 134.96 | 79 |  |
| 9 | 2025-07-07 .. 2025-07-24 | -51.35 | 12 | incomplete, not counted |
