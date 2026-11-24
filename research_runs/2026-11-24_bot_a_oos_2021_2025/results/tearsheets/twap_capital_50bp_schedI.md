# Bot A TWAP - Track 2: Capital.com, 0.50% - schedule I - tearsheet

**Verdict:** NOT READY FOR LIVE DEPLOYMENT - gates not passed: Deflated Sharpe Ratio (zero-mean null, N_eff)

Data sources: NQ futures 1m less fitted basis (cash-equivalent NASDAQ-100)

| Run setting | Value |
|---|---|
| rule | Bot A noise-area momentum, long and short, flat at 15:55 ET |
| anchor | TWAP |
| account | Track 2: Capital.com, 0.50% |
| cost schedule | I (stress: I x1.5) |
| prior effective trials | 22.44 |
| in-sample column | 2015-2020, same account settings |
| out-of-sample | 2021-01-04 .. 2025-07-24 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Holdout | Paper | Live |
|---|---:|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 5.58% | n/a | 6.10% | n/a | n/a | n/a |
| 2. Annualised volatility | 5.38% | n/a | 5.81% | n/a | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 1.04 | n/a | 1.05 | n/a | n/a | n/a |
| 4. Sortino ratio | 2.11 | n/a | 1.89 | n/a | n/a | n/a |
| 5. Calmar ratio | 0.83 | n/a | 1.44 | n/a | n/a | n/a |
| 6. Maximum drawdown | 6.73% | n/a | 4.23% | n/a | n/a | n/a |
| 6b. Max drawdown / annualised volatility (info) | 1.25 | n/a | 0.73 | n/a | n/a | n/a |
| 7. Profit factor (net) | 1.25 | n/a | 1.21 | n/a | n/a | n/a |
| 8. Win rate | 36.08% | n/a | 40.80% | n/a | n/a | n/a |
| 9. Avg win / avg loss | 2.21 | n/a | 1.76 | n/a | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 1.76 | n/a | 1.73 | n/a | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.074 | n/a | 0.064 | n/a | n/a | n/a |
| 11. Independent trades | 1081 | n/a | 880 | n/a | n/a | n/a |
| 12. Annualised turnover (x equity) | 333.8 | n/a | 354.7 | n/a | n/a | n/a |
| 13a. Transaction costs (EUR) | 786.37 | n/a | 632.37 | n/a | n/a | n/a |
| 13b. Financing costs (EUR) | 0.00 | n/a | 0.00 | n/a | n/a | n/a |
| 14a. ES 97.5% daily | 0.65% | n/a | 0.74% | n/a | n/a | n/a |
| 14b. ES 99% daily | 0.75% | n/a | 0.90% | n/a | n/a | n/a |
| 15a. MC P(DD >= 10%) | 53.98% | n/a | 8.21% | n/a | n/a | n/a |
| 15b. MC P(DD >= 20%) | 3.37% | n/a | 0.02% | n/a | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.31% | n/a | 0.00% | n/a | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.11% | n/a | 0.00% | n/a | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.918 | n/a | 1.013 | n/a | n/a | n/a |
| 15f. MC worst 99% drawdown | 24.21% | n/a | 13.57% | n/a | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 234 | n/a | 141 | n/a | n/a | n/a |
| Net P&L (EUR) | 1,898.40 | n/a | 1,524.19 | n/a | n/a | n/a |
| Sample length (years) | 5.92 | n/a | 4.50 | n/a | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it.

Out-of-Sample is the stitched walk-forward series; it was not used to choose anything in this run (earlier runs on the same data are a separate matter: see prior effective trials).

## Release gates (stitched walk-forward OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 1.0470 | > 0.80 | PASS |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.6025 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | n/a | < 15% | N/A (no selection) |
| CSCV rank PBO | n/a | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (no selection) |
| OOS net profit factor | 1.2132 | > 1.20 | PASS |
| OOS net P&L under 1.5x costs | 1119.6522 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0423 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0002 | < 2% | PASS |
| OOS maximum drawdown at the 0.75% risk ceiling | 0.0558 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling | 0.0092 | < 2% | PASS |
| Independent OOS trades | 880 | >= 100 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** One fixed configuration in this run, 22.4 effective trials carried from earlier research: N_eff = 23.4, SR0 = 0.0585 per period (variance 1/T), DSR **0.6025** (per-period SR 0.0660; against zero: 0.9894). The CSCV statistics are not applicable.

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 1,119.65 EUR, Sharpe 0.82, profit factor 1.16, max drawdown 4.58%

## Stitched OOS replayed at the 0.75% risk ceiling

Same signals and fold parameters, nothing re-estimated; only the risk budget per trade changes.

- Max drawdown 5.58%, annualised volatility 7.75%, max drawdown / volatility 0.72, Sharpe 0.98, net P&L 1,939.14 EUR
- Monte Carlo P(DD >= 10%) 32.38%, P(DD >= 20%) 0.92%, worst 99% drawdown 19.76%; breaker pauses 0, hard deactivation none

## Stability over time (informational, not a gate)

Non-overlapping 6-month slices of the stitched OOS curve: 9 of 9 profitable (100.0%), 0 losing, 0 with no trade (active slices: 9).

| Slice | Window | Net P&L | Trades | Note |
|---:|---|---:|---:|---|
| 0 | 2021-01-04 .. 2021-07-02 | 276.64 | 97 |  |
| 1 | 2021-07-06 .. 2022-01-03 | 118.91 | 91 |  |
| 2 | 2022-01-04 .. 2022-07-01 | 336.03 | 111 |  |
| 3 | 2022-07-05 .. 2023-01-03 | 11.72 | 106 |  |
| 4 | 2023-01-04 .. 2023-06-30 | 247.22 | 93 |  |
| 5 | 2023-07-05 .. 2024-01-03 | 95.04 | 97 |  |
| 6 | 2024-01-04 .. 2024-07-02 | 140.95 | 90 |  |
| 7 | 2024-07-05 .. 2025-01-03 | 301.43 | 98 |  |
| 8 | 2025-01-06 .. 2025-07-02 | 66.32 | 85 |  |
| 9 | 2025-07-07 .. 2025-07-24 | -70.06 | 12 | incomplete, not counted |
