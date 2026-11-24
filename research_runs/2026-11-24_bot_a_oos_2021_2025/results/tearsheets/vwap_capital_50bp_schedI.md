# Bot A VWAP - Track 2: Capital.com, 0.50% - schedule I - tearsheet

**Verdict:** NOT READY FOR LIVE DEPLOYMENT - gates not passed: Deflated Sharpe Ratio (zero-mean null, N_eff); OOS net profit factor; Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling

Data sources: NQ futures 1m less fitted basis (cash-equivalent NASDAQ-100)

| Run setting | Value |
|---|---|
| rule | Bot A noise-area momentum, long and short, flat at 15:55 ET |
| anchor | VWAP |
| account | Track 2: Capital.com, 0.50% |
| cost schedule | I (stress: I x1.5) |
| prior effective trials | 22.44 |
| in-sample column | 2015-2020, same account settings |
| out-of-sample | 2021-01-04 .. 2025-07-24 |

## Performance by segment (never pooled)

| Metric | In-Sample | Validation | Out-of-Sample | Holdout | Paper | Live |
|---|---:|---:|---:|---:|---:|---:|
| 1. CAGR (net) | 4.67% | n/a | 5.11% | n/a | n/a | n/a |
| 2. Annualised volatility | 5.15% | n/a | 5.73% | n/a | n/a | n/a |
| 3. Sharpe ratio (net, rf=0) | 0.91 | n/a | 0.90 | n/a | n/a | n/a |
| 4. Sortino ratio | 1.82 | n/a | 1.59 | n/a | n/a | n/a |
| 5. Calmar ratio | 0.68 | n/a | 1.03 | n/a | n/a | n/a |
| 6. Maximum drawdown | 6.88% | n/a | 4.94% | n/a | n/a | n/a |
| 6b. Max drawdown / annualised volatility (info) | 1.34 | n/a | 0.86 | n/a | n/a | n/a |
| 7. Profit factor (net) | 1.22 | n/a | 1.18 | n/a | n/a | n/a |
| 8. Win rate | 35.08% | n/a | 40.00% | n/a | n/a | n/a |
| 9. Avg win / avg loss | 2.27 | n/a | 1.77 | n/a | n/a | n/a |
| 10a. Net expectancy / trade (EUR) | 1.45 | n/a | 1.45 | n/a | n/a | n/a |
| 10b. Net expectancy / trade (R) | 0.069 | n/a | 0.059 | n/a | n/a | n/a |
| 11. Independent trades | 1069 | n/a | 865 | n/a | n/a | n/a |
| 12. Annualised turnover (x equity) | 303.4 | n/a | 336.8 | n/a | n/a | n/a |
| 13a. Transaction costs (EUR) | 689.83 | n/a | 591.26 | n/a | n/a | n/a |
| 13b. Financing costs (EUR) | 0.00 | n/a | 0.00 | n/a | n/a | n/a |
| 14a. ES 97.5% daily | 0.65% | n/a | 0.76% | n/a | n/a | n/a |
| 14b. ES 99% daily | 0.75% | n/a | 0.91% | n/a | n/a | n/a |
| 15a. MC P(DD >= 10%) | 46.17% | n/a | 10.48% | n/a | n/a | n/a |
| 15b. MC P(DD >= 20%) | 1.90% | n/a | 0.00% | n/a | n/a | n/a |
| 15c. MC P(DD >= 30%) | 0.20% | n/a | 0.00% | n/a | n/a | n/a |
| 15d. MC P(ruin / forced stop) | 0.03% | n/a | 0.00% | n/a | n/a | n/a |
| 15e. MC median terminal wealth (x) | 0.926 | n/a | 1.002 | n/a | n/a | n/a |
| 15f. MC worst 99% drawdown | 21.91% | n/a | 14.17% | n/a | n/a | n/a |
| 15g. MC median contiguous recovery (days) | 233 | n/a | 153 | n/a | n/a | n/a |
| Net P&L (EUR) | 1,553.96 | n/a | 1,255.67 | n/a | n/a | n/a |
| Sample length (years) | 5.92 | n/a | 4.50 | n/a | n/a | n/a |

Monte Carlo rows show the most adverse of the three generators (block bootstrap, Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.

Validation is the span the hyper-parameters were selected on (optimistic by construction); In-Sample is the model-estimation span with the parameters later chosen for it.

Out-of-Sample is the stitched walk-forward series; it was not used to choose anything in this run (earlier runs on the same data are a separate matter: see prior effective trials).

## Release gates (stitched walk-forward OOS)

| Gate | Value | Rule | Status |
|---|---:|---|---|
| OOS net Sharpe | 0.8985 | > 0.80 | PASS |
| Deflated Sharpe Ratio (zero-mean null, N_eff) | 0.4735 | > 0.95 | FAIL |
| CSCV P(OOS loss of IS-best) | n/a | < 15% | N/A (no selection) |
| CSCV rank PBO | n/a | < 0.25 when tuned and IS Sharpe IQR > 0.25 | N/A (no selection) |
| OOS net profit factor | 1.1799 | > 1.20 | FAIL |
| OOS net P&L under 1.5x costs | 1001.0936 | > 0 | PASS |
| OOS maximum drawdown (true high-water mark) | 0.0494 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) | 0.0000 | < 2% | PASS |
| OOS maximum drawdown at the 0.75% risk ceiling | 0.0635 | < 10% | PASS |
| Monte Carlo P(DD >= 20%) at the 0.75% risk ceiling | 0.0263 | < 2% | FAIL |
| Independent OOS trades | 865 | >= 100 | PASS |

Circuit breakers in the OOS run: 0 pause(s) at -8%; hard deactivation at -10% of the all-time peak: none. Maximum drawdown is always measured from the un-rebased high-water mark.

## Overfitting diagnostics

- **Gate.** One fixed configuration in this run, 22.4 effective trials carried from earlier research: N_eff = 23.4, SR0 = 0.0585 per period (variance 1/T), DSR **0.4735** (per-period SR 0.0566; against zero: 0.9753). The CSCV statistics are not applicable.

## OOS under 1.5x spread / slippage / commission / financing markup

- Net P&L 1,001.09 EUR, Sharpe 0.74, profit factor 1.14, max drawdown 5.11%

## Stitched OOS replayed at the 0.75% risk ceiling

Same signals and fold parameters, nothing re-estimated; only the risk budget per trade changes.

- Max drawdown 6.35%, annualised volatility 7.26%, max drawdown / volatility 0.87, Sharpe 0.66, net P&L 1,136.68 EUR
- Monte Carlo P(DD >= 10%) 47.60%, P(DD >= 20%) 2.63%, worst 99% drawdown 22.14%; breaker pauses 0, hard deactivation none

## Stability over time (informational, not a gate)

Non-overlapping 6-month slices of the stitched OOS curve: 9 of 9 profitable (100.0%), 0 losing, 0 with no trade (active slices: 9).

| Slice | Window | Net P&L | Trades | Note |
|---:|---|---:|---:|---|
| 0 | 2021-01-04 .. 2021-07-02 | 281.67 | 96 |  |
| 1 | 2021-07-06 .. 2022-01-03 | 92.26 | 90 |  |
| 2 | 2022-01-04 .. 2022-07-01 | 274.20 | 110 |  |
| 3 | 2022-07-05 .. 2023-01-03 | 5.93 | 105 |  |
| 4 | 2023-01-04 .. 2023-06-30 | 223.68 | 92 |  |
| 5 | 2023-07-05 .. 2024-01-03 | 51.75 | 96 |  |
| 6 | 2024-01-04 .. 2024-07-02 | 33.11 | 87 |  |
| 7 | 2024-07-05 .. 2025-01-03 | 319.31 | 94 |  |
| 8 | 2025-01-06 .. 2025-07-02 | 35.74 | 84 |  |
| 9 | 2025-07-07 .. 2025-07-24 | -61.98 | 11 | incomplete, not counted |
