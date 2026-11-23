# Phase 0 plan: volatility-managed NASDAQ-100 on the spent 2015–2025 sample

Written and committed before any result of this phase was computed.

## Question

The only v8 component that added anything on the 2015–2025 run was the RV20 regime filter. Is
that evidence that scaling NASDAQ-100 exposure down when volatility is high is worth doing for a
EUR 5,000 German retail account, or did the filter only stop a losing breakout system from trading
in turbulent weeks?

## What this phase can and cannot show

The sample is spent: it produced the idea. A failure here is informative (the idea does not even
work where it was found). A pass is weak evidence and earns only a pre-registration on untouched
markets, carrying N_eff_prior = 11.71 plus the variants examined here.

Nothing outside this archive is loaded except rates, FX, the NASDAQ-100 total-return index (same
market, for dividends) and public volatility indices for the breadth estimate. No strategy is run
on any other market.

## Data

- 1-minute cash-equivalent NASDAQ-100 bars, regular session, 2015-01-02 to 2025-07-24 (the
  archive behind `market.duckdb`; trade prices, no bid/ask).
- FRED: `NASDAQ100` and `NASDAQXNDX` (dividend yield = total return minus price return),
  `USD_RATE` (EFFR spliced into SOFR), `DEXUSEU`, `ECBESTRVOLWGTTRMDMNRT` (€STR; before
  October 2019 `EONIARATE` minus 8.5 bp).
- Spreads are the broker presets (1.8 points on US100), because no quotes exist.

## Primary specification (one trial)

| Element | Choice |
|---|---|
| Signal | RV20 as in v8: mean squared daily close-to-close log return over 20 sessions |
| Target weight | w* = c / RV20, with c re-estimated each day on an expanding window (minimum 252 sessions) so that the unlevered managed series has had the same volatility as buy-and-hold so far; no full-sample quantity enters |
| Timing | decided on data through the close of day t, traded at the open of day t+1 |
| Cap | 0 ≤ w ≤ 1.0 (Capital.com 1:1 mode, no overnight funding) |
| Deadband | rebalance only when the target differs from the current weight by more than 0.10 |
| Idle cash | 0% (EUR cash at the CFD broker) |
| Costs | half of 1.8 points per unit traded; 0.7% conversion on realised P&L, dividends and funding |
| Dividends | long CFD credited 100% of the index dividend |
| Tax | 26.375% on net realised CFD income per calendar year, losses carried forward, paid at year end; everything liquidated and taxed at the end of the sample |
| Window | first session after the 252-session burn-in to 2025-07-24; the v8 OOS span (from 2019-02-01) is reported beside it |

Benchmarks, both opened on the first day of the window and never rebalanced:

- **B1, same vehicle:** a 1:1 long CFD. Differences from the managed account are timing only.
- **B2, the real alternative:** an accumulating NASDAQ-100 UCITS ETF in EUR (TER 0.30%, dividends
  net of 15% withholding inside the fund, 6 bp spread, EUR 1 ticket), taxed once at liquidation at
  18.4625% (26.375% after the 30% equity partial exemption).

Post-tax daily series are on a liquidation basis: equity less the tax that would be due if the
account were closed that day. Their last value is the post-tax terminal wealth.

## Tests

1. Spanning regression of the managed account's daily post-tax return (less the idle-cash rate) on
   the benchmark's: Newey–West t(α), appraisal ratio.
2. Ledoit–Wolf (2008) studentised circular block bootstrap, 5,000 resamples, block 10 (5 and 21
   as checks): difference in Sharpe ratio, difference in mean-variance certainty equivalent at risk
   aversion 2, 3 and 5.
3. Post-tax terminal EUR wealth.

## Decision rule

**NO-GO** if, for the primary specification on the primary window, post-cost and post-tax, either
the certainty-equivalent difference at risk aversion 3 or the spanning α is not positive against
the ETF benchmark B2. Point estimates decide; significance is reported. B1 is the mechanism test
and is reported whichever way B2 falls. If the two windows disagree, that is stated and the primary
window decides.

A pass leads to a draft pre-registration and a stop for approval, nothing more.

## Mechanism tests (before any strategy accounting)

- Conservation: within-session 1-minute log returns plus the overnight gap equal the
  close-to-close log return, every session.
- Daily variance split into realised variance, bipower variation, jump variation, downside
  semivariance and squared overnight gap.
- Passive index: next-day and next-21-day return, variance and return per unit of variance by
  quintile of lagged RV20; regressions of the forward return and of return over variance on lagged
  RV20, HAR forecast, bipower variation and downside semivariance.
- The v8 filter: passive index return on days the filter allowed and blocked, against the P&L of
  the breakout trades the filter removed.

## Sensitivities (reported, not decisive; each is a further look at spent data)

Deadband 0, 0.05, 0.20; cap 1.5 with Capital.com funding (USD rate + 4.0% on the whole notional on
nights the weight exceeds 1) and with Pepperstone funding (+ 2.5%); idle cash swept into a €STR
ETF (index €STR + 8.5 bp, TER 0.10%, EUR 1 per transfer, only while the net rate is positive);
signals from intraday realised variance, HAR, bipower variation and downside semivariance;
inverse-volatility instead of inverse-variance weights; CFD dividend credit of 70%;
Vorabpauschale on the ETF; the saver's allowance (EUR 801, EUR 1,000 from 2023) available to
both accounts; in-sample scalar and frictionless cash for the literature comparison.

## Validator checks

- With weight 1 and no rebalancing the managed account equals the independently computed
  buy-and-hold to 1e-12; with weight 0 it equals compounded idle cash.
- Frictionless, the appraisal ratio does not depend on the scalar c, and the in-sample scalar gives
  the managed series exactly the benchmark's volatility. The move from in-sample scalar and
  risk-free cash to causal scalar and retail wedges is reported step by step.
- Size on simulated GJR-GARCH paths. The no-timing-value case is a premium proportional to
  variance (μ = λσ²), where the optimal weight is constant; rejection rates of α ≤ 0 and of a
  non-positive Sharpe difference must not exceed 5% there. Under a premium proportional to
  volatility (μ = λσ) timing does have value, so there the check is that the tests cover the true
  difference 95% of the time.

## Breadth

Eigenvalue participation ratio of the correlation matrix of weekly log changes in the public
equity volatility indices on FRED (VIX, VXN, VXD, RVX, VXEEM, VXEWZ; VXFXI to 2022). No European
or Japanese index is available there, which will be stated with the result.
