# Phase 0: volatility-managed NASDAQ-100 on the spent 2015–2025 sample

**Final verdict: NO-GO. The hypothesis is archived and no untouched market will be spent on it.**
The pre-declared specification (inverse-variance weight on
RV20, capped at 1.0, in a EUR 5,000 Capital.com 1:1 account, after costs and German tax) has a
positive spanning α but a negative certainty-equivalent difference against buy-and-hold, and the
rule in [`PLAN.md`](PLAN.md) needs both to be positive. No untouched market was loaded.

**One result outside the plan complicates that.** The filter that actually sat in v8 (hold the
index unless RV20 is above the 80th percentile of its own trailing year) was checked afterwards as
an in-or-out rule. It clears both legs of the rule on point estimates. It is fragile and was looked
at after the planned results were known. The owner of this research was shown both results and
decided to hold to the committed plan: the planned rule failed, so the verdict is final. The filter
result stays on record below as an observation, not as a hypothesis in waiting.

The sample is spent. Nothing here is evidence that either rule works out of sample.

## 1. Mechanism

**Conservation.** Over 2,652 sessions the 1-minute log returns of the session plus the overnight
gap equal the close-to-close log return to 1.9e-15, and the 1-minute opens and closes equal the
hourly bars the backtests used exactly.

**Where the variance sits.** Annualised volatility 22.4% close to close.

| Part of daily variance | Share |
|---|---:|
| Intraday, continuous (bipower variation) | 57.1% |
| Intraday jumps | 4.5% |
| Overnight gap | 38.5% |

Downside moves are 50.4% of intraday variance. The overnight gap earned 10.0% a year of the index's
log return, the session itself 6.1%.

**Does RV20 predict lower returns, or only higher variance?** Only higher variance. Passive index,
next-day excess return including dividends, by quintile of lagged RV20:

| Quintile of RV20 | RV20 (annual vol) | Mean return (annual) | Next-day vol (annual) | Return per unit of variance | Share of total return |
|---:|---:|---:|---:|---:|---:|
| 1 (calm) | 9.7% | 15.8% | 12.0% | 11.0 | 18% |
| 2 | 13.4% | 9.3% | 14.0% | 4.7 | 11% |
| 3 | 16.9% | 19.7% | 16.4% | 7.3 | 24% |
| 4 | 22.5% | 22.2% | 24.6% | 3.7 | 25% |
| 5 (turbulent) | 38.0% | 22.7% | 34.0% | 2.0 | 21% |

- The mean return does not fall with volatility; if anything it rises (slope +14% a year per
  standard deviation of RV20, t = 1.7).
- Variance is highly predictable (elasticity 0.78, R² 0.44), so return per unit of variance falls
  from 11 to 2. That is the textbook condition for volatility management to add α.
- The two most turbulent quintiles produced 46% of the index's total return. Scaling down there
  gives that up.
- Intraday realised variance, bipower variation, downside semivariance and a HAR forecast behave
  the same way (`results/predictive_regressions.csv`).

**What did the v8 filter do?** Both things. February 2019 to July 2025:

| Filter state | Sessions | Passive index, mean (annual) | Passive vol | Breakout trades | Breakout mean R | Breakout net |
|---|---:|---:|---:|---:|---:|---:|
| Allows | 1,285 | +25.0% | 19.7% | 139 | +0.086 | +EUR 121 |
| Blocks | 341 | +1.7% | 38.4% | 32 | −0.458 | −EUR 149 |

The filter removed breakout trades that lost 0.46R each, and the days it blocked were also poor for
the passive index. It did not forfeit the rebounds, because it ranks RV20 against its own trailing
year: after a spike the percentile falls back while the level is still high. The difference in
passive means is not statistically distinguishable from zero (t = −0.9; −1.2 from 2016).

## 2. The EUR 5,000 account

Primary window 2016-02-03 to 2025-07-24 (after the 252-session burn-in of the real-time scalar).

| Account | No tax: terminal | Post-tax: terminal | CAGR | Vol | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Managed, primary specification | EUR 15,282 | EUR 11,789 | 9.5% | 8.8% | 1.07 | 12.9% |
| B1: buy-and-hold 1:1 CFD | EUR 27,108 | EUR 21,277 | 16.6% | 19.0% | 0.90 | 28.6% |
| B2: buy-and-hold accumulating ETF | EUR 27,116 | EUR 23,033 | 17.6% | 21.2% | 0.87 | 29.4% |

Post-tax figures are on a liquidation basis. The managed account held an average weight of 0.63,
rebalanced 26 times a year and paid EUR 29 in spread, EUR 87 in conversion mark-up and EUR 2,432
in tax.

**Frictions, one at a time**

- **Idle cash.** Sweeping it into a €STR ETF adds EUR 149 over 9.5 years; euro rates were negative
  until mid-2022 and each transfer costs a ticket.
- **The 1.0× cliff.** Allowing weights up to 1.5 costs EUR 1,815 in funding at Capital.com
  (EUR 1,423 at Pepperstone) because the charge falls on the whole notional, and leaves the account
  worse off: terminal EUR 10,764, α halved, Sharpe below buy-and-hold. This models switching in
  and out of leveraged mode freely, which is generous.
- **Tax.** Annual taxation at 26.375% costs the managed account 23% of its untaxed terminal
  wealth; deferral at 18.4625% costs the ETF 15%. In certainty-equivalent terms the asymmetry is
  about 1.4 points a year.
- **Saver's allowance (not in your specification).** If EUR 801 / 1,000 a year is unused elsewhere,
  the managed account pays EUR 1,014 instead of EUR 2,432 and ends at EUR 13,971. Buy-and-hold
  still ends far ahead (EUR 23,296).
- **Vorabpauschale.** Changes the ETF's terminal wealth by EUR 18.
- **Deadband.** 0, 0.05, 0.10 and 0.20 give terminal wealth of EUR 11,815, 11,654, 11,789 and
  11,257. Spread is small either way; the band mostly delays the weight.

## 3. Benchmark-relative tests

The v8 gates count round trips and compare gross wins with gross losses. A continuous weight has no
round trips: "trades ≥ 100" and "profit factor > 1.20" would be measuring rebalancing noise. What
can be tested is whether the managed account adds to holding the benchmark.

Primary specification, post-tax, 5,000 bootstrap resamples:

| Test | Against B1 (same vehicle) | Against B2 (ETF) |
|---|---:|---:|
| Spanning α (annual) | +3.1%, t = 1.82 | +3.5%, t = 1.98 |
| β | 0.37 | 0.32 |
| Appraisal ratio | 0.58 | 0.63 |
| Sharpe difference | +0.17, p = 0.18 | +0.20, p = 0.15 |
| Certainty-equivalent difference, risk aversion 2 | −4.9%, p = 0.90 | −5.3%, p = 0.88 |
| risk aversion 3 | −3.4%, p = 0.81 | **−3.4%**, p = 0.77 |
| risk aversion 5 | −0.6%, p = 0.55 | +0.3%, p = 0.47 |
| Post-tax terminal wealth | −EUR 9,488 | −EUR 11,244 |

p-values are one-sided for "managed is better". The v8 OOS span (from February 2019) gives the same
picture: α +3.3% (t = 1.57), certainty-equivalent difference −3.5% at risk aversion 3.

- **Why α is positive and the certainty equivalent is not.** α says the timing beats a fixed
  mix of the index and cash with the same β. The certainty equivalent compares it with holding the
  whole index. At risk aversion 3 an investor wants about 1.4× this index, so an account that
  averages 0.63× loses however good its timing is. Levering the managed account back up is what the
  funding cliff prices out.
- **Every planned sensitivity fails the same leg.** All twelve have a negative certainty-equivalent
  difference at risk aversion 3 (−1.0% to −4.7%) and a positive α (`results/accounts_table.csv`).
  Inverse-volatility weights come closest (−1.0%).
- **The α itself is weak evidence.** It is on the sample that suggested the idea, and the size
  check below shows the t-statistic is somewhat liberal.

## 4. Validator checks

**Conservation limits** (`tests/test_volmanaged.py`). With weight 1 and no rebalancing the
simulator equals an independently written buy-and-hold formula to below 1e-12; with weight 0 it
equals compounded idle cash to below 1e-12.

**From the textbook to the retail account.** Spanning α of the RV20-managed index, primary window:

| Step | α (annual) | t | Mean weight |
|---|---:|---:|---:|
| 1. Moreira–Muir: in-sample scalar, no cap, frictionless | 9.5% | 1.62 | 1.20 |
| 2. Real-time scalar (Cederburg et al.) | 6.0% | 1.50 | 0.87 |
| 3. + cap 1.5 | 5.0% | 1.70 | 0.74 |
| 4. + cap 1.0 | 4.1% | 1.80 | 0.64 |
| 5. EUR 1:1 CFD account, traded at the next open, idle cash 0% | 4.9% | 2.09 | 0.63 |
| 6. + spread and conversion mark-up | 4.8% | 2.03 | 0.63 |
| 7. + deadband 0.10 | 4.4% | 1.96 | 0.63 |
| 8. + German tax, against same-vehicle buy-and-hold | 3.1% | 1.82 | 0.63 |

The in-sample scalar flatters α by more than a third, as the literature reports; the cap and tax
take most of the rest. Two exact properties are tested: the appraisal ratio does not depend on the
scalar, and the in-sample scalar gives the managed series exactly the benchmark's volatility.

**Size on simulated GJR-GARCH paths** (1,000 paths of 2,653 days, 1,000 resamples each):

| Process | True α | True Sharpe difference | Rejects α ≤ 0 | Rejects Sharpe difference ≤ 0 | Rejects the true α (two-sided) | Rejects the true Sharpe difference (two-sided) |
|---|---:|---:|---:|---:|---:|---:|
| Premium ∝ variance (no timing value), no cap | −0.2% | −0.21 | 5.6% | 0.6% | 5.9% | 4.3% |
| Premium ∝ variance, cap 1.0 | 0.0% | −0.11 | 5.5% | 0.9% | 6.0% | 3.8% |
| Premium ∝ volatility (your process), no cap | +3.0% | −0.05 | 17.7% | 3.5% | 8.8% | 6.0% |
| Premium ∝ volatility, cap 1.0 | +1.8% | +0.02 | 18.2% | 7.9% | 9.6% | 5.9% |

- Where timing has no value, both tests reject about 5% of the time or less (Monte Carlo error
  ±0.7 points).
- A premium proportional to volatility is not a null: the true α is +1.8% to +3.0% a year, so the
  17–18% rejections are power, not false positives.
- The Sharpe test covers the true difference as it should. The HAC t(α) over-rejects the true
  value there (9–10% against 5%), so a t of 2 on real data deserves less weight than its face
  value.

## 5. Breadth

Participation ratio of weekly log changes in public equity volatility indices, 2015 to July 2025:

| Basket | Effective independent regimes | Mean pairwise correlation |
|---|---:|---:|
| 4 US indices (NASDAQ-100, S&P 500, Dow, Russell 2000) | 1.18 | 0.90 |
| 6 indices (those plus emerging markets and Brazil) | 1.51 | 0.76 |
| 7 indices, adding China (to February 2022) | 1.59 | 0.74 |

A four-to-six index basket adds between 0.2 and 0.6 of an independent volatility regime to the
NASDAQ-100. FRED carries no European or Japanese volatility index, so developed markets outside
the US are not in this estimate; they are unlikely to behave very differently from the emerging
markets index.

## The result outside the plan

After the planned results, the literal v8 filter was run as a passive rule: fully invested while
RV20 is at or below the 80th percentile of its trailing 252 sessions, in cash otherwise, same
account, costs and tax. This was not in `PLAN.md`.

| Threshold | Window | Terminal (post-tax) | ETF | α vs ETF | t | Certainty-equivalent difference (risk aversion 3) | Max drawdown |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.80 | 2016-02 to 2025-07 | EUR 21,077 | EUR 23,033 | +8.6% | 2.75 | +1.8% (p = 0.34) | 21.0% |
| 0.80 | 2019-02 to 2025-07 | EUR 14,352 | EUR 14,728 | +9.0% | 2.14 | +2.3% (p = 0.34) | 21.0% |
| 0.80 | 2016-02 to 2019-01 | EUR 7,348 | EUR 7,491 | +8.2% | 2.13 | +1.2% (p = 0.44) | 7.1% |
| 0.70 | 2016-02 to 2025-07 | EUR 19,210 | EUR 23,033 | +8.9% | 3.02 | +1.2% | 17.3% |
| 0.90 | 2016-02 to 2025-07 | EUR 14,345 | EUR 23,033 | +2.4% | 0.69 | −3.0% | 45.1% |

- **For it:** it is the rule that showed the pulse; the 0.80 threshold was fixed in the code before
  any real data was loaded; the 2016–2018 span, which the v8 ablation never scored, points the same
  way; it clears both legs of the decision rule on point estimates.
- **Against it:** it was examined after the planned specification failed; at a threshold of 0.90 it
  fails badly; the difference in passive returns between allowed and blocked days is not
  significant (t = −1.2); no bootstrap test reaches 5% (Sharpe difference p = 0.08); it still ends
  with less money than the ETF; and with about 1.5 effective regimes across a six-index basket and
  11.71 prior trials, a pre-registered test would have little power.

## Trials

One planned primary, twelve planned sensitivities and three thresholds of the unplanned filter,
all on spent data. They are recorded in `docs/governance/prior_trials.md`.

## Reproducing

```bash
python prep/build_daily.py ndx_cash_equiv_1m.parquet market.duckdb data
```

```bash
python prep/gate_decomposition.py market.duckdb feature_cache data
```

```bash
python run_phase0.py . 5000
```

```bash
python calibration.py results 1000 1000 4
```

```bash
python breadth.py results
```

The first two need the archive of the 2015–2025 run; their outputs are committed in `data/`, so the
last three run from the repository alone.
