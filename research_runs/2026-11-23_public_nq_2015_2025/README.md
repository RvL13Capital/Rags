# First real-data run: public NQ tape, 2015–2025

**Verdict: NOT READY FOR LIVE DEPLOYMENT.** All four configurations fail the locked release
gates. The pre-declared primary configuration fails six of ten. Capital stays in cash.

**Later note.** The gates quoted below are the ones in force when the run was made. The release
battery was revised once afterwards (profitable folds became informational, the trade floor went
from 150 to 100, and the drawdown gates are also applied at the 0.75% risk ceiling). Under the
revised battery every configuration still fails the Sharpe, DSR and P(OOS loss) gates. The data
set is spent; see [`docs/governance/prior_trials.md`](../../docs/governance/prior_trials.md).

This is a preliminary run, not the planned one. The 2012–2026 vendor exports could not be reached
from the build environment, so it uses a shorter, third-party price history (see "Data"). Code is
v7 (commit `59f1cbd`), frozen; nothing was changed after results were seen. The plan in
[`PLAN.md`](PLAN.md) was written before any result existed.

## Results

Stitched walk-forward out-of-sample, February 2019 to July 2025 (6.45 years, 25 folds), EUR 5,000,
0.25% risk per trade. The ablation retained the same three modules in every run: multi-stage
exits, the deterministic RV20 regime filter and the VPIN filter.

| Configuration | OOS Sharpe | Profit factor | Expectancy (R) | Trades | Net P&L (EUR) | DSR | P(OOS loss) | Profitable folds | Gates failed | Holdout Sharpe (2024-01 on) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1:1, plain stop (primary)** | 0.28 | 1.14 | 0.080 | 137 | 118 | 0.64 | 0.68 | 52% | 6 | 0.05 |
| 1:1, stop-limit | 0.60 | 1.33 | 0.173 | 131 | 238 | 0.85 | 0.35 | 60% | 5 | 0.49 |
| Leveraged, plain stop | 0.09 | 1.04 | 0.025 | 137 | 36 | 0.47 | 0.86 | 52% | 7 | −0.22 |
| Leveraged, stop-limit | 0.44 | 1.23 | 0.131 | 132 | 173 | 0.75 | 0.50 | 56% | 5 | 0.33 |

Gates: Sharpe > 0.80, DSR > 0.95, P(OOS loss) < 15%, profit factor > 1.20, positive under 1.5×
costs, max drawdown < 10%, Monte Carlo P(DD ≥ 20%) < 2%, profitable folds ≥ 65%, trades ≥ 150.

Every run fails the Sharpe, DSR, P(OOS loss), profitable-folds and trade-count gates. The drawdown
and Monte Carlo gates pass everywhere (maximum drawdown 2.1–2.7%), which mostly reflects how small
the positions are: average exposure was 0.23× equity.

### How much is noise

- The primary run's mean trade is +0.08R with a standard deviation of 1.31R over 137 trades: a
  t-statistic of 0.72. It is not distinguishable from zero.
- The best run (1:1, stop-limit) has a Sharpe of 0.60 over 6.45 years. Its standard error is about
  0.4, so even that is roughly 1.5 standard errors from zero.
- Net profit over 6.45 years is EUR 36 to 238 on EUR 5,000, a compound return of 0.1% to 0.7% a
  year.

## Where the thesis broke down

Primary configuration, 137 OOS trades.

| Slice | Trades | Win rate | Expectancy (R) | Net (EUR) |
|---|---:|---:|---:|---:|
| Long | 117 | 45% | +0.119 | +143 |
| Short | 20 | 35% | −0.147 | −25 |
| 2019 | 19 | 47% | +0.066 | +12 |
| 2020 | 23 | 61% | +0.463 | +113 |
| 2021 | 26 | 38% | +0.042 | +10 |
| 2022 | 19 | 26% | −0.328 | −65 |
| 2023 | 25 | 44% | +0.133 | +44 |
| 2024 | 19 | 42% | −0.002 | −4 |
| 2025 (to July) | 6 | 50% | +0.152 | +9 |

- **The edge is one year.** 2020 supplies almost all of the profit; without it the other 114
  trades net about EUR 5.
- **Shorts lose.** There are few of them and they cost money, including in 2022.
- **The exits work as designed but the entries do not select.** Half of all trades (73 of 137) hit
  the initial stop at about −1R; 39% reach +1.5R and 25% reach +2R. That is close to what a
  breakout with no directional information would give.
- **The baseline has nothing.** Before any module, the baseline's OOS Sharpe on the decision
  period was −0.12 with a profit factor of 0.93. The modules lift it to about zero-plus.
- **The Hurst gate can never fire on this market.** Over 2,439 days the DFA exponent of hourly
  returns averaged 0.484 and never reached 0.55 (maximum 0.548). Both versions of the gate left
  zero trades. Hourly NASDAQ-100 returns are, if anything, slightly anti-persistent, which is the
  opposite of what a breakout system needs.
- **The ARMA forecast carries no direction.** Its sign was positive on 99.8% of days, so the
  Pesaran–Timmermann test is undefined and the filter stayed blocked. The largest |Z| was 0.47.
- **Optuna added nothing.** The only tuned parameter was the VPIN threshold, and the in-sample
  Sharpe spread across its values was 0.02.

### Order type and funding

- **Plain stop against stop-limit:** the plain stop costs about 0.09R a trade here (0.080R against
  0.173R at 1:1). That is several times the 0.01R the synthetic comparison suggested. It is one
  history and the two order types take different trades, so the size is uncertain, but the
  direction is the one to plan for: Capital.com has no native stop-limit, so the executable
  version is the worse one unless the limit leg is emulated.
- **Funding:** the leveraged preset paid EUR 74–81 in overnight funding, about EUR 0.59 a trade,
  against zero at 1:1. That takes roughly 0.05R a trade off expectancy and turns the plain-stop
  run negative under 1.5× costs.

## Data

| Input | Source | Notes |
|---|---|---|
| Prices | NQ futures 1-minute bars, Hugging Face `mdelcristo/NQ-F_1min_OHLCV_Parquet` (MIT licence) | Third-party upload with no stated provenance. 2015-01-02 to 2025-07-24. No bid/ask |
| Index closes | FRED `NASDAQ100` | Used to validate the tape and to fit the futures basis |
| Rates, FX | FRED `EFFR`, `SOFR`, `DEXUSEU` | `ndxswing ingest macro` |
| FOMC | federalreserve.gov meeting calendar and historical pages | 84 scheduled meetings in the sample window, cross-checked against the Fed's calendar feed with no differences; statement at 14:00 ET |
| CPI, employment | ALFRED release dates (releases 10 and 50) | 08:30 ET; includes a few revision dates |

**Price construction.** An index CFD is quoted off the future less its fair-value basis, so the
tape was converted to a cash-equivalent index: within each contract, a robust line was fitted to
the daily basis (futures at 16:00 ET minus the official close) and subtracted. That removes the
quarterly roll jumps and the carry embedded in futures prices. Script: `prep/prepare_prices.py`;
detected rolls and fitted segments are in `data_notes/`.

**Validation.** Against the official NASDAQ-100 close on 2,631 days: daily return correlation
0.9992; level error 0.056% standard deviation, 0.99% at worst (16 March 2020, a real
futures-to-cash dislocation). Timestamps were confirmed as UTC from the intraday volume profile.

**Ingestion report.** 2,653 sessions, seven bars per session, 68 holiday dates and the hours after
21 early closes dropped, two sessions without data, no session missing its opening bar, nine
incomplete bars, no price-scale breaks.

## What this run cannot tell you

- **It is not the planned data set.** The history starts in 2015, not 2012, so the OOS span is
  6.45 years instead of about 10.5. The trade-count gate (131–137 against 150) fails partly for
  that reason; the others do not.
- **The tape is unverified at the minute level.** It matches official closes very closely, but
  intraday highs, lows and opens were not checked against a second source.
- **No bid/ask history.** The spread gate could not be evaluated, and every fill uses the preset's
  constant 1.8-point spread. At an index level of 7,000 in 2019 that is 2.6 bp, probably wider
  than the real quote then, so early-year costs are somewhat overstated.
- **Futures-derived, not broker quotes.** The broker's own pricing, especially around the open and
  in fast markets, may differ.
- **Four runs were compared.** The primary was fixed in advance; the other three are diagnostics
  and should not be promoted to "the result" because one of them looks better.

## Reproducing

Scripts are in `prep/`. `prepare_prices.py <dir with NQ_1min_*.parquet> <out>` builds the
cash-equivalent bars, `prepare_events.py <dir with downloaded Fed and ALFRED files> <out>` builds
the event calendar, and the store was loaded with `ndxswing ingest macro`, `ingest intraday` and
`ingest events`. Each run was

```bash
ndxswing ablation --db market.duckdb --symbol NDX --instrument capital_com_eu_unleveraged_1to1 --entry-order stop --trials 60 --jobs 4 --holdout-start 2024-01-01 --out runs/1to1_stop
```

with `--instrument` and `--entry-order` varied. Each folder holds the tearsheet, the ablation
table, the OOS trade ledger and the OOS equity curve.
