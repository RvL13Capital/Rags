# Bot A out of sample, 2021–2025

**Verdict: NO-GO under the pre-committed rule.** None of the three account rows passes every
v8 gate with index-point costs. The rule was run once, as frozen; the plan
([`PLAN.md`](PLAN.md)) and the run script were committed before any bar from 2021 onwards was read.

That verdict needs its context, because this is the first rule in the project whose edge showed up
again on data it had never seen:

- **The gross edge replicated.** +0.084 to +0.089R a trade out of sample against +0.100 to +0.105R
  in sample (gross t = 2.9 on 820 to 865 trades).
- **Net of costs it is positive in every row and under every cost schedule,** including the 1.5×
  stress.
- **Every row fails the DSR gate, as the plan said it would.** With 22.44 prior effective trials
  the gate needs a Sharpe of about 1.70 on 4.5 years; the rows delivered 0.69 to 1.05. Against
  zero, with no prior trials, the same series score 0.93 to 0.99.
- **The Capital.com rows also fail on cost:** Sharpe 0.75 and 0.69 against 0.80, profit factor
  1.15 and 1.13 against 1.20.

## Results (VWAP rule, 1,133 sessions, 2021-01-04 to 2025-07-24)

Schedule II (index points at the price of the day) is the cost of record.

| | Track 1: Capital.com, 0.25% | Track 2: Pepperstone, 0.50% | Track 2: Capital.com, 0.50% |
|---|---:|---:|---:|
| Trades | 865 | 820 | 865 |
| Win rate / payoff | 39.0% / 1.81 | 38.9% / 1.92 | 39.0% / 1.81 |
| Gross / net expectancy (R) | +0.084 / +0.046 | +0.089 / +0.065 | +0.084 / +0.046 |
| t of the mean trade | 1.60 | 2.15 | 1.60 |
| Net P&L on EUR 5,000 | +EUR 512 | +EUR 1,173 | +EUR 911 |
| **Sharpe (> 0.80)** | **0.75 fail** | 1.05 | **0.69 fail** |
| **DSR, 22.44 prior trials (> 0.95)** | **0.35 fail** | **0.60 fail** | **0.30 fail** |
| Probabilistic Sharpe against zero (information) | 0.949 | 0.990 | 0.931 |
| **Profit factor (> 1.20)** | **1.15 fail** | 1.23 | **1.13 fail** |
| Net P&L under 1.5× costs (> 0) | +EUR 321 | +EUR 975 | +EUR 578 |
| Trades (≥ 100) | 865 | 820 | 865 |
| Max drawdown (< 10%) | 2.8% | 4.0% | 5.1% |
| Monte Carlo P(DD ≥ 20%) (< 2%) | 0.0% | **3.0% fail** | 0.0% |
| Max drawdown at 0.75% risk (< 10%) | 6.8% | 4.9% | 6.8% |
| Monte Carlo P(DD ≥ 20%) at 0.75% risk (< 2%) | **2.5% fail** | 0.6% | **2.5% fail** |
| Max drawdown and Monte Carlo at 0.25% risk | pass | pass | pass |
| Gates failed | 4 | 2 | 4 |

Under Schedule I (today's quote in basis points) the picture is better but still not a pass:
Sharpe 0.96, 1.15 and 0.90; the DSR gate fails in all three (0.53, 0.69, 0.47); the profit factor
fails on both Capital.com rows (1.20 and 1.18).

No circuit breaker fired: no intraday halt, no pause, no deactivation. The 1.5× leverage cap bound
on 7 to 8% of trades at 0.50% risk and never at 0.25%. At Pepperstone 148 signals were skipped
because a 0.10 lot was too large for the budget.

### About the Monte Carlo failures

They come from one of the three generators. The symmetric Student-t fit (2.3 degrees of freedom
on the Pepperstone row) mirrors this rule's large winning days onto the loss side and discards its
positive drift; its median path ends below the starting equity for a series that gained 23%. The
block bootstrap and the tempered Lévy generator both give 0.0%. The gate is defined on the most
adverse generator, so it counts as a failure here, but it is measuring that generator's fit to a
positively skewed series more than the rule's drawdown risk.

## Breakdowns

**By year (Schedule II, net R a trade and Sharpe)**

| Year | Trades (Capital.com) | Capital.com 0.25%: net R, Sharpe | Pepperstone 0.50%: net R, Sharpe |
|---|---:|---|---|
| 2021 | 186 | +0.073, 1.18 | +0.093, 1.27 |
| 2022 | 214 | +0.036, 0.60 | +0.057, 1.07 |
| 2023 | 186 | +0.033, 0.52 | +0.052, 0.87 |
| 2024 | 181 | +0.079, 1.18 | +0.090, 1.38 |
| 2025 (to July) | 98 | −0.016, −0.29 | 0.000, 0.15 |

Positive in the four full years; flat in the first seven months of 2025.

**Long against short.** Long trades carry the result: +0.102R (Capital.com) and +0.121R
(Pepperstone) on about 430 trades, against −0.010R and +0.006R for the shorts. In sample the two
sides were equal (+0.087R and +0.084R). Out of sample this is a long-only edge in a rising market.

**Before and after publication of the paper.**

| | Trades (Capital.com) | Capital.com 0.25%: net R, t, Sharpe | Pepperstone 0.50%: net R, t, Sharpe |
|---|---:|---|---|
| January 2021 to April 2024 | 649 | +0.049, 1.47, 0.80 | +0.069, 1.98, 1.08 |
| May 2024 to July 2025 | 216 | +0.038, 0.66, 0.61 | +0.052, 0.88, 0.97 |

Somewhat weaker afterwards, and all of the weakness is in 2025. Fifteen months cannot separate
crowding from an ordinary flat stretch.

**VWAP against TWAP.** The volume-free TWAP variant does slightly better in every row (net
+0.051R and Sharpe 0.84 on Track 1; +0.069R and 1.11 on Pepperstone), as it did in sample. A CME
volume feed is not needed to run the rule. TWAP still fails the DSR gate everywhere.

**2015–2020 under the same account settings.** With index-point costs the in-sample Sharpe is
negative (−0.47, −0.29, −0.60): at index levels of 4,000 to 12,000 a 1.5-point leg is two to five
times what it is today. With basis-point costs it is 0.90 to 1.02. The out-of-sample years sit at
higher index levels, which is why the rule clears its costs there.

## What would have to be true for a GO

- **On the DSR gate:** nothing a 4.5-year sample can supply. Carrying 22.44 prior trials asks for
  a Sharpe near 1.70. The trials were spent on other samples and mostly on unrelated rules; whether
  they should deflate an untouched hold-out is a governance choice, and it was made before the run.
- **On cost:** the Pepperstone schedule. On that row only the DSR gate and the Monte Carlo artefact
  stand between the result and a pass.

## Files

- `step1_in_sample.py`, `results/step1*`: reproduction of the triage ledger, VWAP against TWAP and
  trial correlations, 2015–2020 only.
- `step2_oos.py`, `results/oos_summary.csv`, `results/oos_breakdowns.csv`, `results/oos_detail.json`,
  `results/verdict.json`: the run.
- `results/tearsheets/`: one v8 tearsheet per anchor, account row and cost schedule.
- `results/trades_*.csv`, `results/equity_*.csv`: ledgers and equity of the three rows of record.

```bash
python step2_oos.py ndx_cash_equiv_1m.parquet ../2026-11-23_phase0_volmanaged_ndx/data/ndx_daily_variance.csv ../2026-11-23_phase0_volmanaged_ndx/data/macro_daily.csv results
```

The 1-minute archive is not in the repository; it is rebuilt from the public NQ tape with
`research_runs/2026-11-23_public_nq_2015_2025/prep/prepare_prices.py`.

**The 2021–2025 intraday sample is now spent for this rule.** A second look with different
settings would not be out of sample.
