# Bot A: single out-of-sample test on 2021–2025

Written and committed before any bar from 2021 onwards was read for this rule. The run script
`step2_oos.py` is committed with this plan and is executed once.

## The rule (frozen)

Symmetric intraday noise-area momentum, long and short, flat at 15:55 ET, exactly as implemented
in `src/ndxswing/research/intraday.py` and as first run in the 2015–2020 triage
(`research_runs/2026-11-24_intraday_triage_2015_2020/`). The library implementation reproduces the
triage ledger trade for trade (1,021 trades, largest difference 8.9e-16 R). No constant is changed:
14-session noise area, 30-minute decisions from 10:00 to 14:30, stop distance
max(|entry − session VWAP|, 1.2 × ATR30(14)) with a 0.20% floor, exit on a 30-minute close back
inside max(upper boundary, VWAP), at most two trades a session.

The session anchor of the primary rule is the volume-weighted VWAP. The volume-free TWAP variant is
run beside it and decides nothing.

## What is known from 2015–2020 (Step 1, nothing after 2020 read)

| 2015–2020, fixed EUR 5,000, no breakers | VWAP | TWAP |
|---|---:|---:|
| Pepperstone, EUR 25, costs in basis points: net R, t, Sharpe | +0.086, 2.74, 1.06 | +0.094, 3.09, 1.16 |
| Pepperstone, EUR 25, costs in index points: net R, t, Sharpe | +0.022, 0.70, 0.21 | +0.030, 0.99, 0.26 |
| Capital.com, EUR 25, costs in index points: net R, t, Sharpe | −0.018, −0.59, −0.24 | −0.013, −0.45, −0.18 |

TWAP and VWAP take the same entry on 1,048 of about 1,075 trades and their daily P&L correlates at
0.989, so the rule does not need a futures volume feed. In-sample the result exists only under
the basis-point cost schedule.

## Account settings for this run

EUR 5,000 at the first session of 2021, risk budget a fraction of current equity, circuit breakers
as in v8 (1% intraday loss halts entries, 2% five-session loss halves risk, −4% / −6% of the true
high-water mark scale risk to 0.75 / 0.25 and cap leverage at 1.0, −8% pauses for 21 sessions,
−10% deactivates), notional never above 1.5 × equity (the project's hard cap, which the triage did
not apply), no overnight positions and therefore no funding. Indicators warm up on December 2020.

| Row | Broker | Risk per trade |
|---|---|---:|
| Track 1 (burn-in) | Capital.com, 0.001 lot, 0.7% conversion on realised P&L | 0.25% |
| Track 2 | Pepperstone, 0.10 lot pinned to 25,288 | 0.50% |
| Track 2 | Capital.com, 0.001 lot | 0.50% |

| Cost schedule | Pepperstone | Capital.com |
|---|---|---|
| I: basis points of price | 0.40 bp spread + 0.20 bp slippage a leg | 0.71 bp + 0.20 bp |
| II: index points at the price of the day | 1.0 pt + 0.5 pt a leg | 1.8 pt + 0.5 pt |
| III: 1.5 × II | | |

## Gates (v8 battery)

OOS Sharpe > 0.80; DSR > 0.95; profit factor > 1.20; net P&L under 1.5× costs > 0; trades ≥ 100;
maximum drawdown < 10% and 10,000-path Monte Carlo P(DD ≥ 20%) < 2%, at the row's own risk, at
0.25% and at the 0.75% ceiling. The CSCV gates do not apply (one configuration, nothing selected in
this run).

**Prior trials in the DSR.** 11.71 (swing ablations) + 2 (Phase 0 families) + 8.73 (the nine
intraday rules of the two triage rounds: average pairwise correlation of their daily results 0.034)
= **22.44** effective trials, carried with sampling variance 1/T.

**What that implies, stated before the run.** On about 4.5 years the benchmark Sharpe is about 0.93,
so the DSR gate needs an observed Sharpe of about 1.70. With no prior trials it would need 0.78.
The in-sample Sharpe was 1.06 under the kinder cost schedule. A pass on the DSR gate is therefore
unlikely even if the rule works as well as it did in sample; the probabilistic Sharpe ratio against
zero (no deflation) is reported beside it so the two readings can be told apart.

## Verdict rule

A row passes if every gate passes with Schedule II as the cost of record and Schedule III as the
stress. **GO to paper trading only if all three rows pass for the VWAP rule.** Schedule I is
reported as the forward-looking cost case; a pass under Schedule I alone is a NO-GO and will be
described as such. Nothing is re-run with other settings after the result is seen. A defect found
in the code would be fixed, documented and the run repeated once.

## Breakdowns (reported, not decisive)

Calendar year 2021 to 2025; long against short; before and after publication of Zarattini, Aziz and
Barbon (January 2021 to April 2024 against May 2024 to July 2025); VWAP against TWAP; 2015–2020
under the same account settings.
