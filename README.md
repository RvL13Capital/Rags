# ndxswing — NASDAQ-100 swing-system research stack

> **Status: NOT READY FOR LIVE DEPLOYMENT.**
> A first real-data run (public NQ futures tape, 2015–2025, 6.45 out-of-sample years) fails the
> release gates in all four configurations; the primary one has an out-of-sample Sharpe of 0.28 and
> its long trades earned less than random long entries held equally long. See
> [`research_runs/2026-11-23_public_nq_2015_2025/`](research_runs/2026-11-23_public_nq_2015_2025/README.md).
> That data set is now spent: later research on it carries the trials recorded in
> [`docs/governance/prior_trials.md`](docs/governance/prior_trials.md).
> A follow-up on the same spent sample asked whether the one component that helped, the RV20 filter,
> is worth anything as volatility-managed passive exposure in a EUR 5,000 German retail account:
> under the pre-declared rule it is not, and the idea is archived
> ([`research_runs/2026-11-23_phase0_volmanaged_ndx/`](research_runs/2026-11-23_phase0_volmanaged_ndx/README.md)).
> The release battery was revised once after that run was audited (see "Release gates"); the run
> fails under the old and the new battery alike.
> Everything under `docs/pipeline_verification/` is synthetic and tests the machinery only.

This repository implements the first engineering deliverable of the handover:

| Deliverable | Where |
|---|---|
| `SignalEngine` (bar-by-bar, immutable `SignalProposal`) | `src/ndxswing/signal/engine.py` |
| `PortfolioRiskEngine`, circuit breakers, exact LCB Kelly, EVT | `src/ndxswing/risk/` |
| `EventDrivenBacktester` (order state machine, gap fills, partials, trail, time exit, financing) | `src/ndxswing/backtest/engine.py` |
| `TearsheetReporter` (15 metrics by segment, EVT ES, DSR, CSCV-PBO, 10,000-path Monte Carlo, release gates) | `src/ndxswing/analytics/` |
| `PurgedWalkForwardRunner` (36/12/6 months, 3-month step, purge + embargo, fixed priors or Optuna with neighbourhood penalty) | `src/ndxswing/research/walkforward.py` |
| `SequentialAblationEngine` (baseline, then one module at a time, parsimony benchmarks, PT gate, holdout) | `src/ndxswing/research/ablation.py` |
| DuckDB store, Panama roll, CSV/Parquet/FRED/IBKR ingestion CLI | `src/ndxswing/data/`, `src/ndxswing/cli.py` |
| Causal statistical layer (HMM, DFA Hurst, AR-GJR-GARCH-t, VPIN, EVT, Tsallis q) | `src/ndxswing/features/` |

The repository was empty at takeover, so the statistical prototypes listed as "designed" in the
handover (HMM filter, DFA, GARCH forecaster, VPIN, EVT, Kelly, risk engine, DSR, PBO, Monte Carlo)
were re-implemented here from the locked specification, with the audit fixes built in and tested.

## Install and test

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

160 tests, about a minute. They cover hand-computed fills and P&L for every branch of the order
state machine, Monte Carlo / brute-force references for the statistics, and look-ahead checks that
recompute features and signals on truncated history.

## Running it on real data

1. Load price bars. The simulator needs regular-session bars anchored at the 09:30 ET open: a
   30-minute opening bar, then hourly bars, seven per full session. Which command to use depends
   on what the source delivers.

   **Finer intraday bars (1 to 30 minutes), including round-the-clock CFD or futures feeds.**
   `ingest intraday` keeps NYSE cash sessions only (holidays and the hours after an early close are
   dropped), builds the 09:30-anchored bars and prints what it dropped. Bid and ask may come as
   two files; they are merged into mid prices with the quoted spread.

```bash
ndxswing ingest intraday --db data/market.duckdb --bid-file us100_bid_m1.parquet --ask-file us100_ask_m1.parquet --symbol NDX --source "vendor 1m bid/ask"
```

```bash
ndxswing ingest intraday --db data/market.duckdb --file ndx_1m.parquet --symbol NDX --tz UTC --source "vendor 1m"
```

   **Do not load clock-hour bars.** A 09:00–10:00 bar straddles the open, so the session filter
   drops it and every session starts at 10:00 with six bars; the opening price that the
   anti-chase rule and the overnight gap are measured against is lost. `ingest intraday` refuses
   such input. After any load, `bars_per_session_mode` in the printed report must be 7.

   **Bars that are already session-anchored** (for example IBKR regular-hours 1-hour bars, which
   start at 09:30) load directly:

```bash
ndxswing ingest bars --db data/market.duckdb --file ndx_1h.parquet --symbol NDX --tz America/New_York --source "IBKR TRADES RTH" --bid-col bid --ask-col ask
```

   **Futures contracts** go through `ingest futures` and `ingest panama`; add `--resample` to the
   latter when the contract bars are finer than an hour.

   Things the report tells you and you should read: holiday dates dropped, sessions with no data,
   sessions missing the opening bar, incomplete bars, and suspected price-scale breaks (a close
   more than three times, or less than a third of, the previous one). A scale break in early
   history can be repaired on load with `--price-scale-before DATE --price-scale FACTOR`. The
   session calendar comes from the `exchange_calendars` package when installed
   (`pip install -e ".[calendar]"`); otherwise from built-in rules that match it for 2012–2026.

2. Load EURUSD and the USD overnight rates for CFD funding. One call, no API key: it reads
   FRED's public CSV download (EFFR, SOFR, and DEXUSEU for EURUSD) and stores `USD_RATE` as EFFR
   before 2018-04-03 and SOFR from that date on, with EFFR on any day SOFR is missing. With
   `FRED_API_KEY` set the JSON API is used instead. This step was run live from the build
   environment (data to November 2026).

```bash
ndxswing ingest macro --db data/market.duckdb
```

3. Optionally load a tier-1 event calendar (`ts,name,tier`; FOMC, CPI, NFP, top-3 NDX earnings).

```bash
ndxswing ingest events --db data/market.duckdb --file tier1_events.csv --tz America/New_York
```

4. Run the fixed-prior baseline first (no tuning, no selection), then the sequential ablation with a
   reserved holdout, which also writes the final tearsheet.

```bash
ndxswing walkforward --db data/market.duckdb --symbol NDX --tune none --jobs 4 --out runs/baseline
```

```bash
ndxswing ablation --db data/market.duckdb --symbol NDX --trials 60 --jobs 4 --holdout-start 2024-01-01 --out runs/ablation
```

5. Run the long-only cash ETF track next to the CFD to see what overnight financing costs.

```bash
ndxswing walkforward --db data/market.duckdb --symbol NDX --instrument ucits_etf --tune none --jobs 4 --out runs/etf
```

Other commands: `ndxswing summary`, `ndxswing backtest`, `ndxswing ingest ibkr|futures|panama|synthetic`,
`ndxswing verify`. `ndxswing --help` lists them.

### Broker presets (`--instrument`)

A EUR 5,000 account needs a fractional index CFD and, for automation, a programmatic API. That
rules out IBKR (IBUST100 trades whole units) and OANDA (its v20 REST API documentation excludes the
EU entity, OANDA TMS Brokers S.A.). The CFD figures below are the EU schedules supplied for this
project; the last column says what I could cross-check on public pages.

| Preset | Lot rule | Costs | Cross-check |
|---|---|---|---|
| `capital_com_eu_leveraged` (default) | USD 1 per point per contract, minimum and step 0.001 | 1.8 pt spread, no commission, SOFR ± 4.0% on ACT/360 (0.01111% a day admin fee), 0.7% conversion mark-up on realised P&L and funding for a EUR account | Minimum 0.001, the 0.01111% daily fee with SOFR and the 0.7% retail conversion mark-up all match Capital.com's pages. The spread is a live quote and will move. A REST and WebSocket API is documented officially; that EU accounts can use it is stated by a third-party review, and I found no official EU statement |
| `capital_com_eu_unleveraged_1to1` | as above | 1.8 pt spread, no commission, no overnight funding, 0.7% conversion mark-up on realised P&L (EUR account), gross exposure up to 0.95× equity | Capital.com states that 1:1 positions on most markets carry no overnight funding |
| `pepperstone_eu_ctrader` | USD 1 per point per lot, minimum and step 0.10 | 1.0 pt spread on Razor (`--spread-points 1.2` for Standard), no index commission, overnight rate ± 2.5% on ACT/360 | No index commission is stated by Pepperstone; "from about 1.0 pt" is its advertised spread; 0.1 lot matches the one third-party figure found |
| `ibkr_cfd` | IBUST100, whole units only | 0.010% (min USD 1.00), benchmark ± 1.5%, 1.0 pt spread assumed | Commission, minimum, multiplier and financing spread checked against IBKR's schedule. Sizes to zero on EUR 5,000 |
| `ucits_etf` | whole shares, long only, EUR-quoted and unhedged | 6 bp spread, 2 bp slippage, EUR 1 ticket, no financing | Assumptions; set `--etf-price-ratio` for the actual share class |

Every cost field can be overridden (`--spread-points`, `--spread-bps`, `--slippage-bps`,
`--commission-bps`, `--commission-min`, `--financing-markup`, `--fx-conversion-fee`), and
`--b-risk` sets the risk per trade (default 0.0025). The CLI warns when the lot size, rather than
the strategy, is deciding what trades.

Three things follow from the corrected schedules:

- **Capital.com is the only CFD preset that can carry the EUR 12.50 burn-in budget.** With a 0.10
  lot, a typical 135-point stop risks about EUR 15.70, so Pepperstone sizes most signals to zero
  below roughly 0.35–0.50% risk. That is why the default preset changed.
- **The leveraged preset assumes a EUR account**, so the 0.7% conversion mark-up is on by default.
  Pass `--fx-conversion-fee 0` for a USD sub-account.
- **The 1:1 preset is sized as a cash-backed vehicle.** The rule that initial margin may use at
  most 50% of equity applies to leveraged margin products, which face the ESMA 50% margin
  close-out. A vehicle that posts its full notional (`margin_rate >= 1`) may use up to 95% of
  equity instead, so the 1:1 preset reaches about 0.95× exposure. The same rule now applies to
  the cash ETF preset, which was previously held to 0.5× as well. The 1:1 preset is modelled as a
  EUR account: no overnight funding, 0.7% conversion mark-up on realised P&L.

### Broker execution notes (Capital.com)

These matter for the execution adapter and for how far live fills can be expected to match the
simulator.

- **No native stop-limit.** `POST /api/v1/workingorders` creates a `LIMIT` or a `STOP` order
  (confirmed in the API reference). The strategy's entry is a stop-limit, and the simulator fills
  it as one: never above the limit (trigger + 0.15 ATR), and when a bar opens between the limit
  and the anti-chase bound (trigger + 0.50 ATR) it waits for a pullback to the limit. A plain
  `STOP` would fill at market in that band, up to 0.35 ATR worse than simulated. The adapter
  therefore has to emulate the limit: watch the price stream, and when the trigger trades send a
  `LIMIT` at the limit price, or check the open against the limit before any order is working.
  Until that emulation has been measured in paper trading, treat entry-fill parity as unproven.
- **A plain-stop entry can be simulated** with `--entry-order stop` (default `stop_limit`,
  unchanged). It drops the limit leg only: the anti-chase cancellation at the open and the
  two-bar expiry still apply, which assumes the adapter checks the opening price before a stop is
  live. On synthetic data, paired over 40 independent histories, the plain stop paid through the
  limit on about half its fills, by 0.18 ATR on average, took about 5% more trades, and cost
  about 0.01R a trade in expectancy (−0.007R ± 0.004 with no edge, −0.011R ± 0.005 with the
  planted edge). That is small next to roughly 0.08R of costs per trade, but it is a synthetic
  result: run both modes on the real bars before choosing.
- **Guaranteed and trailing stops are mutually exclusive** (as reported for this endpoint; the
  reference lists `guaranteedStop` and `trailingStop` as separate optional flags and does not
  state the restriction, so confirm it on the demo server). The system does not need the broker's
  trailing stop: the chandelier level is recomputed at each hourly close and sent as an amended
  stop. That leaves the guaranteed stop available, but its premium and its effect on gap-through
  losses are not modelled; the simulator assumes ordinary stops that fill at the open after a gap.
- **Sessions expire after ten minutes without activity**, so the heartbeat that drives the
  kill-switch also has to keep the session alive.

**Lot rules are normalised to one index level** (`--lot-reference-price`, default `last` = the final
close). One lot is then the same notional throughout history. Without this, a 2012 backtest at
index 2,300 trades lots eleven times finer, relative to a fixed EUR 5,000 account, than that account
can trade today, and whole-unit IBUST100 looks tradable when it is not.

## Architecture

```
data (DuckDB)  ->  features (causal, rolling refits)  ->  SignalEngine  --SignalProposal-->
PortfolioRiskEngine + CircuitBreakers  --RiskApprovedOrder-->  EventDrivenBacktester
-> analytics (metrics, DSR, PBO, Monte Carlo, tearsheet)  ->  research (walk-forward, ablation)
```

The signal engine never sizes and never places orders. Size is always floored to the lot step; a
size below the minimum lot is a rejection.

### Causality rules

- Hourly inputs at bar `t` use bars `<= t`; Donchian levels exclude bar `t`.
- Daily inputs (regime, Hurst, GARCH, EVT, EMA direction) are lagged one full session.
- Statistical models are re-estimated on rolling windows that end on the day they are stamped with
  and are applied by forward filtering only (Hamilton recursion; no Viterbi, no smoothing).
- `tests/test_features.py::test_feature_table_has_no_look_ahead` and
  `tests/test_signal_engine.py::test_signals_have_no_look_ahead` rebuild everything on truncated
  history and require identical values.

### Simulation conventions

Bar prices are treated as mid quotes. Buys pay mid + half spread, sells receive mid − half spread;
stop and market fills also pay slippage, limit fills do not. A bar that opens beyond a stop fills at
the open, not at the stop. When a bar touches both the stop and a target, the stop is assumed first.
Stop changes computed at a bar's close apply from the next bar. On the entry bar a touch of the
initial stop counts as a stop-out.

A stop-limit whose trigger is elected without a fill (the bar opened beyond the limit and never came
back) rests as a plain limit for the remaining bar of its window in the same session: it fills at
the next open if that is at or inside the limit, otherwise on a pullback to the limit. The triggered
state is dropped at the session break, where a working order is pulled and re-armed at the next open.
On a pullback fill the bar's high (low, for a short) may have come before the fill, so only the
close counts as favourable movement on that bar; the stop check still uses the whole bar. Both
conventions were checked against the 1-minute tape of the 2015–2025 run: hourly bars equal the
1-minute aggregates on all 268 entry bars and no target or trail was credited from a pre-fill
extreme. Financing is charged per calendar night on the previous close
notional. Sessions, time exits and financing are driven by timestamps, not a fixed bars-per-day.

### Walk-forward and ablation

Each fold: 36 months train, 12 months validation, 6 months OOS, shifted 3 months, with 5 + 5
sessions removed between windows. By default the entry and exit geometry is frozen at its canonical
values (20-hour Donchian, 1.5 ATR stop, 5-day hold) and only the thresholds of the enabled modules
are tuned, at most three (`--tune thresholds`). With nothing to tune, or with `--tune none`, Optuna
is not run and the fold is a plain fixed-prior OOS test. `--tune all` also searches the geometry.
Where something is tuned, it is chosen on validation only with

`J = 0.40·Sharpe + 0.30·Calmar − 0.20·MDD/0.10 − 0.10·Turnover/100 − λ·NeighbourhoodStd(Sharpe)`

and then frozen for the fold's OOS window. The stitched OOS curve is one continuous simulation with
a parameter schedule, so drawdown breakers and Kelly state carry across folds. In-Sample,
Validation and OOS are simulated and reported separately; Paper and Live columns stay `n/a` until
real ledgers exist. After an ablation the OOS column is not selection-free, because modules were
retained on it; the tearsheet says so and shows the reserved period in its own Holdout column.

The ablation starts from the baseline (daily EMA20/50, 20-hour Donchian stop-limit, 1.5 ATR stop,
5-day time exit, realised-vol targeted sizing) and adds one module at a time: exits, regime gate,
GARCH volatility, EVT cap, liquidity and event filters, score sizing, Kelly, and only then the two
challengers that are expected to starve trades, the Hurst gate (specification version, then a
point-estimate-only version) and the ARMA Z filter. A module is retained only if stitched OOS Sharpe
improves by at least 0.05, Calmar does not fall and the trade count stays above a floor.

Because these decisions look at OOS data, the OOS curve of every variant examined is one of the K
configurations compared, and `--holdout-start` reserves a period no decision ever sees.

### Release gates

| Gate | Rule | Notes |
|---|---|---|
| OOS net Sharpe | > 0.80 | |
| Deflated Sharpe Ratio | > 0.95 | zero-mean null (Bailey and López de Prado 2014) with N_eff = max(1, ρ̄ + (1 − ρ̄)·K), ρ̄ the average pairwise correlation of the K compared return series. One fixed configuration: N_eff = 1, SR₀ = 0. `--prior-eff-trials` adds trials already spent on the data |
| CSCV P(OOS loss of the in-sample winner) | < 15% | over all K configurations; not applicable when K = 1 |
| CSCV rank PBO | < 0.25 | enforced only when the released configuration has tuned parameters and the in-sample Sharpe IQR across its parameter draws exceeds 0.25; otherwise reported as N/A |
| OOS net profit factor | > 1.20 | |
| OOS net P&L under 1.5× costs | > 0 | |
| OOS maximum drawdown | < 10% | at the run's risk budget; measured from the never re-based high-water mark |
| Monte Carlo P(DD ≥ 20%) | < 2% | at the run's risk budget; most adverse of the three generators |
| OOS maximum drawdown at the 0.75% risk ceiling | < 10% | the final configuration replayed with the saved fold parameters; nothing re-estimated |
| Monte Carlo P(DD ≥ 20%) at the 0.75% risk ceiling | < 2% | same replay |
| Independent OOS trades | ≥ 100 | |

Printed for information, not gated: the DSR with the cross-trial mean in its benchmark, the
probabilistic Sharpe ratio against zero, maximum drawdown over annualised volatility, and the share
of profitable non-overlapping six-month slices of the stitched OOS curve.

**Revised once, after the first real-data run was audited.** The run had already failed and
fails under both batteries (Sharpe, DSR and P(OOS loss) in every configuration), so nothing here
rescues it. Three things changed, each because the old rule rejected real edges without keeping
false ones out:

- **Profitable folds ≥ 65% is no longer a gate.** It scored 25 overlapping six-month windows that
  count every trade twice. Simulated with the real ledger's trade distribution it passed 9% of
  zero-edge strategies and only 66% of strategies with a true Sharpe of 0.8. It is now reported on
  non-overlapping slices.
- **Trades ≥ 150 became ≥ 100.** The count measures sample length, not edge: at 21 trades a year it
  cannot be met on 6.45 years whatever the strategy does. On synthetic histories it rejected
  configurations with a Sharpe of up to 1.65 because a retained filter had cut their trade count.
  The DSR already scales with sample length.
- **The two survival gates are applied a second time at the 0.75% risk ceiling.** At the 0.25%
  burn-in budget average exposure is about 0.23× and even a losing system shows a 2.5% drawdown,
  so the gates said nothing about the tiers the system is meant to graduate to.

The thresholds of the first two were chosen after seeing synthetic histories and one real run, so
they are fitted; the false-pass measurement under "Pipeline verification" is the check that they
did not open the door. Trials already spent are carried forward with `--prior-eff-trials`
([`docs/governance/prior_trials.md`](docs/governance/prior_trials.md)).

Two readings of the DSR calibration are stricter than a literal one, both on purpose:

- **K counts module variants even when no parameter is tuned.** Choosing among fifteen ablation
  variants is selection.
- **P(OOS loss) is gated whenever K ≥ 2**, not only when parameters are tuned.

And one that is narrower: the rank statistic is computed over parameter draws of the released
configuration only. Module variants are a heterogeneous set in which a three-trade variant can be
the in-sample "best".

One known imprecision is left in place. The DSR benchmark multiplies the cross-trial Sharpe spread
(which already shrinks when variants are correlated) by z(N_eff) (which discounts correlation
again). Under worst-case selection among zero-edge variants the gate then passes 4–9% instead of
5%. The consistent alternative, z(K), would have failed 10 of 35 genuine-edge synthetic histories
that pass today, so the formula stays as it is.

### Drawdown breakers

Two references are kept apart. The all-time high-water mark is never re-based: the −4% and −6% risk
tiers and the hard deactivation at 90% of that peak are measured against it, and deactivation is
terminal inside a backtest. The −8% pause has its own reference, which is re-based when a pause
ends (21 sessions or the next refit) so that a flat system can trade again. A resumed system
therefore trades at quarter risk until it has actually recovered, and repeated pauses cannot step
the account down past −10%. Maximum drawdown on the tearsheet is always peak-to-trough from the
true high-water mark. Live trading never resumes on its own.

## Pipeline verification (synthetic)

```bash
ndxswing verify --out runs/verify --trials 40 --jobs 4
```

Three checks, under 10 minutes on 4 cores. On a stochastic-volatility martingale, with the optimiser
allowed to tune the full geometry, the release gates must fail. On data with planted trend
persistence (the positive control, at twice the base strength; at the base strength the fixed-prior
baseline does not see it) the ablation must run through every module and the OOS result must be
clearly better than on the null. The same baseline is then sized on each broker preset at today's index level. The
latest report is in `docs/pipeline_verification/`. It is labelled synthetic throughout and the
tearsheet refuses to issue a release verdict for synthetic data.

```bash
ndxswing ablation-oc --out runs/verify/ablation_oc --edge 0 --seeds 100:140 --trials 40 --jobs 4
```

runs independent histories through the whole research path (full feature set, sequential ablation
with tuning, release battery on what the ablation retains): `--edge 0` measures the false-pass rate
under selection, `--edge 2` the power. It takes eight to ten minutes per history on one core, keeps
one result file per history so an interrupted batch resumes, and `verify` summarises whatever it
finds in `<out>/ablation_oc`.

## Where the specification needed a decision

| Topic | What was implemented |
|---|---|
| `M_reg` when `P(trend) > 0.75` but `0.10 ≤ P(stress) < 0.20` | 0.75 (the table left this eligible state undefined, which would have meant 0) |
| Spread gate: 1.5× median (5.1) vs 2.0× block (4.D) | hard gate at 1.5× |
| "Expected gross reward-to-risk ≥ 1.5" | `(d·μ_H + σ_H) / (D_stop/P) ≥ 1.5`: a one-sigma favourable H-day move must cover 1.5R |
| Stop multiplier is tuned in [1.2, 2.2] but the skip rule is "2.25 ATR" | skip when `D_stop > 1.5 × stop_mult × ATR` (2.25 ATR at the default) |
| Position too small to split at +1.5R | no partial, stop still moves to breakeven + costs |
| Event blackout "next 2 hours" | window extended to the order's live period, so a pre-market CPI/NFP blocks an order placed at the prior close |
| Turnover term in `J` | divided by 100; Calmar capped at ±3 (six-month Calmar is otherwise explosive) |
| Hurst lower bound | `H − 1.645·sd(bootstrap)`; the percentile bound is available but biased towards 0.5 with 24-bar blocks. A point-estimate-only mode exists for the ablation |
| Rolling 5-day loss | five completed sessions plus the current one |
| Kelly `μ_LCB ≤ 0` and the −8% pause | trades continue off-book ("shadow") so the edge estimate can recover; equity is untouched |
| −8% / −10% breakers in a multi-year backtest | pause reference re-based on resume; tiers and the −10% kill anchored to the never re-based peak; deactivation terminal (see above) |
| Risk tiers after a resumed pause | measured against the true peak, so risk stays at 25% until the account is back within 6% of it |
| HMM emissions | Gaussian with ±4σ winsorised standardised inputs (the permitted alternative to Student-t) |
| ARMA(1,1) | two-step (ARMA mean, then GJR-GARCH-t on residuals); AR(1) is joint MLE and the default |

## Findings to review before spending time on real data

1. **The return arithmetic is binding.** Expected annual return ≈ trades/year × expectancy (R) ×
   risk per trade. At 0.25% risk, 30 trades a year and +0.2R that is about 1.5% a year. Under the
   locked risk limits there is no route to the aspirational multiples; they should not inform any
   sizing decision.
2. **Interactive Brokers cannot carry this system on EUR 5,000.** IBUST100 trades in whole units of
   1 × index. At 25,288 that is about USD 25,300, some 4.7 times the account, so the 1.5× leverage
   cap sizes every order to zero. The execution adapter has to target Capital.com's REST and
   WebSocket API (or cTrader for Pepperstone once the risk budget allows a 0.10 lot) or a cash
   ETF broker, not the planned IBKR gateway.
3. **A 0.1 lot is too coarse for the EUR 12.50 burn-in budget.** With a 135-point stop and the
   1.25× gap buffer, 0.1 units risks about EUR 15.70. It fits EUR 25.00; a 0.001 lot fits any
   budget. Pepperstone's 0.10 minimum therefore rules it out for the burn-in phase.
4. **The ETF route is viable after all.** My earlier note that a Xetra listing overlaps the US
   session for only two hours is out of date: since 1 December 2025 Xetra's extended retail service
   runs 08:00–22:00 for all listed ETFs, for retail orders only and only through participating
   brokers, as Tradegate and gettex already did. Long-only, no financing, wider spreads in the
   evening, and stops must be run client-side.
5. **The Z hurdle and the Hurst gate starve trades.** μ/σ = 0.35 over three days is an annualised
   forecast Sharpe near 3, which daily ARMA means do not reach. DFA on 1,500 hourly returns has a
   sampling error near 0.045, so a lower bound above 0.50 needs a point estimate near 0.58. Both
   are off in the baseline and are evaluated last, as challengers. On the synthetic trend data the
   specification gate left 3 trades in about ten years and the point-estimate version 18.
6. **Costs per trade decide the vehicle, and funding is the largest of them.** In the synthetic
   run at EUR 12.50 risk, a leveraged Capital.com trade in a EUR account cost about EUR 0.96:
   funding 59%, spread and slippage 28%, conversion mark-up 13%. That is roughly 0.08R a trade.
   The 1:1 version cost about EUR 0.39, because it pays no funding, and it gives up no size:
   average exposure was 0.23×, 0.32× and 0.44× of equity at 0.25%, 0.35% and 0.50% risk, the same
   as the leveraged preset. The ETF pays no funding either, but a EUR 1 ticket on each leg is
   EUR 2 a round trip, 0.16R on its own.
7. **What the release gates do, measured on synthetic histories.** Leveraged Capital.com preset,
   revised battery, complete research path. All counts are histories that pass every gate.

   | | No edge | Planted edge (median OOS Sharpe about 0.95) |
   |---|---:|---:|
   | Fixed priors, nothing selected | 0 of 40 | 27 of 40 |
   | Full ablation with tuning | 0 of 40 | 15 of 20 (fresh seeds; 13 of 20 for their fixed-prior baselines) |

   With no edge the ablation still lifts the OOS Sharpe by 0.28 on average, which is the selection
   bias the DSR is there for; the best no-edge history reached a Sharpe of 0.56 and a DSR of 0.88
   and failed three gates. Zero of 40 bounds the false-pass rate below about 7.5% at 95%
   confidence. With the planted edge the remaining rejections come from the Sharpe floor, the DSR
   and P(OOS loss): 15 of the 16 histories whose retained configuration had a Sharpe above 0.80
   passed. Twenty histories put the power somewhere between 51% and 91%. An edge weaker than
   this one will usually be rejected on ten OOS years at about 20 trades a year.
8. **Synthetic numbers before this version were flattered.** The synthetic bars used a coarse
   intrabar path (8 steps an hour). The simulator fills stops and triggers at the level a bar
   touches, which is right for a continuous path but too generous when the path jumps past the
   level: a no-edge series showed about 0.08R a trade of expectancy that did not exist. The
   generator now uses 256 steps, a test guards it, and every figure in this README and in
   `docs/pipeline_verification/` comes from the corrected bars. Real bars are built from ticks
   and are not affected; the no-edge data failed the gates before and after.

9. **What the audit of the first real-data run found.** The run failed because its entries have
   no timing value, not because of the validator. The audit replayed every entry order on the
   1-minute tape and found no fill, stop or target credited in the wrong order. It did find two
   latent simulator defects (a triggered stop-limit forgot its state on its second bar; a pullback
   fill credited the bar's earlier high), neither of which moved a single trade in that run, and
   three miscalibrated gates, described under "Release gates". The stop-limit's EUR 104 gross
   advantage over the plain stop decomposed into EUR 62 from ten trades only one variant happened
   to take (path luck), EUR 33 of price improvement on 56 fills (about 0.036R a trade across the
   book, the only systematic part) and EUR 7 from skipping runaway opens. The ARMA forecast was
   positive on every test point, so its sign test is 0/0; Z is drift over volatility and never
   fell below −0.35, so that gate could not have passed a short.

On the suggestion to score parameters on train plus validation: I dropped it in favour of the
longer validation window. One nuance for the record: in this implementation the statistical models
are re-estimated on rolling windows and applied forward only, so train-window signals are not
in-sample for the HMM or GARCH fits; the stronger argument for a longer validation window is simply
the trade count.

Sources for the broker and venue facts:
[IBKR index CFD commissions](https://www.interactivebrokers.ie/en/pricing/commissions-cfd-index.php),
[IBKR index CFDs, facts and Q&A](https://ibkb.interactivebrokers.com/article/1984),
[Deutsche Börse, Extended Xetra Retail Service](https://cashmarket.deutsche-boerse.com/cash-en/Data-Tech/Initiatives-Releases/extended-xetra-retail-service),
[OANDA v20 REST API, getting started](https://developer.oanda.com/rest-live-v20/introduction/),
[Capital.com US Tech 100, EU page](https://capital.com/en-eu/markets/indices/us-tech-100-index),
[Capital.com charges and fees](https://capital.com/charges-and-fees),
[Capital.com public API](https://open-api.capital.com/),
[Capital.com API request examples](https://capital.com/api-request-examples),
[Capital.com API from Germany (third-party review)](https://brokerchooser.com/broker-reviews/capitalcom-review/capitalcom-api-algorithmic-trading-from-germany),
[Pepperstone index CFDs](https://pepperstone.com/global/trading/instruments/index-cfds).

## Not in this deliverable

- Live execution gateway, bracket reconciliation, heartbeat and kill-switch (backlog item 5). Given finding 2
  it should target the chosen micro-lot CFD or ETF broker, not IBUST100.
- ML meta-labelling challenger (backlog item 4, optional and only after the econometric ablation).
- A result on the planned 2012–2026 vendor data with bid/ask history (the one real-data run so far uses a
  public futures tape). `ingest ibkr` is written against the documented API but has not been run
  against a live gateway.
- EGARCH and Generalised Hyperbolic tails. The exact forecaster covers GARCH and GJR-GARCH; tail
  models cover Student-t / q-Gaussian, tempered Lévy, block bootstrap and POT-EVT.
