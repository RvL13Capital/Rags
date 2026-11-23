# ndxswing — NASDAQ-100 swing-system research stack

> **Status: NOT READY FOR LIVE DEPLOYMENT.**
> No backtest on real market data has been run. The workspace contained no historical OHLCV / bid-ask
> files and the build environment could not reach a market-data vendor, so everything in this
> repository has been verified on **synthetic data and hand-scripted bars only**. Those runs test the
> machinery; they are not evidence of an edge. Real-data walk-forward validation (2012–2026) and
> 8–12 weeks of paper trading are still outstanding.

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

111 tests, about 45 seconds. They cover hand-computed fills and P&L for every branch of the order
state machine, Monte Carlo / brute-force references for the statistics, and look-ahead checks that
recompute features and signals on truncated history.

## Running it on real data

1. Load regular-session 1-hour bars (timestamp = bar start). Quotes are optional; with bid/ask
   columns the relative spread is stored per bar and used for fills and the spread gate.

```bash
ndxswing ingest bars --db data/market.duckdb --file ndx_1h.parquet --symbol NDX --tz America/New_York --source "IBKR TRADES RTH" --bid-col bid --ask-col ask
```

2. Load EURUSD and the USD overnight rates for CFD financing. One call, no API key: it reads
   FRED's public CSV download (EFFR, SOFR, and DEXUSEU for EURUSD) and stores `USD_RATE` = EFFR,
   the only one of the two rates that covers a 2012 start. With `FRED_API_KEY` set the JSON API is
   used instead. This step was run live from the build environment (data to November 2026).

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
EU entity, OANDA TMS Brokers S.A.).

| Preset | Lot rule | Costs | Status of the numbers |
|---|---|---|---|
| `pepperstone_eu_ctrader` (default) | USD 1 per point per lot, minimum and step 0.01 (`--lot-step 0.1` for the coarser rule) | 1.2 pt spread, no index commission, overnight rate ± 2.5% (ACT/360) | Supplied schedule, **not confirmed**. Pepperstone states no commission on index CFDs and advertises spreads from about 1.0 pt. The only minimum-size figure I found, on a third-party page, says 0.1 lot: check the cTrader symbol specification |
| `capital_com_eu_rest` | USD 1 per point per contract, minimum and step 0.10 | 1.2 pt spread, no commission, overnight rate ± 2.5% | Supplied schedule, **contradicted on two points by Capital.com's own EU market page**, which lists a minimum quantity of 0.001 and a long overnight charge of 0.02226% of position value a day, about 8.0% a year or the overnight rate + 4.1%. To model the page: `--lot-step 0.001 --financing-markup 0.041` |
| `ibkr_cfd` | IBUST100, whole units only | 0.010% (min USD 1.00), benchmark ± 1.5%, 1.0 pt spread assumed | Commission, minimum, multiplier and financing spread checked against IBKR's schedule (Nov 2026). Sizes to zero on EUR 5,000 |
| `ucits_etf` | whole shares, long only, EUR-quoted and unhedged | 6 bp spread, 2 bp slippage, EUR 1 ticket, no financing | Assumptions; set `--etf-price-ratio` for the actual share class |

Every cost field can be overridden (`--spread-points`, `--spread-bps`, `--slippage-bps`,
`--commission-bps`, `--commission-min`, `--financing-markup`), and `--b-risk` sets the risk per
trade (default 0.0025). With a 0.10 lot, 0.25% risk is clipped to zero for a typical stop;
`--b-risk 0.0035` is the smallest budget that carries a 135-point stop, and in the synthetic run it
still left about a third of signals below the minimum lot. The CLI warns when the lot size, rather
than the strategy, is deciding what trades.

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
initial stop counts as a stop-out. Financing is charged per calendar night on the previous close
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
real ledgers exist.

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
| OOS net Sharpe | > 0.80 | locked |
| Deflated Sharpe Ratio | > 0.95 | zero-mean null (Bailey and López de Prado 2014) with N_eff = max(1, ρ̄ + (1 − ρ̄)·K), ρ̄ the average pairwise correlation of the K compared return series. One fixed configuration: N_eff = 1, SR₀ = 0 |
| CSCV P(OOS loss of the in-sample winner) | < 15% | over all K configurations; not applicable when K = 1 |
| CSCV rank PBO | < 0.25 | enforced only when the released configuration has tuned parameters and the in-sample Sharpe IQR across its parameter draws exceeds 0.25; otherwise reported as N/A |
| OOS net profit factor | > 1.20 | locked |
| OOS net P&L under 1.5× costs | > 0 | |
| OOS maximum drawdown | < 10% | locked; measured from the never re-based high-water mark |
| Monte Carlo P(DD ≥ 20%) | < 2% | most adverse of the three generators |
| Profitable walk-forward OOS folds | ≥ 65% | |
| Independent OOS trades | ≥ 150 | |

The DSR with the cross-trial mean in its benchmark and the probabilistic Sharpe ratio against zero
are printed next to the gate as information only.

Two readings of the calibration that are stricter than a literal one, both on purpose:

- **K counts module variants even when no parameter is tuned.** Choosing among fifteen ablation
  variants is selection. On the synthetic control this gives N_eff = 6.1 and a DSR of 0.968; with
  N_eff forced to 1 it would be 0.999.
- **P(OOS loss) is gated whenever K ≥ 2**, not only when parameters are tuned.

And one that is narrower: the rank statistic is computed over parameter draws of the released
configuration only. Module variants are a heterogeneous set in which a three-trade variant can be
the in-sample "best"; over that set the control scored 0.73 with an IQR of 0.26, a hair over the
0.25 cut-off, and would have failed on a technicality.

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

Three checks, about 10 minutes on 4 cores. On a stochastic-volatility martingale, with the optimiser
allowed to tune the full geometry, the release gates must fail. On data with planted trend
persistence (the positive control, at twice the base strength; at the base strength the fixed-prior
baseline does not see it) the ablation must run through every module and the OOS result must be
clearly better than on the null. The same baseline is then sized on each broker preset at today's index level. The
latest report is in `docs/pipeline_verification/`. It is labelled synthetic throughout and the
tearsheet refuses to issue a release verdict for synthetic data.

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
   cap sizes every order to zero. The execution adapter has to target a micro-lot CFD API
   (cTrader Open API or Capital.com REST) or a cash ETF broker, not the planned IBKR gateway.
3. **A 0.1 lot is too coarse for the EUR 12.50 burn-in budget.** With a 135-point stop and the
   1.25× gap buffer, 0.1 units risks about EUR 15.70. It fits EUR 25.00; a 0.01 lot fits both. The
   preset default is therefore 0.01.
4. **The ETF route is viable after all.** My earlier note that a Xetra listing overlaps the US
   session for only two hours is out of date: since 1 December 2025 Xetra's extended retail service
   runs 08:00–22:00 for all listed ETFs, for retail orders only and only through participating
   brokers, as Tradegate and gettex already did. Long-only, no financing, wider spreads in the
   evening, and stops must be run client-side.
5. **The Z hurdle and the Hurst gate starve trades.** μ/σ = 0.35 over three days is an annualised
   forecast Sharpe near 3, which daily ARMA means do not reach. DFA on 1,500 hourly returns has a
   sampling error near 0.045, so a lower bound above 0.50 needs a point estimate near 0.58. Both
   are off in the baseline and are evaluated last, as challengers. On the synthetic trend data the
   specification gate left 3 trades in about ten years and the point-estimate version 26.
6. **Costs per trade decide the vehicle.** In the synthetic runs a CFD trade cost about EUR 0.70
   (spread, slippage and financing, the last being more than half), around 0.06R at EUR 12.50 risk.
   The ETF pays no financing, but a EUR 1 ticket on each leg is EUR 2 a round trip, 0.16R on its
   own. At this account size the ETF only wins with a near-zero ticket fee.
7. **What the recalibrated gates do, measured on synthetic histories.** Fixed-prior baseline, 40
   independent histories each, complete battery: with no edge, 0 of 40 pass (which bounds the
   false-positive rate below about 7.5% at 95% confidence); with the planted edge (median OOS Sharpe
   0.93), 22 of 40 pass. So a real edge of that size has roughly even odds of being confirmed on
   ten OOS years at about 18 trades a year, and a weaker one will usually be rejected. The gates
   that reject it are the profitable-folds share and the Sharpe floor, not the overfitting tests.
   The fold and trade thresholds were set after seeing the control land at 69% and 189, so they
   are fitted to this trade frequency; the no-edge result is what shows they did not open the door.

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
[Pepperstone index CFDs](https://pepperstone.com/global/trading/instruments/index-cfds).

## Not in this deliverable

- Live execution gateway, bracket reconciliation, heartbeat and kill-switch (backlog item 5). Given finding 2
  it should target the chosen micro-lot CFD or ETF broker, not IBUST100.
- ML meta-labelling challenger (backlog item 4, optional and only after the econometric ablation).
- Any real-data result. `ingest ibkr` is written against the documented API but has not been run
  against a live gateway; `ingest fred` is tested against a mocked response only.
- EGARCH and Generalised Hyperbolic tails. The exact forecaster covers GARCH and GJR-GARCH; tail
  models cover Student-t / q-Gaussian, tempered Lévy, block bootstrap and POT-EVT.
