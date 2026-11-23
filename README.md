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
| `PurgedWalkForwardRunner` (36/6/6 months, 3-month step, purge + embargo, Optuna with neighbourhood penalty) | `src/ndxswing/research/walkforward.py` |
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

96 tests, about 40 seconds. They cover hand-computed fills and P&L for every branch of the order
state machine, Monte Carlo / brute-force references for the statistics, and look-ahead checks that
recompute features and signals on truncated history.

## Running it on real data

1. Load regular-session 1-hour bars (timestamp = bar start). Quotes are optional; with bid/ask
   columns the relative spread is stored per bar and used for fills and the spread gate.

```bash
ndxswing ingest bars --db data/market.duckdb --file ndx_1h.parquet --symbol NDX --tz America/New_York --source "IBKR TRADES RTH" --bid-col bid --ask-col ask
```

2. Load EURUSD (USD per EUR) and a USD benchmark rate for CFD financing. With a FRED key
   (`FRED_API_KEY`) the fixings can be pulled directly; a CSV works as well (`ingest series`).

```bash
ndxswing ingest fred --db data/market.duckdb --series DEXUSEU --name EURUSD
```

```bash
ndxswing ingest fred --db data/market.duckdb --series DFF --name USD_RATE --scale 0.01
```

3. Optionally load a tier-1 event calendar (`ts,name,tier`; FOMC, CPI, NFP, top-3 NDX earnings).

```bash
ndxswing ingest events --db data/market.duckdb --file tier1_events.csv --tz America/New_York
```

4. Run the sequential ablation with a reserved holdout, which also writes the final tearsheet.

```bash
ndxswing ablation --db data/market.duckdb --symbol NDX --trials 150 --jobs 4 --holdout-start 2024-01-01 --out runs/ablation
```

Other commands: `ndxswing summary`, `ndxswing backtest`, `ndxswing walkforward`,
`ndxswing ingest ibkr|futures|panama|synthetic`, `ndxswing verify`. `ndxswing --help` lists them.

Broker assumptions live in `config.py` and on the command line (`--spread-bps`, `--slippage-bps`,
`--commission-bps`, `--commission-min`, `--financing-markup`, `--lot-step`). The defaults
(1.5 bp spread, 1 bp slippage per side, benchmark ± 3.0% financing, 0.01 lot step, 5% margin) are
placeholders and must be replaced by the actual broker schedule.

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

Each fold: 36 months train, 6 months validation, 6 months OOS, shifted 3 months, with 5 + 5 sessions
removed between windows. Parameters are chosen on validation only with

`J = 0.40·Sharpe + 0.30·Calmar − 0.20·MDD/0.10 − 0.10·Turnover/100 − λ·NeighbourhoodStd(Sharpe)`

and then frozen for the fold's OOS window. The stitched OOS curve is one continuous simulation with
a parameter schedule, so drawdown breakers and Kelly state carry across folds. In-Sample,
Validation and OOS are simulated and reported separately; Paper and Live columns stay `n/a` until
real ledgers exist.

The ablation starts from the baseline (daily EMA20/50, 20-hour Donchian stop-limit, 1.5 ATR stop,
5-day time exit, realised-vol targeted sizing) and adds one module at a time. A module is retained
only if stitched OOS Sharpe improves by at least 0.05, Calmar does not fall and the trade count
stays above a floor. Because these decisions look at OOS data, every variant is counted as a trial
in the Deflated Sharpe Ratio, and `--holdout-start` reserves a period no decision ever sees.

## Pipeline verification (synthetic)

```bash
ndxswing verify --out runs/verify --trials 40 --jobs 4
```

Two falsification checks, about 10 minutes on 4 cores. On a stochastic-volatility martingale the
release gates must fail; on data with planted trend persistence the ablation must run through every
module and the OOS result should be visibly better than on the null. The latest report is in
`docs/pipeline_verification/`. It is labelled synthetic throughout and the tearsheet refuses to
issue a release verdict for synthetic data.

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
| Hurst lower bound | `H − 1.645·sd(bootstrap)`; the percentile bound is available but biased towards 0.5 with 24-bar blocks |
| Rolling 5-day loss | five completed sessions plus the current one |
| Kelly `μ_LCB ≤ 0` and the −8% pause | trades continue off-book ("shadow") so the edge estimate can recover; equity is untouched |
| −8% / −10% breakers in a multi-year backtest | resume after 21 / 63 sessions or at the next refit with the drawdown reference re-based; reported max drawdown always uses the true peak. Live trading never resumes automatically |
| HMM emissions | Gaussian with ±4σ winsorised standardised inputs (the permitted alternative to Student-t) |
| ARMA(1,1) | two-step (ARMA mean, then GJR-GARCH-t on residuals); AR(1) is joint MLE and the default |

## Findings to review before spending time on real data

1. **The return arithmetic is binding.** Expected annual return ≈ trades/year × expectancy (R) ×
   risk per trade. At 0.25% risk, 30 trades a year and +0.2R that is about 1.5% a year. Under the
   locked risk limits there is no route to the aspirational multiples; they should not inform any
   sizing decision.
2. **The Z hurdle of 0.35 is very unlikely to fire.** An H-day forecast with μ/σ = 0.35 corresponds
   to an annualised forecast Sharpe near 3 for H = 3. Daily ARMA means do not get there; the filter
   is evaluated only if the Pesaran–Timmermann test is significant, and expect it to starve trades.
3. **The Hurst gate is close to a no-trade switch.** DFA on 1,500 hourly returns has a sampling
   error near 0.04–0.05, so `H_LCB > 0.50` needs a point estimate around 0.58. On the synthetic
   trend data it left 2 trades in ten years.
4. **Six-month validation windows hold 10–20 trades.** Selection on them is dominated by noise
   (validation Sharpe above 2 routinely became about 0 out of sample in the synthetic runs). DSR and
   PBO catch it, but consider scoring on train + validation.
5. **The UCITS ETF path does not fit an hourly US-session system.** A Xetra listing overlaps the US
   regular session for roughly two hours, so stops on US-afternoon bars cannot be worked there. The
   fractional index CFD is the only vehicle the simulator models as tradable through the session.
6. **Financing matters.** For multi-day CFD holds the overnight charge was of the same order as
   spread plus slippage in the synthetic runs. Use the real broker markup.
7. **PBO is noisy.** On pure noise the single-sample CSCV statistic centres on 0.5 with a standard
   deviation near 0.19, so read it together with DSR and the fold table.

## Not in this deliverable

- Live `IBKRExecutionGateway`, bracket reconciliation, heartbeat and kill-switch (backlog item 5).
- ML meta-labelling challenger (backlog item 4, optional and only after the econometric ablation).
- Any real-data result. `ingest ibkr` is written against the documented API but has not been run
  against a live gateway; `ingest fred` is tested against a mocked response only.
- EGARCH and Generalised Hyperbolic tails. The exact forecaster covers GARCH and GJR-GARCH; tail
  models cover Student-t / q-Gaussian, tempered Lévy, block bootstrap and POT-EVT.
