"""Command line interface.

    ndxswing ingest bars     --db data/market.duckdb --file ndx_1h.parquet --symbol NDX --source "IBKR TRADES"
    ndxswing ingest series   --db ... --file eurusd.csv --name EURUSD
    ndxswing ingest fred     --db ... --series DEXUSEU --name EURUSD
    ndxswing ingest fred     --db ... --series DFF --name USD_RATE --scale 0.01
    ndxswing ingest events   --db ... --file tier1_events.csv
    ndxswing ingest ibkr     --db ... --symbol NDX --sec-type IND --exchange NASDAQ
    ndxswing ingest futures  --db ... --file nq_contracts.parquet --root NQ
    ndxswing ingest panama   --db ... --root NQ --symbol NQ_PANAMA
    ndxswing ingest synthetic --db ... --symbol SYN            (pipeline verification only)
    ndxswing summary         --db ...
    ndxswing backtest        --db ... --symbol NDX --modules multi_stage_exits,trend_flip_exit
    ndxswing walkforward     --db ... --symbol NDX --trials 150 --jobs 4 --out runs/wf
    ndxswing ablation        --db ... --symbol NDX --trials 100 --jobs 4 --holdout-start 2024-01-01 --out runs/abl
    ndxswing verify          --out runs/verify                  (synthetic null + planted-edge checks)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


from .analytics.montecarlo import MonteCarloConfig
from .analytics.stats import compute_metrics
from .config import MODULE_ORDER, BacktestConfig, CostModel, ModuleFlags, StrategyParams, ndx_fractional_cfd
from .data import ingest as ing
from .data.store import MarketDataStore
from .data.synthetic import SYNTHETIC_SOURCE, SyntheticSpec, generate_synthetic_market
from .features.pipeline import FeatureConfig, load_or_build_feature_set
from .research.ablation import AblationConfig, SequentialAblationEngine
from .research.walkforward import BacktestContext, PurgedWalkForwardRunner, WalkForwardConfig


def _flags(spec: str | None, regime_model: str = "hmm3") -> ModuleFlags:
    if not spec or spec == "baseline":
        return ModuleFlags.baseline()
    if spec == "full":
        return ModuleFlags.full(regime_model=regime_model)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in MODULE_ORDER]
    if bad:
        raise SystemExit(f"unknown modules {bad}; choose from {list(MODULE_ORDER)}")
    return ModuleFlags(**{n: True for n in names}, regime_model=regime_model)


def _context(a: argparse.Namespace) -> BacktestContext:
    with MarketDataStore(a.db, read_only=True) as store:
        bars = store.load_hourly(a.symbol, a.start, a.end)
        source = "; ".join(store.sources(a.symbol))
        fx = store.load_series(a.fx) if store.has_series(a.fx) else None
        rate = store.load_series(a.rate) if store.has_series(a.rate) else None
        events = store.load_events()
    fcfg = FeatureConfig(hurst_boot=a.hurst_boot)
    fs = load_or_build_feature_set(bars, fcfg, a.cache, source=source, verbose=True)
    costs = CostModel(spread_bps=a.spread_bps, slippage_bps=a.slippage_bps, commission_bps=a.commission_bps,
                      commission_min=a.commission_min, financing_markup=a.financing_markup)
    cfg = BacktestConfig(initial_equity=a.equity, instrument=ndx_fractional_cfd(a.lot_step), costs=costs)
    if fx is None:
        print(f"[warn] no '{a.fx}' series in the store: P&L is NOT converted to EUR "
              f"(treating the instrument as account-currency quoted)", file=sys.stderr)
        cfg = replace(cfg, instrument=replace(cfg.instrument, quote_ccy=cfg.account_ccy))
    return BacktestContext(fs, cfg, fx, rate, events if len(events) else None)


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default="data/market.duckdb")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--fx", default="EURUSD", help="daily series: USD per EUR")
    p.add_argument("--rate", default="USD_RATE", help="daily benchmark rate series (decimal, annual)")
    p.add_argument("--cache", default="data/feature_cache")
    p.add_argument("--hurst-boot", type=int, default=200)
    p.add_argument("--equity", type=float, default=5000.0)
    p.add_argument("--lot-step", type=float, default=0.01)
    p.add_argument("--spread-bps", type=float, default=1.5)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--commission-min", type=float, default=0.0)
    p.add_argument("--financing-markup", type=float, default=0.03)
    p.add_argument("--out", default="runs/latest")


def _wf_config(a: argparse.Namespace) -> WalkForwardConfig:
    return WalkForwardConfig(n_trials=a.trials, n_jobs=a.jobs, top_k=a.top_k)


# ------------------------------------------------------------------ commands


def cmd_ingest(a: argparse.Namespace) -> None:
    with MarketDataStore(a.db) as store:
        if a.what == "bars":
            n = ing.ingest_bars_file(store, a.file, a.symbol, a.source, ts_col=a.ts_col, tz=a.tz,
                                     bid_col=a.bid_col, ask_col=a.ask_col)
            print(json.dumps(ing.data_quality_report(store.load_hourly(a.symbol)), indent=2))
        elif a.what == "series":
            n = ing.ingest_series_file(store, a.file, a.name, a.source, a.date_col, a.value_col)
        elif a.what == "fred":
            n = ing.ingest_fred(store, a.series, a.name, a.start, a.scale)
        elif a.what == "events":
            n = ing.ingest_events_file(store, a.file, a.source, a.tz)
        elif a.what == "ibkr":
            bars = ing.download_ibkr_hourly(a.symbol, a.sec_type, a.exchange, a.currency, a.start, a.show,
                                            a.host, a.port, a.client_id)
            n = store.upsert_hourly(a.store_symbol or a.symbol, bars, f"IBKR {a.show} RTH 1h")
            print(json.dumps(ing.data_quality_report(bars), indent=2))
        elif a.what == "futures":
            n = store.upsert_futures(a.root, ing.read_table(a.file), a.source)
        elif a.what == "panama":
            from .data.futures import panama_adjust

            cont = panama_adjust(store.load_futures(a.root), a.roll_days)
            n = store.upsert_hourly(a.symbol, cont, f"PANAMA({a.root}, roll T-{a.roll_days})")
        elif a.what == "synthetic":
            m = generate_synthetic_market(SyntheticSpec(seed=a.seed, edge=a.edge))
            n = store.upsert_hourly(a.symbol, m["hourly"], SYNTHETIC_SOURCE)
            store.upsert_series(f"{a.symbol}_EURUSD", m["fx"], SYNTHETIC_SOURCE)
            store.upsert_series(f"{a.symbol}_RATE", m["rate"], SYNTHETIC_SOURCE)
            print(f"synthetic FX / rate stored as {a.symbol}_EURUSD / {a.symbol}_RATE "
                  f"(pass --fx / --rate to use them)")
        else:  # pragma: no cover
            raise SystemExit(f"unknown ingest target {a.what}")
        print(f"ingested {n} rows into {a.db}")


def cmd_summary(a: argparse.Namespace) -> None:
    with MarketDataStore(a.db, read_only=True) as store:
        print(store.summary().to_string(index=False))


def cmd_backtest(a: argparse.Namespace) -> None:
    ctx = _context(a)
    res = ctx.backtester().run(StrategyParams(), _flags(a.modules, a.regime_model))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(out / "equity.csv")
    res.trades.to_csv(out / "trades.csv", index=False)
    m = compute_metrics(res)
    print("NOTE: single full-sample run with default parameters - in-sample diagnostics, not validation.")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=2))
    print("signal funnel:", res.stats["funnel"])


def cmd_walkforward(a: argparse.Namespace) -> None:
    ctx = _context(a)
    runner = PurgedWalkForwardRunner(ctx, _flags(a.modules, a.regime_model), _wf_config(a))
    wf = runner.run(verbose=True)
    ts = runner.tearsheet(wf, n_configs=a.configs, mc_config=MonteCarloConfig(n_paths=a.mc_paths))
    paths = ts.save(a.out)
    wf.fold_table.to_csv(Path(a.out) / "folds.csv", index=False)
    print(paths["markdown"].read_text())


def cmd_ablation(a: argparse.Namespace) -> None:
    ctx = _context(a)
    eng = SequentialAblationEngine(ctx, _wf_config(a), AblationConfig(holdout_start=a.holdout_start,
                                                                     min_trades=a.min_trades))
    res = eng.run(verbose=True)
    res.save(a.out)
    runner = PurgedWalkForwardRunner(ctx, res.final_flags, _wf_config(a))
    ts = runner.tearsheet(res.final, n_configs=a.configs, n_trials_total=a.trials * res.n_variants,
                          mc_config=MonteCarloConfig(n_paths=a.mc_paths),
                          meta={"ablation variants evaluated": res.n_variants})
    paths = ts.save(a.out)
    print(res.to_markdown())
    print(paths["markdown"].read_text())


def cmd_verify(a: argparse.Namespace) -> None:
    from .research.verify import run_pipeline_verification

    ok = run_pipeline_verification(a.out, trials=a.trials, jobs=a.jobs, mc_paths=a.mc_paths, years=a.years)
    raise SystemExit(0 if ok else 1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ndxswing", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="load data into the DuckDB store")
    pi.add_argument("what", choices=["bars", "series", "fred", "events", "ibkr", "futures", "panama", "synthetic"])
    pi.add_argument("--db", default="data/market.duckdb")
    pi.add_argument("--file")
    pi.add_argument("--symbol")
    pi.add_argument("--store-symbol")
    pi.add_argument("--name")
    pi.add_argument("--source", default="local file")
    pi.add_argument("--ts-col", default="ts")
    pi.add_argument("--tz", default="UTC", help="time zone of naive timestamps")
    pi.add_argument("--bid-col")
    pi.add_argument("--ask-col")
    pi.add_argument("--date-col", default="date")
    pi.add_argument("--value-col", default="value")
    pi.add_argument("--series")
    pi.add_argument("--start", default="2010-01-01")
    pi.add_argument("--scale", type=float, default=1.0)
    pi.add_argument("--sec-type", default="IND")
    pi.add_argument("--exchange", default="NASDAQ")
    pi.add_argument("--currency", default="USD")
    pi.add_argument("--show", default="TRADES")
    pi.add_argument("--host", default="127.0.0.1")
    pi.add_argument("--port", type=int, default=7497)
    pi.add_argument("--client-id", type=int, default=77)
    pi.add_argument("--root")
    pi.add_argument("--roll-days", type=int, default=5)
    pi.add_argument("--seed", type=int, default=42)
    pi.add_argument("--edge", type=float, default=1.0)
    pi.set_defaults(fn=cmd_ingest)

    ps = sub.add_parser("summary", help="list what the store holds")
    ps.add_argument("--db", default="data/market.duckdb")
    ps.set_defaults(fn=cmd_summary)

    pb = sub.add_parser("backtest", help="one full-sample run with default parameters")
    _add_run_args(pb)
    pb.add_argument("--modules", default="baseline")
    pb.add_argument("--regime-model", default="hmm3")
    pb.set_defaults(fn=cmd_backtest)

    for name, fn in (("walkforward", cmd_walkforward), ("ablation", cmd_ablation)):
        p = sub.add_parser(name)
        _add_run_args(p)
        p.add_argument("--trials", type=int, default=100)
        p.add_argument("--top-k", type=int, default=8)
        p.add_argument("--jobs", type=int, default=1)
        p.add_argument("--configs", type=int, default=64, help="configurations for PBO / DSR")
        p.add_argument("--mc-paths", type=int, default=10_000)
        if name == "walkforward":
            p.add_argument("--modules", default="baseline")
            p.add_argument("--regime-model", default="hmm3")
        else:
            p.add_argument("--holdout-start", default=None)
            p.add_argument("--min-trades", type=int, default=200)
        p.set_defaults(fn=fn)

    pv = sub.add_parser("verify", help="synthetic pipeline verification (null and planted-edge data)")
    pv.add_argument("--out", default="runs/verify")
    pv.add_argument("--trials", type=int, default=40)
    pv.add_argument("--jobs", type=int, default=4)
    pv.add_argument("--mc-paths", type=int, default=10_000)
    pv.add_argument("--years", type=float, default=15.0)
    pv.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
