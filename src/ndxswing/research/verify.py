"""End-to-end PIPELINE VERIFICATION on synthetic data.

This is a falsification test of the research machinery, not a backtest of the
NASDAQ-100:

1. NULL data (stochastic-volatility martingale, no exploitable structure):
   the full walk-forward + tearsheet must NOT clear the release gates.  If it
   did, the pipeline would be manufacturing an edge (look-ahead or overfit
   leakage).
2. PLANTED-EDGE data (persistent trend regimes): a positive control.  The
   sequential ablation must run through every module and the stitched OOS
   result must be clearly better than on the null data.  A sensitivity table
   shows how strong the planted edge has to be before the fixed-prior
   baseline sees it at all.
3. VEHICLE comparison on the planted data: the same fixed-prior baseline on a
   micro-lot CFD (0.01 and 0.1 lots), the IBKR IBUST100 CFD (whole units) and
   a long-only cash UCITS ETF, with lots normalised to today's index level.
   This checks the sizing mechanics per broker; it is not a cost forecast.

No number printed here is evidence about any real market.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ..analytics.montecarlo import MonteCarloConfig
from ..analytics.stats import compute_metrics
from ..config import (
    BacktestConfig, ModuleFlags, StrategyParams, ibkr_cfd_costs, ibkr_ibust100, micro_cfd_costs, ndx_fractional_cfd,
    ucits_etf_costs, ucits_ndx_etf,
)
from ..data.synthetic import SyntheticSpec, generate_synthetic_market
from ..features.pipeline import FeatureConfig, FeatureSet, load_or_build_feature_set
from .ablation import AblationConfig, SequentialAblationEngine
from .walkforward import BacktestContext, PurgedWalkForwardRunner, WalkForwardConfig

BANNER = ("> **PIPELINE VERIFICATION - SYNTHETIC DATA.** These results exercise the code on simulated prices.\n"
          "> They are not empirical market performance and say nothing about the NASDAQ-100.\n")

#: index level at which lot rules are applied (the handover's CME snapshot)
REFERENCE_INDEX_LEVEL = 25_288.25


def _relative(costs, level: float = REFERENCE_INDEX_LEVEL):  # noqa: ANN001, ANN202
    """Synthetic prices sit at an arbitrary level, so a spread quoted in index
    points is converted to the basis points it represents at the reference level."""
    if costs.spread_points is None:
        return costs
    return replace(costs, spread_points=None, spread_bps=costs.spread_points / level * 1e4)


def _market(edge: float, end: str, cache: Path, seed: int) -> tuple[dict, FeatureSet]:
    m = generate_synthetic_market(SyntheticSpec(edge=edge, end=end, seed=seed))
    fs = load_or_build_feature_set(m["hourly"], FeatureConfig(), cache, source=m["source"], verbose=True)
    return m, fs


def _context(m: dict, fs: FeatureSet) -> BacktestContext:
    cfg = BacktestConfig(instrument=ndx_fractional_cfd(0.01, REFERENCE_INDEX_LEVEL), costs=_relative(micro_cfd_costs()))
    return BacktestContext(fs, cfg, m["fx"], m["rate"], m["events"])


def vehicle_comparison(m: dict, fs: FeatureSet, b_risk: float = 0.0025) -> pd.DataFrame:
    """Fixed-prior baseline (no tuning, no selection) per vehicle over the full
    sample.  The quoted-spread column of the synthetic bars is dropped so each
    vehicle is charged its own preset spread."""
    hourly = fs.hourly.drop(columns=[c for c in ("spread_rel", "spread_med") if c in fs.hourly])
    bare = FeatureSet(hourly, fs.daily, fs.config, fs.source)
    ref = REFERENCE_INDEX_LEVEL
    vehicles = [
        ("Micro CFD, 0.01 lot", ndx_fractional_cfd(0.01, ref), micro_cfd_costs()),
        ("Micro CFD, 0.1 lot", ndx_fractional_cfd(0.1, ref), micro_cfd_costs()),
        ("IBKR IBUST100, 1 unit", ibkr_ibust100(ref), ibkr_cfd_costs()),
        ("UCITS ETF, long only", ucits_ndx_etf(lot_reference_price=ref), ucits_etf_costs()),
    ]
    rows = []
    params = StrategyParams(b_risk=b_risk)
    for name, inst, costs in vehicles:
        ctx = BacktestContext(bare, BacktestConfig(instrument=inst, costs=_relative(costs)), m["fx"], m["rate"])
        res = ctx.backtester().run(params, ModuleFlags.baseline())
        met = compute_metrics(res)
        rej = res.stats["rejections"]
        rows.append({
            "vehicle": name, "signals": res.stats["funnel"]["signals"], "orders": res.stats["orders"],
            "rejected_below_min_lot": rej.get("below_min_lot", 0), "rejected_short": rej.get("short_not_allowed", 0),
            "trades": met["n_trades"], "net_pnl": met["net_pnl"], "transaction_costs": met["transaction_costs"],
            "financing_costs": met["financing_costs"], "sharpe": met["sharpe"],
        })
    return pd.DataFrame(rows)


def detection_sensitivity(end: str, seed: int, strengths: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)) -> pd.DataFrame:
    """Fixed-prior baseline walk-forward (no tuning, no selection) at several
    planted-edge strengths: the statistical power of the simplest test."""
    cheap = FeatureConfig(with_regime=False, with_hurst=False, with_garch=False, with_evt=False, with_vpin=False)
    rows = []
    for k in strengths:
        m = generate_synthetic_market(SyntheticSpec(edge=k, end=end, seed=seed))
        from ..features.pipeline import build_feature_set

        fs = build_feature_set(m["hourly"], cheap, source=m["source"], n_jobs=1)
        wf = PurgedWalkForwardRunner(_context(m, fs), ModuleFlags.baseline(), WalkForwardConfig(tune="none")).run()
        met = compute_metrics(wf.oos)
        rows.append({"edge": k, "trend_drift_per_day": 0.0012 * k, "sharpe": met["sharpe"],
                     "profit_factor": met["profit_factor"], "expectancy_r": met["expectancy_r"],
                     "trades": met["n_trades"], "net_pnl": met["net_pnl"],
                     "profitable_folds": float((wf.fold_table["oos_net_pnl"] > 0).mean())})
    return pd.DataFrame(rows)


def run_pipeline_verification(out_dir: str | Path, trials: int = 40, jobs: int = 4, mc_paths: int = 10_000,
                              years: float = 15.0, seed: int = 42, planted_edge: float = 2.0) -> bool:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "feature_cache"
    end = f"{2012 + int(years) - 1}-10-30" if years < 15 else "2026-10-30"
    wf_cfg = WalkForwardConfig(n_trials=trials, n_jobs=jobs, top_k=4)
    mc = MonteCarloConfig(n_paths=mc_paths)
    t0 = time.time()

    # ---- 1. null: no edge exists, the gates must fail.  The search is given
    # the full geometry to tune here on purpose: the firewall has to hold when
    # the optimiser has room to mine noise.
    print("[verify] NULL data: building features", flush=True)
    m0, fs0 = _market(0.0, end, cache, seed)
    ctx0 = _context(m0, fs0)
    flags0 = ModuleFlags(multi_stage_exits=True, trend_flip_exit=True)
    run0 = PurgedWalkForwardRunner(ctx0, flags0, replace(wf_cfg, tune="all"))
    print(f"[verify] NULL data: walk-forward over {len(run0.folds)} folds", flush=True)
    wf0 = run0.run()
    ts0 = run0.tearsheet(wf0, mc_config=mc, title="PIPELINE VERIFICATION - null (no-edge) synthetic data")
    rep0 = ts0.build()
    ts0.save(out / "null")
    gates0 = {g["gate"]: g["status"] for g in rep0["gates"]}
    null_rejected = any(s == "FAIL" for s in gates0.values())

    # ---- 2. planted edge: sequential ablation end to end (default tuning policy)
    print("[verify] PLANTED-EDGE data: building features", flush=True)
    m1, fs1 = _market(planted_edge, end, cache, seed)
    ctx1 = _context(m1, fs1)
    eng = SequentialAblationEngine(ctx1, wf_cfg, AblationConfig())
    print("[verify] PLANTED-EDGE data: sequential ablation", flush=True)
    abl = eng.run(verbose=True)
    abl.save(out / "planted")
    run1 = PurgedWalkForwardRunner(ctx1, abl.final_flags, wf_cfg)
    ts1 = run1.tearsheet(abl.final, extra_trials=abl.variant_oos, mc_config=mc,
                         meta={"ablation variants evaluated": abl.n_variants},
                         title="PIPELINE VERIFICATION - planted-edge synthetic data")
    rep1 = ts1.build()
    ts1.save(out / "planted")

    # ---- 3. vehicles
    print("[verify] PLANTED-EDGE data: vehicle comparison", flush=True)
    veh = {b: vehicle_comparison(m1, fs1, b) for b in (0.0025, 0.0050)}
    for b, df in veh.items():
        df.to_csv(out / f"vehicles_risk_{int(b * 1e4)}bp.csv", index=False)

    print("[verify] detection sensitivity of the fixed-prior baseline", flush=True)
    sens = detection_sensitivity(end, seed)
    sens.to_csv(out / "detection_sensitivity.csv", index=False)

    s0, s1 = rep0["segments"]["oos"], rep1["segments"]["oos"]
    planted_seen = s1["sharpe"] > s0["sharpe"] + 0.5 and s1["net_pnl"] > 0 and s1["profit_factor"] > 1.2
    v25 = veh[0.0025].set_index("vehicle")
    sizing_ok = v25.loc["IBKR IBUST100, 1 unit", "trades"] == 0 and v25.loc["Micro CFD, 0.01 lot", "trades"] > 0
    ok = null_rejected and planted_seen and sizing_ok

    def line(name: str, s: dict, rep: dict) -> str:
        d = rep["dsr"] or {}
        p = rep["pbo"]["pbo"] if rep["pbo"] else float("nan")
        return (f"| {name} | {s['sharpe']:.2f} | {s['profit_factor']:.2f} | {100 * s['max_drawdown']:.2f}% | "
                f"{s['n_trades']} | {s['net_pnl']:.2f} | {d.get('dsr', float('nan')):.3f} | "
                f"{d.get('dsr_zero_mean_null', float('nan')):.3f} | {p:.3f} |")

    gates1 = {g["gate"]: g["status"] for g in rep1["gates"]}
    failed1 = [k for k, v in gates1.items() if v == "FAIL"]

    def vehicle_table(df: pd.DataFrame) -> list[str]:
        L = ["| Vehicle | Signals | Orders | Rejected: below minimum lot | Rejected: short | Trades | Net P&L (EUR) | "
             "Transaction costs | Financing |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in df.to_dict(orient="records"):
            L.append(f"| {r['vehicle']} | {r['signals']} | {r['orders']} | {r['rejected_below_min_lot']} | "
                     f"{r['rejected_short']} | {r['trades']} | {r['net_pnl']:.2f} | {r['transaction_costs']:.2f} | "
                     f"{r['financing_costs']:.2f} |")
        return L

    md = [
        "# Pipeline verification report", "", BANNER,
        f"Synthetic sample {2012}-{end[:4]}; walk-forward {wf_cfg.train_months}m train / {wf_cfg.val_months}m "
        f"validation / {wf_cfg.oos_months}m OOS; {trials} Optuna trials per fold where anything is tuned; "
        f"{mc_paths} Monte Carlo paths; lots normalised to index level {REFERENCE_INDEX_LEVEL:,.2f}; "
        f"runtime {time.time() - t0:.0f}s.", "",
        "| Data set | OOS Sharpe | OOS PF | OOS max DD | OOS trades | OOS net P&L (EUR) | DSR (spec) | "
        "DSR (zero-mean null) | PBO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        line("Null (martingale), full geometry tuned", s0, rep0),
        line(f"Planted edge x{planted_edge:g} (after ablation, thresholds only)", s1, rep1), "",
        "## Checks", "",
        f"- Null data rejected by the release gates: **{'PASS' if null_rejected else 'FAIL'}** "
        f"(failed gates: {', '.join(k for k, v in gates0.items() if v == 'FAIL') or 'none'})",
        f"- Planted edge clearly visible out of sample (Sharpe at least 0.5 above null, positive net P&L, "
        f"profit factor above 1.2): **{'PASS' if planted_seen else 'FAIL'}**",
        f"- Sizing per vehicle at EUR 5,000 (whole-unit IBUST100 never sizes, 0.01-lot CFD does): "
        f"**{'PASS' if sizing_ok else 'FAIL'}**", "",
        "## Power of the release gates", "",
        f"The positive control is a real edge by construction, yet it fails {len(failed1)} of {len(gates1)} gates: "
        f"{', '.join(failed1) or 'none'}. With about {s1['n_trades'] / s1['years']:.0f} trades a year the specified "
        "battery cannot confirm an edge of this size on roughly ten OOS years. Two of the failures are structural: "
        "the specified DSR adds the cross-trial mean to its benchmark, so sibling variants that all share the edge "
        "raise the bar to their own level; and CSCV-PBO measures whether the in-sample ranking of the compared "
        "variants persists, which is close to a coin flip when the variants are near-identical.", "",
        "## How strong must an edge be to be seen?", "",
        "Fixed-prior baseline, no tuning, stitched walk-forward OOS, same seed. Strength 1 is a drift of 0.12% a day "
        "inside trend regimes against roughly 0.9% daily volatility; the positive control above uses strength "
        f"{planted_edge:g}.", "",
        "| Planted strength | Trend drift / day | OOS Sharpe | OOS PF | Expectancy (R) | Trades | Net P&L (EUR) | "
        "Profitable folds |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {r['edge']:g} | {100 * r['trend_drift_per_day']:.2f}% | {r['sharpe']:.2f} | {r['profit_factor']:.2f} | "
          f"{r['expectancy_r']:.3f} | {r['trades']} | {r['net_pnl']:.2f} | {100 * r['profitable_folds']:.0f}% |"
          for r in sens.to_dict(orient="records")], "",
        "## Planted-edge ablation", "", abl.to_markdown(), "",
        "## Vehicles: fixed-prior baseline on the planted-edge data", "",
        "Full sample, default parameters, no tuning. Spreads are each vehicle's preset, expressed at the reference "
        "index level. The ETF track takes long signals only and carries EURUSD on its notional.", "",
        "Risk per trade 0.25% (EUR 12.50):", "", *vehicle_table(veh[0.0025]), "",
        "Risk per trade 0.50% (EUR 25.00):", "", *vehicle_table(veh[0.0050]), "",
        "Full tearsheets: `null_tearsheet.md`, `planted_tearsheet.md`.", "",
    ]
    (out / "VERIFICATION.md").write_text("\n".join(md))
    print("\n".join(md))
    return ok
