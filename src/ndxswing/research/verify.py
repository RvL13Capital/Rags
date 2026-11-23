"""End-to-end PIPELINE VERIFICATION on synthetic data.

This is a falsification test of the research machinery, not a backtest of the
NASDAQ-100:

1. NULL data (stochastic-volatility martingale, no exploitable structure):
   the full walk-forward + tearsheet must NOT clear the release gates.  If it
   did, the pipeline would be manufacturing an edge (look-ahead or overfit
   leakage).
2. PLANTED-EDGE data (persistent trend regimes): the sequential ablation must
   run through every module and the stitched OOS result should be visibly
   better than on the null data.

No number printed here is evidence about any real market.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..analytics.montecarlo import MonteCarloConfig
from ..config import BacktestConfig, ModuleFlags
from ..data.synthetic import SyntheticSpec, generate_synthetic_market
from ..features.pipeline import FeatureConfig, load_or_build_feature_set
from .ablation import AblationConfig, SequentialAblationEngine
from .walkforward import BacktestContext, PurgedWalkForwardRunner, WalkForwardConfig

BANNER = ("> **PIPELINE VERIFICATION - SYNTHETIC DATA.** These results exercise the code on simulated prices.\n"
          "> They are not empirical market performance and say nothing about the NASDAQ-100.\n")


def _context(edge: float, end: str, cache: Path, seed: int) -> BacktestContext:
    m = generate_synthetic_market(SyntheticSpec(edge=edge, end=end, seed=seed))
    fs = load_or_build_feature_set(m["hourly"], FeatureConfig(), cache, source=m["source"], verbose=True)
    return BacktestContext(fs, BacktestConfig(), m["fx"], m["rate"], m["events"])


def run_pipeline_verification(out_dir: str | Path, trials: int = 40, jobs: int = 4, mc_paths: int = 10_000,
                              years: float = 15.0, seed: int = 42) -> bool:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "feature_cache"
    end = f"{2012 + int(years) - 1}-10-30" if years < 15 else "2026-10-30"
    wf_cfg = WalkForwardConfig(n_trials=trials, n_jobs=jobs, top_k=4)
    mc = MonteCarloConfig(n_paths=mc_paths)
    t0 = time.time()

    # ---- 1. null: no edge exists, the gates must fail
    print("[verify] NULL data: building features", flush=True)
    ctx0 = _context(0.0, end, cache, seed)
    flags0 = ModuleFlags(multi_stage_exits=True, trend_flip_exit=True)
    run0 = PurgedWalkForwardRunner(ctx0, flags0, wf_cfg)
    print(f"[verify] NULL data: walk-forward over {len(run0.folds)} folds", flush=True)
    wf0 = run0.run()
    ts0 = run0.tearsheet(wf0, mc_config=mc, title="PIPELINE VERIFICATION - null (no-edge) synthetic data")
    rep0 = ts0.build()
    ts0.save(out / "null")
    gates0 = {g["gate"]: g["status"] for g in rep0["gates"]}
    null_rejected = any(s == "FAIL" for s in gates0.values())

    # ---- 2. planted edge: sequential ablation end to end
    print("[verify] PLANTED-EDGE data: building features", flush=True)
    ctx1 = _context(1.0, end, cache, seed)
    eng = SequentialAblationEngine(ctx1, wf_cfg, AblationConfig())
    print("[verify] PLANTED-EDGE data: sequential ablation", flush=True)
    abl = eng.run(verbose=True)
    abl.save(out / "planted")
    run1 = PurgedWalkForwardRunner(ctx1, abl.final_flags, wf_cfg)
    ts1 = run1.tearsheet(abl.final, n_trials_total=trials * abl.n_variants, mc_config=mc,
                         meta={"ablation variants evaluated": abl.n_variants},
                         title="PIPELINE VERIFICATION - planted-edge synthetic data")
    rep1 = ts1.build()
    ts1.save(out / "planted")

    s0, s1 = rep0["segments"]["oos"], rep1["segments"]["oos"]
    planted_seen = s1["sharpe"] > s0["sharpe"] and s1["net_pnl"] > 0
    ok = null_rejected and planted_seen

    def line(name: str, s: dict, rep: dict) -> str:
        d = rep["dsr"]["dsr"] if rep["dsr"] else float("nan")
        p = rep["pbo"]["pbo"] if rep["pbo"] else float("nan")
        return (f"| {name} | {s['sharpe']:.2f} | {s['profit_factor']:.2f} | {100 * s['max_drawdown']:.2f}% | "
                f"{s['n_trades']} | {s['net_pnl']:.2f} | {d:.3f} | {p:.3f} |")

    md = [
        "# Pipeline verification report", "", BANNER,
        f"Synthetic sample {2012}-{end[:4]}, {trials} Optuna trials per fold, {mc_paths} Monte Carlo paths, "
        f"runtime {time.time() - t0:.0f}s.", "",
        "| Data set | OOS Sharpe | OOS PF | OOS max DD | OOS trades | OOS net P&L (EUR) | DSR | PBO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        line("Null (martingale)", s0, rep0), line("Planted edge (after ablation)", s1, rep1), "",
        "## Checks", "",
        f"- Null data rejected by the release gates: **{'PASS' if null_rejected else 'FAIL'}** "
        f"(failed gates: {', '.join(k for k, v in gates0.items() if v == 'FAIL') or 'none'})",
        f"- Planted edge visible out of sample (Sharpe above null and positive net P&L): "
        f"**{'PASS' if planted_seen else 'FAIL'}**", "",
        "## Planted-edge ablation", "", abl.to_markdown(), "",
        "Full tearsheets: `null/tearsheet.md`, `planted/tearsheet.md`.", "",
    ]
    (out / "VERIFICATION.md").write_text("\n".join(md))
    print("\n".join(md))
    return ok
