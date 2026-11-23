"""Daily OOS return series of every configuration examined in the 2015-2025 run.

Re-runs the four sequential ablations with the settings of the committed run
and writes one column per variant (and, separately, per random parameter draw
of each final configuration).  The output feeds ``prior_trials_summary.py``,
which turns it into the K_prior / N_eff_prior carried into later research.

Usage: prior_trials.py <market.duckdb> <feature cache dir> <out dir> [trials] [jobs]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from ndxswing import cli
from ndxswing.config import StrategyParams
from ndxswing.research.ablation import AblationConfig, SequentialAblationEngine
from ndxswing.research.walkforward import PurgedWalkForwardRunner, WalkForwardConfig

db, cache, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
trials = int(sys.argv[4]) if len(sys.argv) > 4 else 60
jobs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
out.mkdir(parents=True, exist_ok=True)
RUNS = {"1to1_stop": ("capital_com_eu_unleveraged_1to1", "stop"),
        "1to1_stoplimit": ("capital_com_eu_unleveraged_1to1", "stop_limit"),
        "lev_stop": ("capital_com_eu_leveraged", "stop"),
        "lev_stoplimit": ("capital_com_eu_leveraged", "stop_limit")}

for run, (instrument, mode) in RUNS.items():
    if (out / f"{run}_variants.csv").exists():
        continue
    a = argparse.Namespace(
        db=db, symbol="NDX", start=None, end=None, fx="EURUSD", rate="USD_RATE", cache=cache, hurst_boot=200,
        equity=5000.0, instrument=instrument, lot_step=None, entry_order=mode, b_risk=0.0025,
        lot_reference_price="last", etf_price_ratio=0.0022, spread_points=None, spread_bps=None, slippage_bps=None,
        commission_bps=None, commission_min=None, financing_markup=None, fx_conversion_fee=None)
    ctx = cli._context(a)
    wf_cfg = WalkForwardConfig(n_trials=trials, n_jobs=jobs, top_k=8)
    base = StrategyParams(b_risk=0.0025)
    res = SequentialAblationEngine(ctx, wf_cfg, AblationConfig(holdout_start="2024-01-01", min_trades=200), base).run(verbose=True)
    rows = res.table[res.table["sharpe"].notna()].reset_index(drop=True)
    assert len(rows) == len(res.variant_oos), (len(rows), len(res.variant_oos))
    cols = {f"{run}|{r.step}|{r.module}|{r.variant}": v.daily_returns() for r, v in zip(rows.itertuples(), res.variant_oos)}
    pd.DataFrame(cols).to_csv(out / f"{run}_variants.csv")
    rows.assign(run=run).to_csv(out / f"{run}_table.csv", index=False)
    runner = PurgedWalkForwardRunner(ctx, res.final_flags, wf_cfg, base)
    draws = runner.config_trials(res.final, 64)
    pd.DataFrame({f"{run}|draw|{k}": d.daily_returns() for k, d in enumerate(draws)}).to_csv(out / f"{run}_draws.csv")
    res.final.oos.equity.to_csv(out / f"{run}_final_equity.csv")
    print(f"[prior] {run}: {len(cols)} variants, {len(draws)} draws, final net {res.final.oos.equity.iloc[-1] - 5000:.2f}", flush=True)
