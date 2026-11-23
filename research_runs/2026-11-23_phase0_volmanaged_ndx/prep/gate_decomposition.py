"""What did the v8 RV20 filter do: the breakout trades it removed, tagged by filter state.

Runs the exits-only configuration (the step before the filter was added) on
the v8 OOS span with fixed priors and writes every trade with the filter's
verdict at its signal: allowed when the previous session's RV20 percentile
was at or below 0.80.  Also writes that percentile per session.

Usage: gate_decomposition.py <market.duckdb> <feature cache dir> <out dir>
"""
import argparse
import sys

import numpy as np
import pandas as pd

from ndxswing import cli
from ndxswing.config import ModuleFlags, StrategyParams

db, cache, out = sys.argv[1], sys.argv[2], sys.argv[3]
a = argparse.Namespace(
    db=db, symbol="NDX", start=None, end=None, fx="EURUSD", rate="USD_RATE", cache=cache, hurst_boot=200, equity=5000.0,
    instrument="capital_com_eu_unleveraged_1to1", lot_step=None, entry_order="stop", b_risk=0.0025,
    lot_reference_price="last", etf_price_ratio=0.0022, spread_points=None, spread_bps=None, slippage_bps=None,
    commission_bps=None, commission_min=None, financing_markup=None, fx_conversion_fee=None)
ctx = cli._context(a)
p = StrategyParams(b_risk=0.0025)
daily = ctx.features.daily
pct = daily["rv20_pct"]
allowed_next = (pct <= p.rv_filter_pct).shift(1)             # what the engine sees during the next session
pd.DataFrame({"rv20_pct": pct, "filter_allows_next_session": allowed_next}).to_csv(f"{out}/rv20_filter_by_session.csv")

res = ctx.backtester().run(p, ModuleFlags.baseline().with_(multi_stage_exits=True), start="2019-02-01", end="2025-07-24")
t = res.real_trades().copy()
sess = pd.DatetimeIndex(t["entry_session"])
# the order was placed at the bar before the fill; its session is the entry session or the one before.
# The filter is evaluated on the signal bar, which uses the previous session's percentile.
sig_ts = pd.DatetimeIndex(t["entry_ts"])
t["filter_allows"] = allowed_next.reindex(sess).to_numpy()
keep = ["entry_ts", "exit_ts", "direction", "net_pnl", "gross_pnl", "r_multiple", "exit_reason", "sessions_held",
        "entry_session", "filter_allows"]
t[keep].to_csv(f"{out}/breakout_trades_by_filter.csv", index=False)
g = t.groupby("filter_allows").agg(n=("net_pnl", "size"), net=("net_pnl", "sum"), mean_r=("r_multiple", "mean"),
                                   win=("net_pnl", lambda s: (s > 0).mean()))
print(g.round(3).to_string())
gated = ctx.backtester().run(p, ModuleFlags.baseline().with_(multi_stage_exits=True, regime_gate=True, regime_model="rv20"),
                             start="2019-02-01", end="2025-07-24").real_trades()
print(f"exits only: {len(t)} trades, net {t.net_pnl.sum():.2f}; with the RV20 filter: {len(gated)} trades, net {gated.net_pnl.sum():.2f}")
