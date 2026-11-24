"""Step 1, 2015-2020 ONLY (the reader drops every bar from 2021 on before it reaches memory).

1. The library implementation of Bot A must reproduce the triage ledger trade for trade.
2. Diagnostic: the same rule with a volume-free session TWAP in place of the session VWAP.
3. Effective number of independent trials among the nine intraday rules run on 2015-2020.

Usage: step1_in_sample.py <1m parquet> <daily csv> <macro csv> <triage dir> <out dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ndxswing.analytics.stats import average_pairwise_correlation, effective_trials
from ndxswing.research import intraday as it

SRC, DAILY, MACRO, TRIAGE, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
CUT = "2021-01-01"
minutes = it.read_minute_bars(SRC, before=CUT)
sess = pd.read_csv(DAILY, index_col=0, parse_dates=True, usecols=["date", "n_min", "close"])
sess = sess[sess.index < CUT]
fx = pd.read_csv(MACRO, index_col=0, parse_dates=True, usecols=["date", "eurusd"])["eurusd"]
fx = fx[fx.index < CUT]
grid = it.SessionGrid.from_minute_bars(minutes, sess, fx, source="NQ futures 1m less fitted basis (cash-equivalent NASDAQ-100)")
assert grid.dates.max() < pd.Timestamp(CUT)
print(f"{len(grid)} full sessions {grid.dates[0].date()} .. {grid.dates[-1].date()}")
inp = {a: it.noise_area_inputs(grid, a) for a in ("vwap", "twap")}
PEP, CAP = it.cost_schedules("pepperstone"), it.cost_schedules("capital")
PLAIN = dict(limits=None, max_leverage=None, compounding=False)       # the triage convention
res: dict = {}

# ---- 1. reproduction of the triage ledger
ref = pd.read_csv(TRIAGE / "results/trades_A_both_Pepperstone_25.csv", parse_dates=["date"])
run = it.run_noise_area(grid, inp["vwap"], PEP["I"], b_risk=0.005, **PLAIN)
t = run.trades
m = ref.merge(t, left_on=["date", "j", "k"], right_on=["entry_session", "entry_minute", "direction"], how="outer", indicator=True)
assert (m._merge == "both").all() and len(t) == len(ref) == 1021, m._merge.value_counts()
diff = float(np.abs(m.net_r - m.r_multiple).max())
assert diff < 1e-9 and (m.x == m.exit_minute).all()
print(f"library implementation reproduces the triage ledger: {len(t)} trades, max |net R difference| {diff:.1e}")
res["reproduction"] = {"trades": len(t), "max_abs_net_r_difference": diff, "mean_net_r": float(t.r_multiple.mean())}


def row(label: str, r: it.IntradayResult) -> dict:
    t = r.trades
    nr = t.r_multiple.to_numpy()
    daily = r.equity.diff().fillna(r.equity.iloc[0] - r.initial_equity).to_numpy() / r.initial_equity
    w, l = nr[nr > 0], nr[nr <= 0]
    return {"run": label, "trades": len(t), "skipped_min_lot": r.stats["skipped_min_lot"], "win_rate": float((nr > 0).mean()),
            "payoff": float(w.mean() / abs(l.mean())), "gross_exp_r": float(t.gross_r.mean()), "net_exp_r": float(nr.mean()),
            "net_sharpe": float(daily.mean() / daily.std() * np.sqrt(252)), "t": float(nr.mean() / nr.std(ddof=1) * np.sqrt(len(nr))),
            "long_n": int((t.direction > 0).sum()), "long_net_r": float(t[t.direction > 0].r_multiple.mean()),
            "short_n": int((t.direction < 0).sum()), "short_net_r": float(t[t.direction < 0].r_multiple.mean()),
            "mean_stop_pct": float(100 * t.stop_pct.mean()), "net_eur": float(t.net_pnl.sum())}


# ---- 2. VWAP (locked primary) against TWAP (no volume feed), both cost schedules
rows, keep = [], {}
for sched in ("I", "II"):
    for acct, costs, risk in (("Pepperstone, EUR 25.00", PEP, 0.005), ("Pepperstone, EUR 12.50", PEP, 0.0025), ("Capital.com, EUR 25.00", CAP, 0.005)):
        for a in ("vwap", "twap"):
            r = it.run_noise_area(grid, inp[a], costs[sched], b_risk=risk, **PLAIN)
            rows.append({"schedule": sched, "account": acct, "anchor": a.upper(), **row("", r)})
            keep[(sched, acct, a)] = r
tab = pd.DataFrame(rows).drop(columns=["run"])
tab.to_csv(OUT / "step1_vwap_vs_twap_2015_2020.csv", index=False)
pd.set_option("display.width", 260)
print(tab.round(3).to_string(index=False))
v, w = keep[("I", "Capital.com, EUR 25.00", "vwap")], keep[("I", "Capital.com, EUR 25.00", "twap")]
kv = set(zip(v.trades.entry_session, v.trades.entry_minute, v.trades.direction))
kw = set(zip(w.trades.entry_session, w.trades.entry_minute, w.trades.direction))
dv, dw = v.equity.diff().fillna(0.0), w.equity.diff().fillna(0.0)
both = v.trades.merge(w.trades, on=["entry_session", "entry_minute", "direction"], suffixes=("_v", "_w"))
res["vwap_vs_twap"] = {"trades_vwap": len(kv), "trades_twap": len(kw), "same_entry": len(kv & kw),
                       "daily_pnl_correlation": float(np.corrcoef(dv, dw)[0, 1]),
                       "same_entry_same_exit_minute": int((both.exit_minute_v == both.exit_minute_w).sum()),
                       "mean_stop_pct_vwap": float(100 * v.trades.stop_pct.mean()), "mean_stop_pct_twap": float(100 * w.trades.stop_pct.mean()),
                       "mean_abs_anchor_gap_bps": float(1e4 * np.mean(np.abs(inp["vwap"].anchor / inp["twap"].anchor - 1)))}
print(json.dumps(res["vwap_vs_twap"], indent=1))

# ---- 3. effective trials among the nine intraday rules of the two triage rounds (daily net R per session)
src = (TRIAGE / "as_run/triage.py").read_text().split("SIG, small = signals()")[0]
argv, sys.argv = sys.argv, ["triage.py", SRC, DAILY, MACRO, "unused"]
ns: dict = {}
exec(src, ns)                                      # rebuilds its own 2015-2020 arrays; nothing after 2020 is read there either
sys.argv = argv
sig, _ = ns["signals"]()
D = ns["D"]
series = {}
for name, s in sig.items():
    tt = ns["simulate"](s, 0.5, 0.5, 0.0)
    x = np.zeros(D)
    np.add.at(x, tt.d.to_numpy(), tt.net_r.to_numpy())
    series[name] = x
for name, f in (("A noise-area, long and short", "trades_A_both_Pepperstone_25.csv"), ("A noise-area, trend-aligned", "trades_A_trend_Pepperstone_25.csv"),
                ("B squeeze", "trades_B_both_Pepperstone_25.csv"), ("C VWAP limit fade", "trades_C_both_Capital.com_25.csv")):
    tt = pd.read_csv(TRIAGE / "results" / f)
    x = np.zeros(D)
    np.add.at(x, tt.d.to_numpy(), tt.net_r.to_numpy())
    series[name] = x
M = pd.DataFrame(series)
assert (M.std() > 0).all() and len(M.columns) == 9
c = M.corr()
rho = average_pairwise_correlation(M.to_numpy())
n_eff = effective_trials(9, rho)
c.round(3).to_csv(OUT / "step1_intraday_trial_correlations.csv")
print(c.round(2).to_string())
res["intraday_trials"] = {"k_raw": 9, "rho_bar": round(rho, 4), "n_eff": round(n_eff, 2), "mean_signed_correlation": round(float(c.to_numpy()[~np.eye(9, dtype=bool)].mean()), 4)}
res["prior_eff_trials_total"] = round(11.71 + 2.0 + n_eff, 2)
print(json.dumps({k: res[k] for k in ("intraday_trials", "prior_eff_trials_total")}, indent=1))
(OUT / "step1.json").write_text(json.dumps(res, indent=1))
