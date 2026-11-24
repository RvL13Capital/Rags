"""Step 2: the single out-of-sample run of Bot A on 2021-2025, as laid down in PLAN.md.

Usage: step2_oos.py <1m parquet> <daily csv> <macro csv> <out dir> [mc paths]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ndxswing.analytics.montecarlo import MonteCarloConfig, run_monte_carlo
from ndxswing.analytics.stats import compute_metrics
from ndxswing.analytics.tearsheet import TearsheetReporter
from ndxswing.research import intraday as it

SRC, DAILY, MACRO, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
MC = MonteCarloConfig(n_paths=int(sys.argv[5]) if len(sys.argv) > 5 else 10_000)
OOS_START, PUBLICATION, PRIOR = "2021-01-01", "2024-05-01", 22.44
(OUT / "tearsheets").mkdir(parents=True, exist_ok=True)
sess = pd.read_csv(DAILY, index_col=0, parse_dates=True, usecols=["date", "n_min", "close"])
fx = pd.read_csv(MACRO, index_col=0, parse_dates=True, usecols=["date", "eurusd"])["eurusd"]
grid = it.SessionGrid.from_minute_bars(it.read_minute_bars(SRC), sess, fx,
                                       source="NQ futures 1m less fitted basis (cash-equivalent NASDAQ-100)")
inp = {a: it.noise_area_inputs(grid, a) for a in ("vwap", "twap")}
oos_days = grid.dates[grid.dates >= OOS_START]
print(f"{len(grid)} full sessions in the archive; out-of-sample: {len(oos_days)} sessions {oos_days[0].date()} .. {oos_days[-1].date()}")
ROWS = [("Track 1: Capital.com, 0.25%", "capital", 0.0025), ("Track 2: Pepperstone, 0.50%", "pepperstone", 0.0050),
        ("Track 2: Capital.com, 0.50%", "capital", 0.0050)]
CEILING = 0.0075
pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 90)


def run(anchor, broker, sched, risk, start=OOS_START, end=None):
    costs = it.cost_schedules(broker)
    c = costs["I"].stressed(1.5) if sched == "I x1.5" else costs[sched]
    return it.run_noise_area(grid, inp[anchor], c, b_risk=risk, start=start, end=end)


def trade_stats(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"trades": 0}
    nr = t.r_multiple.to_numpy()
    w, l = nr[nr > 0], nr[nr <= 0]
    lg, sh = t[t.direction > 0], t[t.direction < 0]
    return {"trades": len(t), "win_rate": float((nr > 0).mean()), "payoff": float(w.mean() / abs(l.mean())) if len(w) and len(l) else np.nan,
            "gross_exp_r": float(t.gross_r.mean()), "net_exp_r": float(nr.mean()),
            "t": float(nr.mean() / nr.std(ddof=1) * np.sqrt(len(nr))) if len(nr) > 2 else np.nan,
            "long_n": len(lg), "long_net_r": float(lg.r_multiple.mean()) if len(lg) else np.nan,
            "short_n": len(sh), "short_net_r": float(sh.r_multiple.mean()) if len(sh) else np.nan,
            "net_eur": float(t.net_pnl.sum()), "mean_stop_pct": float(100 * t.stop_pct.mean()),
            "mean_leverage": float(t.leverage.mean()), "share_leverage_capped": float((t.binding == "max_leverage").mean())}


def sharpe(r: it.IntradayResult) -> float:
    x = r.as_backtest_result().daily_returns().to_numpy()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 2 and x.std() > 0 else float("nan")


summary, detail, verdict_rows = [], {}, {}
for anchor in ("vwap", "twap"):
    for label, broker, risk in ROWS:
        for base, stress in (("II", "III"), ("I", "I x1.5")):
            oos = run(anchor, broker, base, risk)
            stressed = run(anchor, broker, stress, risk)
            ceil = run(anchor, broker, base, CEILING)
            burn = oos if risk == 0.0025 else run(anchor, broker, base, 0.0025)
            ins = run(anchor, broker, base, risk, start=None, end="2020-12-31")
            ts = TearsheetReporter(
                {"in_sample": ins.as_backtest_result(), "oos": oos.as_backtest_result()}, oos_stress=stressed.as_backtest_result(),
                n_trials=1, prior_eff_trials=PRIOR, oos_ceiling=ceil.as_backtest_result(), ceiling_risk=CEILING, mc_config=MC,
                meta={"rule": "Bot A noise-area momentum, long and short, flat at 15:55 ET", "anchor": anchor.upper(),
                      "account": label, "cost schedule": f"{base} (stress: {stress})", "prior effective trials": PRIOR,
                      "in-sample column": "2015-2020, same account settings", "out-of-sample": f"{oos_days[0].date()} .. {oos_days[-1].date()}"},
                title=f"Bot A {anchor.upper()} - {label} - schedule {base}")
            rep = ts.build()
            name = f"{anchor}_{broker}_{int(risk * 1e4)}bp_sched{base}"
            (OUT / "tearsheets" / f"{name}.md").write_text(ts.to_markdown(rep))
            gates = {g["gate"]: (g["status"], g["value"]) for g in rep["gates"]}
            bm = compute_metrics(burn.as_backtest_result())
            bmc = run_monte_carlo(burn.as_backtest_result().daily_returns().to_numpy(), MC)
            burn_p20 = max(v["p_dd_20"] for v in bmc.values() if "error" not in v)
            gates["Max drawdown at 0.25% risk"] = ("PASS" if bm["max_drawdown"] < 0.10 else "FAIL", bm["max_drawdown"])
            gates["Monte Carlo P(DD >= 20%) at 0.25% risk"] = ("PASS" if burn_p20 < 0.02 else "FAIL", burn_p20)
            failed = [k for k, (s, _) in gates.items() if s not in ("PASS",) and not s.startswith("N/A")]
            o, d = rep["segments"]["oos"], rep["dsr"]
            row = {"anchor": anchor.upper(), "account": label, "schedule": base, **trade_stats(oos.trades),
                   "sharpe": o["sharpe"], "profit_factor": o["profit_factor"], "cagr": o["cagr"], "max_dd": o["max_drawdown"],
                   "mc_p_dd20": o["mc_p_dd_20"], "dsr": d["dsr"], "psr_zero": d["psr_zero"], "sr0_ann": d["sr0_per_period"] * np.sqrt(252),
                   "stress_net_eur": rep["oos_cost_stress"]["net_pnl"], "ceiling_max_dd": rep["risk_ceiling"]["max_drawdown"],
                   "ceiling_mc_p_dd20": rep["risk_ceiling"]["mc_p_dd_20"], "burn_in_max_dd": bm["max_drawdown"], "burn_in_mc_p_dd20": burn_p20,
                   "final_equity": o["final_equity"], "in_sample_sharpe": rep["segments"]["in_sample"]["sharpe"],
                   "in_sample_net_r": float(ins.trades.r_multiple.mean()), "skipped_min_lot": oos.stats["skipped_min_lot"],
                   "halted_entries": oos.stats["halted_entries"], "pauses": oos.stats["n_pauses"],
                   "deactivated": str(oos.stats["deactivated_at"]), "gates_failed": len(failed), "failed": "; ".join(failed)}
            summary.append(row)
            detail[name] = {"gates": {k: {"status": s, "value": v} for k, (s, v) in gates.items()}, "row": row,
                            "breaker_events": oos.stats.get("breaker_events", [])}
            if anchor == "vwap" and base == "II":
                verdict_rows[label] = failed
                oos.trades.to_csv(OUT / f"trades_{name}.csv", index=False)
                oos.equity.to_csv(OUT / f"equity_{name}.csv")
            if base == "II" or anchor == "vwap":
                # breakdowns
                t = oos.trades
                yr = [{"anchor": anchor.upper(), "account": label, "schedule": base, "slice": str(y), **trade_stats(g),
                       "sharpe": sharpe(oos.window(f"{y}-01-01", f"{y}-12-31"))} for y, g in t.groupby(t.entry_session.dt.year)]
                pre, post = t[t.entry_session < PUBLICATION], t[t.entry_session >= PUBLICATION]
                yr.append({"anchor": anchor.upper(), "account": label, "schedule": base, "slice": "before publication (2021-01 .. 2024-04)",
                           **trade_stats(pre), "sharpe": sharpe(oos.window(None, "2024-04-30"))})
                yr.append({"anchor": anchor.upper(), "account": label, "schedule": base, "slice": "after publication (2024-05 .. 2025-07)",
                           **trade_stats(post), "sharpe": sharpe(oos.window(PUBLICATION, None))})
                detail[name]["breakdown"] = yr
res = pd.DataFrame(summary)
res.to_csv(OUT / "oos_summary.csv", index=False)
bd = pd.DataFrame([r for v in detail.values() for r in v.get("breakdown", [])])
bd.to_csv(OUT / "oos_breakdowns.csv", index=False)
(OUT / "oos_detail.json").write_text(json.dumps(detail, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

cols = ["anchor", "account", "schedule", "trades", "win_rate", "payoff", "gross_exp_r", "net_exp_r", "t", "sharpe", "profit_factor",
        "net_eur", "max_dd", "mc_p_dd20", "dsr", "psr_zero", "stress_net_eur", "ceiling_max_dd", "ceiling_mc_p_dd20", "gates_failed"]
print(res[cols].round(3).to_string(index=False))
print(res[["anchor", "account", "schedule", "long_n", "long_net_r", "short_n", "short_net_r", "in_sample_sharpe", "in_sample_net_r", "sr0_ann",
           "skipped_min_lot", "halted_entries", "pauses", "deactivated", "mean_leverage", "share_leverage_capped", "final_equity"]].round(3).to_string(index=False))
print(res[["anchor", "account", "schedule", "failed"]].to_string(index=False))
b = bd[(bd.anchor == "VWAP")]
print(b[["account", "schedule", "slice", "trades", "win_rate", "gross_exp_r", "net_exp_r", "t", "long_net_r", "short_net_r", "net_eur", "sharpe"]].round(3).to_string(index=False))
go = all(len(v) == 0 for v in verdict_rows.values()) and len(verdict_rows) == 3
print("\nVERDICT (VWAP rule, Schedule II as cost of record, all three rows must pass):", "GO" if go else "NO-GO")
for k, v in verdict_rows.items():
    print(f"  {k}: {'passes every gate' if not v else 'fails ' + '; '.join(v)}")
(OUT / "verdict.json").write_text(json.dumps({"verdict": "GO" if go else "NO-GO", "rows": verdict_rows}, indent=1))
