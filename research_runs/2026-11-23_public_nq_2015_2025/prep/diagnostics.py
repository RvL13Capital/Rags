import sys, json, glob
import numpy as np, pandas as pd
run, cache = sys.argv[1], sys.argv[2]
pd.set_option("display.width", 220)
rows = []
for name in ("1to1_stop", "1to1_stoplimit", "lev_stop", "lev_stoplimit"):
    j = json.load(open(f"{run}/{name}/tearsheet.json"))
    o = j["segments"]["oos"]; d = j["dsr"]; p = j["pbo"] or {}
    fails = [g["gate"] for g in j["gates"] if g["status"] == "FAIL"]
    rows.append({"run": name, "sharpe": o["sharpe"], "pf": o["profit_factor"], "exp_R": o["expectancy_r"], "trades": o["n_trades"],
                 "net_pnl": o["net_pnl"], "txn": o["transaction_costs"], "fin": o["financing_costs"], "mdd": o["max_drawdown"],
                 "win": o["win_rate"], "cagr": o["cagr"], "dsr": d["dsr"], "n_eff": d["n_eff"], "psr0": d["psr_zero"],
                 "p_loss": p.get("prob_oos_loss"), "folds_ok": j["folds"]["profitable_share"], "n_folds": j["folds"]["n_folds"],
                 "stress_pnl": j["oos_cost_stress"]["net_pnl"], "years": o["years"], "gates_failed": len(fails)})
print(pd.DataFrame(rows).round(3).to_string(index=False))

t = pd.read_csv(f"{run}/1to1_stop/trades_oos.csv", parse_dates=["entry_ts", "exit_ts"])
t = t[~t["shadow"]]
print("\nPRIMARY (1:1, stop) OOS trades:", len(t), t.entry_ts.min().date(), "->", t.entry_ts.max().date())
g = lambda df: pd.Series({"n": len(df), "win%": round((df.net_pnl > 0).mean() * 100, 1), "exp_R": round(df.r_multiple.mean(), 3),
                          "net": round(df.net_pnl.sum(), 1), "avg_cost_R": round(((df.commission + df.spread_cost + df.slippage_cost + df.fx_fee + df.financing) / df.initial_risk).mean(), 3)})
print("by direction\n", t.groupby("direction").apply(g).to_string())
print("by exit reason\n", t.groupby("exit_reason").apply(g).to_string())
print("by year\n", t.groupby(t.entry_ts.dt.year).apply(g).to_string())
print("MFE/MAE (R): median mfe", round(t.mfe_r.median(), 2), "| share reaching +1.5R", round((t.mfe_r >= 1.5).mean(), 3), "| share reaching +2R", round((t.mfe_r >= 2).mean(), 3), "| median hold sessions", t.sessions_held.median(), "| avg leverage", round(t.leverage.mean(), 2))
print("R-multiple: mean", round(t.r_multiple.mean(), 3), "sd", round(t.r_multiple.std(), 2), "t-stat", round(t.r_multiple.mean() / t.r_multiple.std() * np.sqrt(len(t)), 2))

f = glob.glob(cache + "/features_*_daily.parquet")
d = pd.read_parquet(f[0])
print("\nHurst on real data:", d[["hurst", "hurst_lcb", "lo_rs_v"]].describe().loc[["count", "mean", "std", "min", "50%", "max"]].round(3).to_string())
print("days H>=0.55:", int((d.hurst >= 0.55).sum()), "| days H>=0.55 and LCB>0.5:", int(((d.hurst >= 0.55) & (d.hurst_lcb > 0.5)).sum()), "of", int(d.hurst.notna().sum()))
print("AR forecast mu_3: share positive", round((d.mu_3.dropna() > 0).mean(), 4), "| z_3 max abs", round(d.z_3.abs().max(), 3), "| garch_ok share", round(d.loc[d.sigma_1d.notna(), "garch_ok"].mean(), 3))
print("regime: p_trend>0.65 share", round((d.p_trend > 0.65).mean(), 3), "| rv20_pct<=0.8 share", round((d.rv20_pct <= 0.8).mean(), 3))
