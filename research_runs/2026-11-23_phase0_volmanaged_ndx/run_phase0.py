"""Phase 0: volatility-managed NASDAQ-100 on the spent 2015-2025 sample.

Reads the committed daily tables in ``data/`` and writes ``results/``.
Everything here follows PLAN.md, which was committed before this was run.

Usage: run_phase0.py <this directory> [n_boot]
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ndxswing.analytics import relative as rel
from ndxswing.research import volmanaged as vm
from ndxswing.research.volmanaged import CashSweep, GermanTax, MarketDays, RetailCosts

HERE = Path(sys.argv[1])
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
d = pd.read_csv(HERE / "data/ndx_daily_variance.csv", index_col=0, parse_dates=True)
mac = pd.read_csv(HERE / "data/macro_daily.csv", index_col=0, parse_dates=True)
filt = pd.read_csv(HERE / "data/rv20_filter_by_session.csv", index_col=0, parse_dates=True)
trades = pd.read_csv(HERE / "data/breakout_trades_by_filter.csv", parse_dates=["entry_session"])
R: dict = {}
ANN = 252.0
pd.set_option("display.width", 250)


def show(title: str, df: pd.DataFrame, nd: int = 3) -> None:
    print(f"\n## {title}\n{df.round(nd).to_string()}")


# ------------------------------------------------------------------ 1. mechanism
gap2 = d["r_gap"].fillna(0.0) ** 2
tv = d["rv5"] + gap2
R["variance"] = {
    "sessions": int(len(d)), "identity_max_abs_residual": float((d.sum_r1 + d.r_gap - d.r_cc).abs().max()),
    "ann_vol_close_to_close": float(np.sqrt(ANN * (d.r_cc**2).mean())),
    "ann_vol_from_parts": float(np.sqrt(ANN * tv.mean())),
    "share_intraday_continuous_bv": float(np.minimum(d.bv5, d.rv5).sum() / tv.sum()),
    "share_intraday_jumps_jv": float(d.jv5.sum() / tv.sum()),
    "share_overnight_gap": float(gap2.sum() / tv.sum()),
    "downside_share_of_intraday": float(d.rsm5.sum() / d.rv5.sum()),
    "rv1_over_rv5": float(d.rv1.sum() / d.rv5.sum()),
    "mean_overnight_return_ann": float(d.r_gap.mean() * ANN), "mean_intraday_return_ann": float(d.sum_r1.mean() * ANN),
}
print(json.dumps(R["variance"], indent=1))

f = vm.frictionless_excess(d, mac)                       # daily excess return incl. dividends
sig = vm.variance_signals(d)
fwd1, fwd1v = f.shift(-1), tv.shift(-1)
H = 21
fwdH = f.rolling(H).sum().shift(-H)
fwdHv = tv.rolling(H).sum().shift(-H)

# quintiles of lagged RV20 (the v8 signal): what the passive index did next
q = pd.qcut(sig["rv20"], 5, labels=False)
rows = []
total_log = d["r_cc"].shift(-1)
for k in range(5):
    m = q == k
    mu, var = fwd1[m].mean(), fwd1v[m].mean()
    rows.append({"quintile of RV20": k + 1, "days": int(m.sum()), "RV20 as annual vol": float(np.sqrt(ANN * sig.rv20[m].mean())),
                 "next-day mean excess (ann.)": mu * ANN, "next-day vol (ann.)": float(np.sqrt(ANN * var)),
                 "mean / variance": mu / var, "Sharpe (ann.)": mu / fwd1[m].std() * np.sqrt(ANN),
                 "t(mean)": vm.hac_mean_tstat(fwd1[m].dropna().to_numpy()),
                 "share of total log return": float(total_log[m].sum() / total_log[q.notna()].sum()),
                 "next-21d mean / variance": fwdH[m].mean() / fwdHv[m].mean()})
quint = pd.DataFrame(rows).set_index("quintile of RV20")
show("Passive index by quintile of lagged RV20", quint)
quint.to_csv(OUT / "passive_by_rv20_quintile.csv")

# predictive regressions on standardised signals
rows = []
for name in ("rv20", "rvi20", "bv20", "rsm20", "har"):
    s = sig[name]
    z = (s - s.mean()) / s.std()
    r1 = rel.spanning_regression(fwd1.to_numpy(), z.to_numpy())
    rH = rel.spanning_regression(fwdH.to_numpy(), z.to_numpy(), lags=2 * H)
    ratio = (fwdH / fwdHv)
    rr = rel.spanning_regression(ratio.to_numpy(), z.to_numpy(), lags=2 * H)
    rv = rel.spanning_regression(np.log(fwd1v).to_numpy(), np.log(s).to_numpy())
    rows.append({"signal": name, "next-day return: slope per 1 sd (ann. %)": r1["beta"] * ANN * 100, "t": r1["t_beta"],
                 "next-21d return: slope per 1 sd (%)": rH["beta"] * 100, "t ": rH["t_beta"],
                 "next-21d return / variance: mean": float(ratio.mean()), "slope per 1 sd": rr["beta"], "t  ": rr["t_beta"],
                 "log next-day variance on log signal: slope": rv["beta"], "R2": rv["r2"]})
pred = pd.DataFrame(rows).set_index("signal")
show("Does the variance signal predict the return, or only the variance?", pred)
pred.to_csv(OUT / "predictive_regressions.csv")

# the v8 filter on the passive index against the breakout trades it removed
allow = filt["filter_allows_next_session"].astype("boolean")
oos = (f.index >= "2019-02-01")
rows = []
for lab, m in (("filter allows", (allow == True) & oos), ("filter blocks", (allow == False) & oos)):   # noqa: E712
    x = f[m.fillna(False).to_numpy()]
    tr = trades[trades["filter_allows"] == (lab == "filter allows")]
    rows.append({"state": lab, "sessions": int(len(x)), "passive mean excess (ann.)": x.mean() * ANN,
                 "passive vol (ann.)": x.std() * np.sqrt(ANN), "passive Sharpe": rel.sharpe_ann(x.to_numpy()),
                 "passive share of total return": float(x.sum() / f[oos].sum()),
                 "breakout trades": int(len(tr)), "breakout mean R": float(tr.r_multiple.mean()),
                 "breakout net EUR": float(tr.net_pnl.sum()), "breakout win rate": float((tr.net_pnl > 0).mean())})
gate = pd.DataFrame(rows).set_index("state")
show("v8 RV20 filter: passive index vs the breakout system, Feb 2019 - Jul 2025", gate)
gate.to_csv(OUT / "v8_filter_decomposition.csv")
R["filter"] = gate.to_dict(orient="index")

# ------------------------------------------------------------- 2/3. the accounts
CAPITAL = RetailCosts()
PEPPER = RetailCosts(spread_points=1.0, fx_fee=0.0, funding_markup=0.025)
TAX = GermanTax()
w_all = {name: vm.managed_weights(sig[name], d["r_cc"], "causal", 1.0)["w"] for name in sig.columns}
w_all["rv20_inverse_vol"] = vm.managed_weights(sig["rv20"], d["r_cc"], "causal", 0.5)["w"]
first = w_all["rv20"].shift(1).first_valid_index()
WINDOWS = {"primary": (first, d.index[-1]), "v8_oos": (pd.Timestamp("2019-02-01"), d.index[-1])}
R["windows"] = {k: [str(a.date()), str(b.date())] for k, (a, b) in WINDOWS.items()}


def target(mkt: MarketDays, name: str) -> np.ndarray:
    return w_all[name].shift(1).reindex(mkt.index).to_numpy()


def evaluate(acct: vm.AccountResult, bench: vm.AccountResult, idle: np.ndarray, col: str = "liq_post", boot: bool = True) -> dict:
    a, b = acct.returns(col).to_numpy() - idle, bench.returns(col).to_numpy() - idle
    sp = rel.spanning_regression(a, b)
    out = {"alpha_ann": sp["alpha_ann"], "t_alpha": sp["t_alpha"], "beta": sp["beta"], "appraisal": sp["appraisal_ratio"],
           "d_sharpe": rel.sharpe_ann(a) - rel.sharpe_ann(b),
           **{f"d_cer{g}": rel.cer_ann(a, g) - rel.cer_ann(b, g) for g in (2, 3, 5)},
           "d_terminal": float(acct.frame[col].iloc[-1] - bench.frame[col].iloc[-1])}
    if boot:
        s = rel.paired_difference_test(a, b, "sharpe", n_boot=NB, seed=1)
        out |= {"d_sharpe_p1": s["p_one_sided"], "d_sharpe_p2": s["p_two_sided"], "d_sharpe_ci": s["ci95_ann"]}
        for g in (2, 3, 5):
            c = rel.paired_difference_test(a, b, "cer", gamma=g, n_boot=NB, seed=1)
            out |= {f"d_cer{g}_p1": c["p_one_sided"], f"d_cer{g}_ci": c["ci95_ann"]}
    return out


def describe(acct: vm.AccountResult, years: float, col: str = "liq_post") -> dict:
    r = acct.returns(col).to_numpy()
    v = acct.frame[col].to_numpy()
    peak = np.maximum.accumulate(np.r_[acct.initial, v])[1:]
    t = acct.totals
    return {"terminal": float(v[-1]), "cagr": float((v[-1] / acct.initial) ** (1 / years) - 1), "vol": float(r.std() * np.sqrt(ANN)),
            "sharpe": rel.sharpe_ann(r), "max_dd": float(np.max(1 - v / peak)), "avg_weight": float(acct.frame["weight"].mean()),
            "rebalances_per_year": t["rebalances"] / years, "spread": t["spread"], "fx_fee": t["fx_fee"], "funding": t["funding"],
            "dividends": t["dividends"], "interest": t["interest"], "tax": t["tax"], "cer3": rel.cer_ann(r, 3.0)}


SPECS = {   # name: (signal, kwargs for the CFD simulator)
    "PRIMARY: RV20, cap 1.0, deadband 0.10, idle 0%": ("rv20", {}),
    "deadband 0": ("rv20", {"deadband": 0.0}),
    "deadband 0.05": ("rv20", {"deadband": 0.05}),
    "deadband 0.20": ("rv20", {"deadband": 0.20}),
    "idle cash swept to EUR overnight ETF": ("rv20", {"sweep": CashSweep()}),
    "cap 1.5, Capital.com funding cliff": ("rv20", {"w_max": 1.5, "funding": "cliff"}),
    "cap 1.5, Pepperstone funding cliff": ("rv20", {"w_max": 1.5, "funding": "cliff", "costs": PEPPER}),
    "signal: intraday RV + gap": ("rvi20", {}),
    "signal: bipower + gap": ("bv20", {}),
    "signal: downside semivariance": ("rsm20", {}),
    "signal: HAR forecast": ("har", {}),
    "inverse volatility weights": ("rv20_inverse_vol", {}),
    "CFD dividend credit 70%": ("rv20", {"costs": replace(CAPITAL, dividend_credit=0.7)}),
}
TAXES = {"spec": TAX, "allowance": GermanTax(allowance_to_2022=801.0, allowance_from_2023=1000.0),
         "vorab": GermanTax(vorabpauschale=True)}
table, detail = [], {}
for wname, (a0, a1) in WINDOWS.items():
    mkt = MarketDays.from_frames(d, mac, a0, a1)
    years = len(mkt) / ANN
    zero = np.zeros(len(mkt))
    b1 = vm.simulate_cfd_account(mkt, np.r_[1.0, np.full(len(mkt) - 1, np.nan)], costs=CAPITAL, tax=TAX, rebalance=False)
    b2 = vm.simulate_etf_buy_and_hold(mkt, tax=TAX)
    detail[wname] = {"B1 buy-and-hold 1:1 CFD": describe(b1, years), "B2 buy-and-hold accumulating ETF": describe(b2, years)}
    for name, (signal, kw) in SPECS.items():
        if wname != "primary" and not name.startswith("PRIMARY"):
            continue
        kw = {"costs": CAPITAL, "tax": TAX} | kw
        acct = vm.simulate_cfd_account(mkt, target(mkt, signal), **kw)
        idle = zero
        if kw.get("sweep") is not None:
            idle = np.r_[0.0, [kw["sweep"].net_rate(x) for x in mkt.eur_rate[:-1]] * mkt.nights[1:] / 360.0]
        boot = name.startswith("PRIMARY")
        row = {"window": wname, "spec": name, **describe(acct, years),
               **{f"B1_{k}": v for k, v in evaluate(acct, b1, idle, boot=boot).items()},
               **{f"B2_{k}": v for k, v in evaluate(acct, b2, idle, boot=boot).items()}}
        if boot:
            # the same three accounts with no tax at all (separate runs: nothing is paid along the way either)
            off = GermanTax(enabled=False)
            a_nt = vm.simulate_cfd_account(mkt, target(mkt, signal), **(kw | {"tax": off}))
            b1_nt = vm.simulate_cfd_account(mkt, np.r_[1.0, np.full(len(mkt) - 1, np.nan)], costs=CAPITAL, tax=off, rebalance=False)
            b2_nt = vm.simulate_etf_buy_and_hold(mkt, tax=off)
            pre = {**{f"B1_{k}": v for k, v in evaluate(a_nt, b1_nt, idle, boot=False).items()},
                   **{f"B2_{k}": v for k, v in evaluate(a_nt, b2_nt, idle, boot=False).items()}}
            detail[wname]["PRIMARY pre-tax"] = {**describe(a_nt, years), **pre}
            detail[wname]["B1 pre-tax"], detail[wname]["B2 pre-tax"] = describe(b1_nt, years), describe(b2_nt, years)
            detail[wname]["PRIMARY post-tax"] = row
            # block-length checks and the terminal-wealth bootstrap
            a_r, b_r = acct.returns().to_numpy(), b2.returns().to_numpy()
            detail[wname]["block_checks_B2"] = {
                str(bl): {"d_sharpe_p1": rel.paired_difference_test(a_r, b_r, "sharpe", block=bl, n_boot=NB, seed=2)["p_one_sided"],
                          "d_cer3_p1": rel.paired_difference_test(a_r, b_r, "cer", gamma=3.0, block=bl, n_boot=NB, seed=2)["p_one_sided"]}
                for bl in (5, 10, 21)}
            tw = rel.paired_block_bootstrap(a_r, b_r, lambda x, y: 5000.0 * (np.prod(1 + x, axis=1) - np.prod(1 + y, axis=1)),
                                            block=21, n_boot=NB, seed=3)
            detail[wname]["terminal_wealth_bootstrap_B2"] = tw
            acct.frame.join(b1.frame.add_prefix("b1_")).join(b2.frame.add_prefix("b2_")).round(4).to_csv(OUT / f"accounts_{wname}.csv")
        table.append(row)
    # tax sensitivities of the primary specification
    for tname, tx in TAXES.items():
        if tname == "spec":
            continue
        acct = vm.simulate_cfd_account(mkt, target(mkt, "rv20"), costs=CAPITAL, tax=tx)
        b1t = vm.simulate_cfd_account(mkt, np.r_[1.0, np.full(len(mkt) - 1, np.nan)], costs=CAPITAL, tax=tx, rebalance=False)
        b2t = vm.simulate_etf_buy_and_hold(mkt, tax=tx)
        table.append({"window": wname, "spec": f"tax: {tname} (benchmarks under the same rule)", **describe(acct, years),
                      **{f"B1_{k}": v for k, v in evaluate(acct, b1t, zero, boot=False).items()},
                      **{f"B2_{k}": v for k, v in evaluate(acct, b2t, zero, boot=False).items()},
                      "b1_terminal": b1t.terminal, "b2_terminal": b2t.terminal})
tab = pd.DataFrame(table)
tab.to_csv(OUT / "accounts_table.csv", index=False)
cols = ["window", "spec", "terminal", "cagr", "vol", "sharpe", "max_dd", "avg_weight", "rebalances_per_year", "spread", "fx_fee",
        "funding", "tax", "B1_alpha_ann", "B1_t_alpha", "B1_d_sharpe", "B1_d_cer3", "B1_d_terminal", "B2_alpha_ann", "B2_t_alpha",
        "B2_d_sharpe", "B2_d_cer3", "B2_d_terminal"]
show("Accounts (post-tax, liquidation basis)", tab[cols].set_index(["window", "spec"]), 4)
for wname in WINDOWS:
    print(f"\n### benchmarks, {wname}")
    print(pd.DataFrame({k: v for k, v in detail[wname].items() if k.startswith("B")}).T[
        ["terminal", "cagr", "vol", "sharpe", "max_dd", "spread", "fx_fee", "dividends", "tax", "cer3"]].round(4).to_string())
    p = detail[wname]["PRIMARY post-tax"]
    print({k: (np.round(v, 4).tolist() if isinstance(v, tuple) else round(v, 4)) for k, v in p.items()
           if k.startswith(("B1_", "B2_"))})
    print("no tax at all:", {k: round(v, 4) for k, v in detail[wname]["PRIMARY pre-tax"].items()
                             if k.startswith(("B1_", "B2_")) or k in ("terminal", "sharpe", "vol", "max_dd")})
    print("block checks vs B2:", detail[wname]["block_checks_B2"], "terminal wealth bootstrap:", detail[wname]["terminal_wealth_bootstrap_B2"])
R["accounts"] = detail

# ------------------------------------------------- 4b. literature attenuation ladder
a0, a1 = WINDOWS["primary"]
fw = f[(f.index >= a0) & (f.index <= a1)]
ladder = []


def frictionless(name: str, scalar: str, cap: float | None) -> None:
    w = vm.managed_weights(sig["rv20"], f, scalar, 1.0, sample=slice(a0, a1))["w"]
    if cap is not None:
        w = w.clip(upper=cap)
    m = (w.shift(1) * f).reindex(fw.index)
    sp = rel.spanning_regression(m.to_numpy(), fw.to_numpy())
    ladder.append({"step": name, "alpha_ann_pct": sp["alpha_ann"] * 100, "t_alpha": sp["t_alpha"], "appraisal": sp["appraisal_ratio"],
                   "beta": sp["beta"], "mean_weight": float(w.shift(1).reindex(fw.index).mean()),
                   "d_sharpe": rel.sharpe_ann(m.dropna().to_numpy()) - rel.sharpe_ann(fw.to_numpy())})


frictionless("1. Moreira-Muir: in-sample scalar, no cap, frictionless, cash at the USD rate", "insample", None)
frictionless("2. real-time scalar (Cederburg et al.), no cap, frictionless", "causal", None)
frictionless("3. real-time scalar, cap 1.5, frictionless", "causal", 1.5)
frictionless("4. real-time scalar, cap 1.0, frictionless", "causal", 1.0)
mkt = MarketDays.from_frames(d, mac, a0, a1)
b1_free = vm.simulate_cfd_account(mkt, np.r_[1.0, np.full(len(mkt) - 1, np.nan)], costs=replace(CAPITAL, spread_points=0.0, fx_fee=0.0),
                                  tax=GermanTax(enabled=False), rebalance=False)
steps = [("5. EUR 1:1 CFD account, cap 1.0, traded at next open, daily, no costs, idle cash 0%",
          dict(deadband=0.0, costs=replace(CAPITAL, spread_points=0.0, fx_fee=0.0), tax=GermanTax(enabled=False)), "free"),
         ("6. + spread and conversion mark-up", dict(deadband=0.0, costs=CAPITAL, tax=GermanTax(enabled=False)), "cost"),
         ("7. + deadband 0.10", dict(deadband=0.10, costs=CAPITAL, tax=GermanTax(enabled=False)), "cost"),
         ("8. + German tax (primary specification, vs same-vehicle buy-and-hold)", dict(deadband=0.10, costs=CAPITAL, tax=TAX), "tax")]
bench = {"free": b1_free,
         "cost": vm.simulate_cfd_account(mkt, np.r_[1.0, np.full(len(mkt) - 1, np.nan)], costs=CAPITAL, tax=GermanTax(enabled=False), rebalance=False),
         "tax": vm.simulate_cfd_account(mkt, np.r_[1.0, np.full(len(mkt) - 1, np.nan)], costs=CAPITAL, tax=TAX, rebalance=False)}
for name, kw, bk in steps:
    acct = vm.simulate_cfd_account(mkt, target(mkt, "rv20"), **kw)
    e = evaluate(acct, bench[bk], np.zeros(len(mkt)), boot=False)
    ladder.append({"step": name, "alpha_ann_pct": e["alpha_ann"] * 100, "t_alpha": e["t_alpha"], "appraisal": e["appraisal"],
                   "beta": e["beta"], "mean_weight": float(acct.frame["weight"].mean()), "d_sharpe": e["d_sharpe"]})
acct = vm.simulate_cfd_account(mkt, target(mkt, "rv20"), costs=CAPITAL, tax=TAX)
e = evaluate(acct, vm.simulate_etf_buy_and_hold(mkt, tax=TAX), np.zeros(len(mkt)), boot=False)
ladder.append({"step": "9. primary specification vs buy-and-hold ETF, post-tax", "alpha_ann_pct": e["alpha_ann"] * 100,
               "t_alpha": e["t_alpha"], "appraisal": e["appraisal"], "beta": e["beta"],
               "mean_weight": float(acct.frame["weight"].mean()), "d_sharpe": e["d_sharpe"]})
lad = pd.DataFrame(ladder).set_index("step")
show("From the textbook to the retail account: spanning alpha step by step (primary window)", lad)
lad.to_csv(OUT / "attenuation_ladder.csv")
R["ladder"] = lad.to_dict(orient="index")
# the same first two steps for every signal
rows = []
for name in sig.columns:
    for scalar in ("insample", "causal"):
        w = vm.managed_weights(sig[name], f, scalar, 1.0, sample=slice(a0, a1))["w"]
        m = (w.shift(1) * f).reindex(fw.index)
        sp = rel.spanning_regression(m.to_numpy(), fw.to_numpy())
        rows.append({"signal": name, "scalar": scalar, "alpha_ann_pct": sp["alpha_ann"] * 100, "t_alpha": sp["t_alpha"],
                     "appraisal": sp["appraisal_ratio"], "vol_ratio": float(m.std() / fw.std())})
show("Frictionless managed index by signal", pd.DataFrame(rows).set_index(["signal", "scalar"]))
pd.DataFrame(rows).to_csv(OUT / "frictionless_by_signal.csv", index=False)

# ---------------------------------------------------------------------------------
# NOT IN THE PLAN.  Added after the planned results above had been seen: the literal v8
# filter (hold the index while RV20 is at or below the 80th percentile of its own trailing
# 252 sessions, otherwise hold cash) as a passive-exposure rule.  The planned primary was the
# textbook inverse-variance weight; this is the rule that actually sat in v8.
# ---------------------------------------------------------------------------------
from ndxswing.features.indicators import trailing_pct_rank

rv20_ann = np.sqrt(ANN * sig["rv20"])
pct = trailing_pct_rank(rv20_ann, 252)
assert np.allclose(pct.reindex(filt.index).dropna(), filt["rv20_pct"].reindex(pct.dropna().index).dropna(), atol=1e-9)
rows = []
spans = {"primary": WINDOWS["primary"], "v8_oos": WINDOWS["v8_oos"],
         "before the v8 OOS span": (WINDOWS["primary"][0], pd.Timestamp("2019-01-31"))}
for thr in (0.70, 0.80, 0.90):
    allow_next = (pct <= thr).shift(1)
    for wname, (a0, a1) in spans.items():
        mkt = MarketDays.from_frames(d, mac, a0, a1)
        years = len(mkt) / ANN
        tgt = allow_next.reindex(mkt.index).astype(float).to_numpy()
        acct = vm.simulate_cfd_account(mkt, tgt, deadband=0.5, costs=CAPITAL, tax=TAX)
        b1 = vm.simulate_cfd_account(mkt, np.r_[1.0, np.full(len(mkt) - 1, np.nan)], costs=CAPITAL, tax=TAX, rebalance=False)
        b2 = vm.simulate_etf_buy_and_hold(mkt, tax=TAX)
        boot = thr == 0.80
        x = f.reindex(mkt.index)
        al = allow_next.reindex(mkt.index).astype(bool).to_numpy()
        dm = rel.spanning_regression(x.to_numpy(), (~al).astype(float))
        rows.append({"threshold": thr, "window": wname, **describe(acct, years),
                     "blocked_share": float((~al).mean()), "passive_mean_allowed": float(x[al].mean() * ANN),
                     "passive_mean_blocked": float(x[~al].mean() * ANN), "t_difference": dm["t_beta"],
                     **{f"B1_{k}": v for k, v in evaluate(acct, b1, np.zeros(len(mkt)), boot=boot).items()},
                     **{f"B2_{k}": v for k, v in evaluate(acct, b2, np.zeros(len(mkt)), boot=boot).items()},
                     "b1_terminal": b1.terminal, "b2_terminal": b2.terminal})
post = pd.DataFrame(rows)
post.to_csv(OUT / "posthoc_binary_filter.csv", index=False)
show("POST HOC (not in the plan): the literal v8 RV20 percentile filter as a passive in-or-out rule, post-tax",
     post[["threshold", "window", "terminal", "b1_terminal", "b2_terminal", "sharpe", "max_dd", "avg_weight", "rebalances_per_year",
           "tax", "blocked_share", "passive_mean_allowed", "passive_mean_blocked", "t_difference", "B2_alpha_ann", "B2_t_alpha",
           "B2_d_sharpe", "B2_d_cer2", "B2_d_cer3", "B2_d_cer5", "B1_alpha_ann", "B1_t_alpha", "B1_d_cer3"]].set_index(["threshold", "window"]), 4)
b = post[(post.threshold == 0.80)].set_index("window")
print({w: {k: (np.round(b.loc[w, k], 4).tolist() if isinstance(b.loc[w, k], tuple) else round(float(b.loc[w, k]), 4))
           for k in ("B2_d_sharpe_p1", "B2_d_cer3_p1", "B2_d_cer2_p1", "B2_d_cer5_p1", "B1_d_sharpe_p1", "B1_d_cer3_p1")} for w in b.index})
R["posthoc_binary_filter"] = post.to_dict(orient="records")

(OUT / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
