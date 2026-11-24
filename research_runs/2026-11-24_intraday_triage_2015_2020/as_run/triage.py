"""Intraday triage, 2015-01-01 .. 2020-12-31 ONLY.  Bars after 2020 are filtered out by the parquet
reader and never enter memory.  No parameter is tuned: every number below is the first and only run."""
import sys
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa

SRC, DAILY, MACRO = sys.argv[1], sys.argv[2], sys.argv[3]
END = pd.Timestamp("2021-01-01", tz="UTC")
P_REF = 25_288.25                      # index level at which the lot rules bite (project convention)
FLAT = 380                             # 15:50 ET
tbl = ds.dataset(SRC).to_table(filter=ds.field("ts") < pa.scalar(END.to_pydatetime(), type=pa.timestamp("ns", tz="UTC")))
m = tbl.to_pandas().set_index("ts").sort_index()
assert m.index.max() < END
m.index = m.index.tz_convert("America/New_York")
day = m.index.tz_localize(None).normalize()
mi = (m.index.hour * 60 + m.index.minute - 570).to_numpy()
sess = pd.read_csv(DAILY, index_col=0, parse_dates=True, usecols=["date", "n_min", "close"])
sess = sess[sess.index < "2021-01-01"]
fx = pd.read_csv(MACRO, index_col=0, parse_dates=True, usecols=["date", "eurusd"])["eurusd"]
fx = fx[fx.index < "2021-01-01"]
prev_close = sess["close"].shift(1)
full = sess.index[sess["n_min"] >= 385]
O = np.full((len(full), 390), np.nan); H = O.copy(); L = O.copy(); C = O.copy()
pos = pd.Series(np.arange(len(full)), index=full)
ok = day.isin(full) & (mi >= 0) & (mi < 390)
r = pos.reindex(day[ok]).to_numpy().astype(int); c = mi[ok]
for A, col in ((O, "open"), (H, "high"), (L, "low"), (C, "close")):
    A[r, c] = m[col].to_numpy()[ok]
for j in range(390):                    # a missing minute repeats the last price
    miss = np.isnan(C[:, j])
    if miss.any():
        fill = C[miss, j - 1] if j else O[miss, 1]
        O[miss, j] = H[miss, j] = L[miss, j] = C[miss, j] = fill
D = len(full)
PC = prev_close.reindex(full).to_numpy(); FX = fx.reindex(full).ffill().to_numpy()
print(f"sessions used: {D} full sessions {full[0].date()} .. {full[-1].date()} (of {len(sess)}; early closes and broken sessions excluded)")

BROKERS = {"Pepperstone 0.10 lot": dict(hs=0.5, slip=0.5, fee=0.0, step=0.10),
           "Capital.com 0.001 lot": dict(hs=0.9, slip=0.5, fee=0.007, step=0.001)}


def walk(d, k, j0, stop, tp, hs, slip, entry_minute_stop_only=False):
    """Exit of a position of direction k on day d from minute j0: (exit fill, exit cost in points, reason)."""
    for j in range(j0, FLAT):
        o, h, l = O[d, j], H[d, j], L[d, j]
        adv, fav = (l, h) if k > 0 else (h, l)
        if (stop - o) * k >= 0 and not (entry_minute_stop_only and j == j0):
            return o - k * (hs + slip), hs + slip, "stop_gap"
        if (stop - adv) * k >= 0:
            return stop - k * (hs + slip), hs + slip, "stop"
        if tp is not None and ((fav - k * hs) - tp) * k >= 0:
            px = max(tp, o - hs) if k > 0 else min(tp, o + hs)
            return (px if not (entry_minute_stop_only and j == j0) else tp), hs, "target"
    o = O[d, FLAT]
    return o - k * (hs + slip), hs + slip, "time"


def signals():
    """Broker-independent set-ups: day, direction, entry minute, entry mid, stop, target, planned risk in points, entry type."""
    out = {k: [] for k in ("1A opening-range breakout", "1B failed breakout, target mid-range", "1B failed breakout, target far edge",
                           "2 gap fill", "3 power-hour continuation")}
    skipped_small = {k: 0 for k in out}
    for d in range(D):
        o930 = O[d, 0]
        orh, orl = H[d, :30].max(), L[d, :30].min()
        rng = orh - orl
        # ---- 1A
        if 0.0025 <= rng / o930 <= 0.01:
            for j in range(30, 150):
                up, dn = H[d, j] >= orh, L[d, j] <= orl
                if up and dn:
                    break
                if up or dn:
                    k = 1 if up else -1
                    lvl = orh if up else orl
                    mid = max(lvl, O[d, j]) if up else min(lvl, O[d, j])
                    out["1A opening-range breakout"].append(dict(d=d, k=k, j=j, mid=mid, stop=orl if up else orh,
                                                                 tp=lvl + k * 2.0 * rng, risk=rng, kind="stop"))
                    break
        # ---- 1B: six 15-minute bars 10:00 .. 11:30
        for b in range(6):
            a, e = 30 + 15 * b, 45 + 15 * b
            bh, bl, bc = H[d, a:e].max(), L[d, a:e].min(), C[d, e - 1]
            hi, lo = bh > orh, bl < orl
            if hi and lo:
                break
            if (hi or lo) and orl <= bc <= orh:
                k = -1 if hi else 1
                ext = bh if hi else bl
                stop = ext - k * 0.5 * rng
                if (bc - stop) * k < 0.0025 * bc:
                    stop = bc - k * 0.0025 * bc                      # stop distance floor of 0.25% of the index
                risk = abs(bc - stop)
                for name, tgt in (("1B failed breakout, target mid-range", 0.5 * (orh + orl)),
                                  ("1B failed breakout, target far edge", orl if hi else orh)):
                    if (tgt - bc) * k > 0:
                        out[name].append(dict(d=d, k=k, j=e, mid=bc, stop=stop, tp=tgt, risk=risk, kind="market"))
                break
        # ---- 2
        if np.isfinite(PC[d]):
            gap = (o930 - PC[d]) / PC[d]
            c10 = C[d, 29]
            if 0.0025 <= abs(gap) <= 0.0085:
                k = -1 if gap > 0 else 1
                if (c10 - o930) * k > 0 and (PC[d] - c10) * k > 0:
                    entry = O[d, 30]
                    stop = (orh + 2.0) if k < 0 else (orl - 2.0)
                    risk = (entry - stop) * k
                    if risk > 0 and (PC[d] - entry) * k >= risk:
                        if risk < 0.0025 * entry:
                            skipped_small["2 gap fill"] += 1
                        else:
                            out["2 gap fill"].append(dict(d=d, k=k, j=30, mid=entry, stop=stop, tp=PC[d], risk=risk, kind="market"))
        # ---- 3
        sh, sl, c14 = H[d, :270].max(), L[d, :270].min(), C[d, 269]
        sr = sh - sl
        if sr >= 0.006 * o930:
            posn = (c14 - sl) / sr
            k = 1 if posn >= 0.8 else (-1 if posn <= 0.2 else 0)
            if k:
                entry = O[d, 270]
                risk = 0.35 * sr
                if risk < 0.0025 * entry:
                    skipped_small["3 power-hour continuation"] += 1
                else:
                    out["3 power-hour continuation"].append(dict(d=d, k=k, j=270, mid=entry, stop=c14 - k * risk, tp=None,
                                                                 risk=risk, kind="market"))
    return out, skipped_small


def simulate(sig, hs, slip, fee):
    rows = []
    for s in sig:
        d, k = s["d"], s["k"]
        fill = s["mid"] + k * (hs + slip)
        ex, xcost, why = walk(d, k, s["j"], s["stop"], s["tp"], hs, slip, entry_minute_stop_only=(s["kind"] == "stop"))
        pts = (ex - fill) * k
        gross = pts + hs + slip + xcost
        rows.append(dict(d=d, k=k, risk=s["risk"], mid=s["mid"], pts=pts, gross_r=gross / s["risk"],
                         net_r=(pts - fee * abs(pts)) / s["risk"], why=why, j=s["j"],
                         stop_pct=s["risk"] / s["mid"], tp_pct=None if s["tp"] is None else abs(s["tp"] - s["mid"]) / s["mid"]))
    return pd.DataFrame(rows)


def random_matrix(t, hs, slip, fee):
    """Net R of each trade's bracket (same minute, direction, stop and target distance in % of price) on EVERY day."""
    out = np.empty((len(t), D))
    for i, s in enumerate(t.itertuples()):
        k, j = s.k, s.j
        e = O[:, j]
        stop = e * (1 - k * s.stop_pct)
        hh, ll, oo = H[:, j:FLAT], L[:, j:FLAT], O[:, j:FLAT]
        adv = ll if k > 0 else hh
        hit_s = (stop[:, None] - adv) * k >= 0
        i_s = np.where(hit_s.any(1), hit_s.argmax(1), 10_000)
        if s.tp_pct is not None and np.isfinite(s.tp_pct):
            tp = e * (1 + k * s.tp_pct)
            fav = (hh - hs) if k > 0 else (ll + hs)
            hit_t = (fav - tp[:, None]) * k >= 0
            i_t = np.where(hit_t.any(1), hit_t.argmax(1), 10_000)
        else:
            tp, i_t = e, np.full(D, 10_000)
        rows = np.arange(D)
        o_s = oo[rows, np.minimum(i_s, oo.shape[1] - 1)]
        stop_px = np.where((stop - o_s) * k >= 0, o_s, stop) - k * (hs + slip)
        o_t = oo[rows, np.minimum(i_t, oo.shape[1] - 1)]
        tp_px = np.maximum(tp, o_t - hs) if k > 0 else np.minimum(tp, o_t + hs)
        time_px = O[:, FLAT] - k * (hs + slip)
        ex = np.where((i_s <= i_t) & (i_s < 10_000), stop_px, np.where(i_t < 10_000, tp_px, time_px))
        pts = (ex - (e + k * (hs + slip))) * k
        out[i] = (pts - fee * np.abs(pts)) / (s.stop_pct * e)
    return out


SIG, small = signals()
rng = np.random.default_rng(20201231)
rows = []
for cand, sig in SIG.items():
    for bname, b in BROKERS.items():
        t = simulate(sig, b["hs"], b["slip"], b["fee"])
        rm = random_matrix(t, b["hs"], b["slip"], b["fee"])
        for risk_eur in (12.50, 25.00):
            fxd = FX[t.d.to_numpy()]
            for norm in (True, False):
                step = b["step"] * (P_REF / t.mid.to_numpy() if norm else 1.0)
                units = np.floor(risk_eur * fxd / t.risk.to_numpy() / step + 1e-9) * step
                take = units >= step - 1e-12
                if norm:
                    keep, u = take, units
                else:
                    raw_skip = int((~take).sum())
            x = t[keep]
            pnl = u[keep] * (x.pts - b["fee"] * x.pts.abs()).to_numpy() / fxd[keep]
            daily = np.zeros(D); np.add.at(daily, x.d.to_numpy(), pnl / 5000.0)
            nr = x.net_r.to_numpy()
            n = len(x)
            sub = rm[keep]
            draws = sub[np.arange(n)[None, :], rng.integers(0, D, size=(4000, n))].mean(1) if n else np.array([np.nan])
            lg, sh = x[x.k > 0], x[x.k < 0]
            rows.append({
                "Candidate": cand, "Broker": bname, "Risk": f"EUR {risk_eur:.2f}", "Signals": len(t),
                "Skipped: stop < 0.25%": small[cand], "Trades": n, "Skipped: min lot": int((~keep).sum()),
                "Skipped at historical prices": raw_skip,
                "Win rate": float((nr > 0).mean()) if n else np.nan, "Gross exp (R)": float(x.gross_r.mean()) if n else np.nan,
                "Net exp (R)": float(nr.mean()) if n else np.nan,
                "Net Sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan,
                "t": float(nr.mean() / nr.std(ddof=1) * np.sqrt(n)) if n > 2 else np.nan,
                "Long n": len(lg), "Long net R": float(lg.net_r.mean()) if len(lg) else np.nan,
                "Short n": len(sh), "Short net R": float(sh.net_r.mean()) if len(sh) else np.nan,
                "Random net R": float(sub.mean()) if n else np.nan,
                "Share of random >= actual": float((draws >= nr.mean()).mean()) if n else np.nan,
                "Net EUR total": float(pnl.sum()), "exits": x.why.value_counts().to_dict()})
res = pd.DataFrame(rows)
res.to_csv(sys.argv[4], index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 60)
print(res.drop(columns=["exits"]).round(3).to_string(index=False))
print(res[res.Risk == "EUR 25.00"][["Candidate", "Broker", "exits"]].to_string(index=False))
