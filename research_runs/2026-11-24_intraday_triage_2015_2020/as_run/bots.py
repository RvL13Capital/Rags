"""Automated intraday bot triage, 2015-01-01 .. 2020-12-31 ONLY (later bars are filtered out by the
parquet reader and never enter memory).  Costs are fractions of the contemporaneous price.
Nothing is tuned: every number is the first and only run of the rule as written."""
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

SRC, DAILY, MACRO, EVENTS, OUT = sys.argv[1:6]
END = pd.Timestamp("2021-01-01", tz="UTC")
P_REF, FLAT, BIG = 25_288.25, 385, 10_000          # lot rules bite at P_REF; flat at 15:55 ET
tbl = ds.dataset(SRC).to_table(filter=ds.field("ts") < pa.scalar(END.to_pydatetime(), type=pa.timestamp("ns", tz="UTC")))
m = tbl.to_pandas().set_index("ts").sort_index()
assert m.index.max() < END
m.index = m.index.tz_convert("America/New_York")
day = m.index.tz_localize(None).normalize()
mi = (m.index.hour * 60 + m.index.minute - 570).to_numpy()
sess = pd.read_csv(DAILY, index_col=0, parse_dates=True, usecols=["date", "n_min", "close"])
sess = sess[sess.index < "2021-01-01"]
fxs = pd.read_csv(MACRO, index_col=0, parse_dates=True, usecols=["date", "eurusd"])["eurusd"]
ev = pd.read_csv(EVENTS, parse_dates=["ts"])
ev = ev[ev.ts < "2021-01-01"]
full = sess.index[sess["n_min"] >= 385]
D = len(full)
O = np.full((D, 390), np.nan); H = O.copy(); L = O.copy(); C = O.copy(); V = np.zeros((D, 390))
pos = pd.Series(np.arange(D), index=full)
ok = day.isin(full) & (mi >= 0) & (mi < 390)
r_, c_ = pos.reindex(day[ok]).to_numpy().astype(int), mi[ok]
for A, col in ((O, "open"), (H, "high"), (L, "low"), (C, "close"), (V, "volume")):
    A[r_, c_] = m[col].to_numpy()[ok]
for j in range(390):
    miss = np.isnan(C[:, j])
    if miss.any():
        O[miss, j] = H[miss, j] = L[miss, j] = C[miss, j] = (C[miss, j - 1] if j else O[miss, 1])
PC = sess["close"].shift(1).reindex(full).to_numpy()
FX = fxs[fxs.index < "2021-01-01"].reindex(full).ffill().to_numpy()
ema50 = sess["close"].ewm(span=50, adjust=False).mean()
trend = np.sign((sess["close"] - ema50).shift(1)).reindex(full).to_numpy().copy()   # previous close against its EMA50
trend[: 50] = 0
EVENT_DAY = np.isin(full.values, ev.ts.dt.normalize().unique())
print(f"{D} full sessions {full[0].date()} .. {full[-1].date()}; scheduled FOMC/CPI/NFP days: {int(EVENT_DAY.sum())}")

# ---- session VWAP (typical price x volume, cumulative; equal weights where a session has no volume)
typ = (H + L + C) / 3.0
W = np.where(V.sum(1, keepdims=True) > 0, V, 1.0)
cw = np.cumsum(W, 1)
VW = np.where(cw > 0, np.cumsum(typ * W, 1) / np.where(cw > 0, cw, 1.0), np.cumsum(typ, 1) / np.arange(1, 391))


def wilder_atr(h, l, c, n=14):
    """ATR over a (D, bars) grid read row by row (the overnight gap is part of the first bar's true range)."""
    hf, lf, cf = h.ravel(), l.ravel(), c.ravel()
    pc = np.r_[np.nan, cf[:-1]]
    tr = np.nanmax(np.c_[hf - lf, np.abs(hf - pc), np.abs(lf - pc)], axis=1)
    atr = np.full(len(tr), np.nan)
    atr[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr.reshape(h.shape)


def bars(width):
    nb = 390 // width
    sh = (D, nb, width)
    return (O[:, ::width], H.reshape(sh).max(2), L.reshape(sh).min(2), C[:, width - 1::width], V.reshape(sh).sum(2))


O30, H30, L30, C30, _ = bars(30)
O15, H15, L15, C15, V15 = bars(15)
ATR30, ATR15 = wilder_atr(H30, L30, C30), wilder_atr(H15, L15, C15)


def trailing(x, n, fn):
    """fn over the previous n sessions (same column), row d uses rows d-n .. d-1."""
    out = np.full(x.shape, np.nan)
    for d in range(n, len(x)):
        out[d] = fn(x[d - n:d], axis=0)
    return out


# Bot A inputs
move = np.abs(C30 / O[:, [0]] - 1.0)
SIG = trailing(move, 14, np.mean)                                   # sigma_open at each 30-minute close
base_hi, base_lo = np.fmax(O[:, 0], PC), np.fmin(O[:, 0], PC)
UP, LO = base_hi[:, None] * (1 + SIG), base_lo[:, None] * (1 - SIG)
# Bot B inputs: the six bars before bar b, same session only
R90 = np.full((D, 26), np.nan); PH = R90.copy(); PL = R90.copy()
for b in range(6, 26):
    PH[:, b], PL[:, b] = H15[:, b - 6:b].max(1), L15[:, b - 6:b].min(1)
    R90[:, b] = PH[:, b] - PL[:, b]
Q25 = trailing(R90, 20, lambda a, axis: np.percentile(a, 25, axis=axis))
MEDV = trailing(V15, 20, np.median)
# Bot C inputs
r1 = np.diff(np.log(np.c_[O[:, 0], C]), axis=1)
RV = np.cumsum(r1**2, 1)
BV = np.pi / 2.0 * np.cumsum(np.c_[np.zeros(D), np.abs(r1[:, 1:]) * np.abs(r1[:, :-1])], 1)

BROKERS = {"Pepperstone": dict(hs=0.20e-4, slip=0.20e-4, fee=0.0, step=0.10),
           "Capital.com": dict(hs=0.355e-4, slip=0.20e-4, fee=0.007, step=0.001)}


def first_true(mask):
    return np.where(mask.any(1), mask.argmax(1), BIG)


def settle(dd, k, entry_mid, entry_fill, i_s, stop_at, e_mkt, i_t=None, tp_px=None, hs=0.0, slip=0.0):
    """Common exit resolution.  i_s: minute of the stop, stop_at: stop level in force at that minute,
    e_mkt: minute of a discretionary market exit (or BIG), i_t / tp_px: minute and price of a limit exit."""
    n = len(dd)
    rows = np.arange(n)
    i_t = np.full(n, BIG) if i_t is None else i_t
    is_stop = (i_s <= i_t) & (i_s < e_mkt) & (i_s < FLAT)
    is_tp = ~is_stop & (i_t < e_mkt) & (i_t < FLAT)
    x = np.where(is_stop, i_s, np.where(is_tp, i_t, np.minimum(e_mkt, FLAT)))
    o_x = O[dd, np.minimum(x, 389)]
    stop_mid = np.where((stop_at - o_x) * k >= 0, o_x, stop_at)            # opened through the stop: filled at the open
    mid = np.where(is_stop, stop_mid, o_x)
    fill = mid * (1 - k * (hs + slip))
    if tp_px is not None:
        mid = np.where(is_tp, tp_px * (1 + k * hs), mid)
        fill = np.where(is_tp, tp_px, fill)
    return x, (fill - entry_fill) * k, (mid - entry_mid) * k, np.where(is_stop, "stop", np.where(is_tp, "target", np.where(x < FLAT, "signal", "time")))


def manage_a(dd, i, k, entry_mid, dstop, hs, slip):
    """Bot A from the open of the bar after 30-minute close i."""
    j = 30 * (i + 1)
    stop = entry_mid - k * dstop
    adv = L[dd, j:FLAT] if k > 0 else H[dd, j:FLAT]
    i_s = first_true((stop[:, None] - adv) * k >= 0)
    i_s = np.where(i_s < BIG, i_s + j, BIG)
    ii = np.arange(i + 1, 12)                                              # 30-minute closes after entry, to 15:30
    cm = 29 + 30 * ii
    band = UP[np.ix_(dd, ii)] if k > 0 else LO[np.ix_(dd, ii)]
    vw = VW[np.ix_(dd, cm)]
    thr = np.maximum(band, vw) if k > 0 else np.minimum(band, vw)
    out = (C[np.ix_(dd, cm)] - thr) * k < 0
    e = first_true(out)
    e = np.where(e < BIG, cm[np.minimum(e, len(cm) - 1)] + 1, BIG)
    return settle(dd, k, entry_mid, entry_mid * (1 + k * (hs + slip)), i_s, stop, e, hs=hs, slip=slip)


def manage_b(dd, b, k, entry_mid, dstop, hs, slip):
    """Bot B from the open of bar b + 1: chandelier trail, monotone, updated at each 15-minute close."""
    be = b + 1
    j = 15 * be
    stop0 = entry_mid - k * dstop
    ext = np.maximum.accumulate(H15[dd, be:], 1) if k > 0 else np.minimum.accumulate(L15[dd, be:], 1)
    trail = ext - k * 2.2 * ATR15[dd, be:]
    lvl = np.maximum.accumulate(trail, 1) if k > 0 else np.minimum.accumulate(trail, 1)
    lvl = np.maximum(lvl, stop0[:, None]) if k > 0 else np.minimum(lvl, stop0[:, None])
    per_bar = np.c_[stop0, lvl[:, :-1]]                                    # level in force during each bar
    per_min = np.repeat(per_bar, 15, axis=1)[:, : FLAT - j]
    adv = L[dd, j:FLAT] if k > 0 else H[dd, j:FLAT]
    hit = (per_min - adv) * k >= 0
    i_rel = first_true(hit)
    stop_at = per_min[np.arange(len(dd)), np.minimum(i_rel, per_min.shape[1] - 1)]
    i_s = np.where(i_rel < BIG, i_rel + j, BIG)
    return settle(dd, k, entry_mid, entry_mid * (1 + k * (hs + slip)), i_s, stop_at, np.full(len(dd), BIG), hs=hs, slip=slip)


def manage_c(dd, je, k, fill, dstop, hs, slip):
    """Bot C from a limit fill at ``fill`` inside minute je: stop first, target = previous minute's session VWAP."""
    stop = fill - k * dstop
    adv = L[dd, je:FLAT] if k > 0 else H[dd, je:FLAT]
    i_s = first_true((stop[:, None] - adv) * k >= 0)
    in_entry_minute = i_s == 0
    i_s = np.where(i_s < BIG, i_s + je, BIG)
    tp = VW[dd, je:FLAT - 1]                                              # resting during minutes je+1 .. FLAT-1
    reach = H[dd, je + 1:FLAT] * (1 - hs) if k > 0 else L[dd, je + 1:FLAT] * (1 + hs)
    hit = (reach - tp) * k >= 0
    i_rel = first_true(hit)
    rows = np.arange(len(dd))
    tp_at = tp[rows, np.minimum(i_rel, tp.shape[1] - 1)]
    o_t = O[dd, np.minimum(i_rel + je + 1, 389)]
    tp_px = np.maximum(tp_at, o_t * (1 - hs)) if k > 0 else np.minimum(tp_at, o_t * (1 + hs))
    i_t = np.where(i_rel < BIG, i_rel + je + 1, BIG)
    x, pnl, gross, why = settle(dd, k, fill * (1 - k * hs), fill, i_s, stop, np.full(len(dd), BIG), i_t, tp_px, hs, slip)
    # a stop inside the entry minute is filled at the stop, never at that minute's open
    pnl = np.where(in_entry_minute, (stop * (1 - k * (hs + slip)) - fill) * k, pnl)
    gross = np.where(in_entry_minute, (stop - fill * (1 - k * hs)) * k, gross)
    return x, pnl, gross, why


def bracket_all_days(je, k, stop_pct, tp_pct, hs, slip, fee):
    """Bot C benchmark: the trade's bracket (limit-style entry at the minute's open, same stop and target
    distance in % of price) on every session."""
    e_mid = O[:, je]
    fill = e_mid * (1 + k * hs)
    stop, tp = fill * (1 - k * stop_pct), fill * (1 + k * tp_pct)
    adv = L[:, je:FLAT] if k > 0 else H[:, je:FLAT]
    reach = H[:, je:FLAT] * (1 - hs) if k > 0 else L[:, je:FLAT] * (1 + hs)
    i_s = first_true((stop[:, None] - adv) * k >= 0)
    i_t = first_true((reach - tp[:, None]) * k >= 0)
    rows = np.arange(D)
    o_s = O[rows, np.minimum(i_s + je, 389)]
    stop_fill = np.where((stop - o_s) * k >= 0, o_s, stop) * (1 - k * (hs + slip))
    ex = np.where((i_s <= i_t) & (i_s < BIG), stop_fill, np.where(i_t < BIG, tp, O[:, FLAT] * (1 - k * (hs + slip))))
    pts = (ex - fill) * k
    return (pts - fee * np.abs(pts)) / (stop_pct * fill)


def size(risk_eur, d, entry, dstop, step):
    eff = step * P_REF / entry                                             # the lot is pinned to today's index level
    u = np.floor(risk_eur * FX[d] / dstop / eff + 1e-9) * eff
    return u if u >= eff - 1e-12 else 0.0


def run(bot, b, risk_eur, mode="both"):
    hs, slip, fee, step = b["hs"], b["slip"], b["fee"], b["step"]
    T, skip_lot, skip_small = [], 0, 0
    for d in range(D):
        free, n = 0, 0                                                    # first minute at which the bot is flat again
        dd = np.array([d])

        def take(j, k, entry, dstop, fn, key, fill=None):
            nonlocal free, n, skip_lot
            u = size(risk_eur, d, entry, dstop, step)
            if u == 0.0:
                skip_lot += 1
                return
            x, pnl, gross, why = fn(dd, key, k, np.array([entry if fill is None else fill]), np.array([dstop]), hs, slip)
            pnl, gross = float(pnl[0]), float(gross[0])
            net = pnl - fee * abs(pnl)
            T.append(dict(d=d, j=j, k=k, key=key, entry=entry, dstop=dstop, units=u, gross_r=gross / dstop, net_r=net / dstop,
                          eur=u * net / FX[d], x=int(x[0]), why=str(why[0])))
            free, n = int(x[0]), n + 1

        if bot == "A":
            if not (np.isfinite(SIG[d, 0]) and np.isfinite(PC[d]) and np.isfinite(ATR30[d, 0])):
                continue
            for i in range(10):                                           # closes 10:00 .. 14:30
                cm = 29 + 30 * i
                if n >= 2:
                    break
                if free > cm:
                    continue
                c = C[d, cm]
                k = 1 if c > UP[d, i] else (-1 if c < LO[d, i] else 0)
                if k == 0 or (mode == "trend" and k != trend[d]):
                    continue
                entry = O[d, cm + 1]
                dstop = max(abs(entry - VW[d, cm]), 1.2 * ATR30[d, i])
                if dstop / entry < 0.0020:
                    skip_small += 1
                    continue
                take(cm + 1, k, entry, dstop, manage_a, i)
        elif bot == "B":
            if not np.isfinite(Q25[d, 6]):
                continue
            for bb in range(6, 19):                                       # bars closing 11:15 .. 14:15
                ce = 15 * bb + 14
                if n >= 2:
                    break
                if free > ce:
                    continue
                if not (R90[d, bb] <= Q25[d, bb] and V15[d, bb] >= 1.3 * MEDV[d, bb]):
                    continue
                k = 1 if C15[d, bb] > PH[d, bb] else (-1 if C15[d, bb] < PL[d, bb] else 0)
                if k == 0:
                    continue
                entry = O[d, ce + 1]
                dstop = max(R90[d, bb], 1.5 * ATR15[d, bb])
                if dstop / entry < 0.0020:
                    skip_small += 1
                    continue
                take(ce + 1, k, entry, dstop, manage_b, bb)
        else:
            if EVENT_DAY[d]:
                continue
            t = np.arange(44, 270)                                        # information at 10:15 .. 14:00
            sig = C[d, t] * np.sqrt(BV[d, t])
            gate = (BV[d, t] > 0) & (RV[d, t] / np.maximum(BV[d, t], 1e-12) <= 1.8)
            lb, ls = VW[d, t] - 2.2 * sig, VW[d, t] + 2.2 * sig
            buy = L[d, t + 1] * (1 + hs) < lb
            sell = H[d, t + 1] * (1 - hs) > ls
            for q in np.flatnonzero(gate & (buy ^ sell)):
                if n >= 2:
                    break
                tt = int(t[q])
                if free > tt:
                    continue
                k = 1 if buy[q] else -1
                fill = min(lb[q], O[d, tt + 1] * (1 + hs)) if k > 0 else max(ls[q], O[d, tt + 1] * (1 - hs))
                dstop = 1.5 * sig[q]
                if dstop / fill < 0.0020:
                    skip_small += 1
                    continue
                before = n
                take(tt + 1, k, fill, dstop, manage_c, tt + 1, fill=fill)
                if n > before:
                    T[-1]["tp_pct"] = abs(VW[d, tt] - fill) / fill
    return pd.DataFrame(T), skip_lot, skip_small


BENCH: dict = {}


def bench_row(bot, bname, b, key, k):
    """Net R of a same-minute, same-direction entry managed by the bot's own exit rules on EVERY session."""
    ck = (bot, bname, key, k)
    if ck not in BENCH:
        hs, slip, fee = b["hs"], b["slip"], b["fee"]
        dd = np.arange(D)
        if bot == "A":
            cm = 29 + 30 * key
            entry = O[:, cm + 1]
            dstop = np.maximum(np.abs(entry - VW[:, cm]), 1.2 * ATR30[:, key])
            valid = np.isfinite(SIG[:, 0]) & np.isfinite(PC) & np.isfinite(dstop) & (dstop / entry >= 0.0020)
            fn = manage_a
        else:
            entry = O[:, 15 * key + 15]
            dstop = np.maximum(R90[:, key], 1.5 * ATR15[:, key])
            valid = np.isfinite(Q25[:, 6]) & np.isfinite(dstop) & (dstop / entry >= 0.0020)
            fn = manage_b
        ds_ = np.where(valid, dstop, 1.0)
        _, pnl, _, _ = fn(dd, key, k, entry, ds_, hs, slip)
        BENCH[ck] = np.where(valid, (pnl - fee * np.abs(pnl)) / ds_, np.nan)
    return BENCH[ck]


rng = np.random.default_rng(20201231)
rows = []
for label, bot, mode in (("A noise-area momentum, long and short", "A", "both"), ("A noise-area momentum, trend-aligned", "A", "trend"),
                         ("B squeeze and volume expansion", "B", "both"), ("C jump-filtered VWAP limit fade", "C", "both")):
    for bname, b in BROKERS.items():
        for risk in (25.00, 12.50):
            t, skip_lot, skip_small = run(bot, b, risk, mode)
            n = len(t)
            if n == 0:
                rows.append({"Bot": label, "Account": f"{bname}, EUR {risk:.2f}", "Trades": 0, "Skipped (min lot)": skip_lot})
                continue
            nr = t.net_r.to_numpy()
            daily = np.zeros(D)
            np.add.at(daily, t.d.to_numpy(), t.eur.to_numpy() / 5000.0)
            if bot == "C":
                mat = [bracket_all_days(int(s.j), int(s.k), s.dstop / s.entry, s.tp_pct, b["hs"], b["slip"], b["fee"]) for s in t.itertuples()]
            else:
                mat = [bench_row(bot, bname, b, int(s.key), int(s.k)) for s in t.itertuples()]
            mat = np.array(mat)
            draws = np.empty(4000)
            valid_idx = [np.flatnonzero(np.isfinite(x)) for x in mat]
            pick = np.array([vi[rng.integers(0, len(vi), 4000)] for vi in valid_idx])            # (n, 4000)
            draws = mat[np.arange(n)[:, None], pick].mean(0)
            w, l = nr[nr > 0], nr[nr <= 0]
            lg, sh = t[t.k > 0], t[t.k < 0]
            rows.append({
                "Bot": label, "Account": f"{bname}, EUR {risk:.2f}", "Trades": n, "Skipped (min lot)": skip_lot,
                "Skipped (stop < 0.20%)": skip_small, "Win rate": (nr > 0).mean(),
                "Payoff": w.mean() / abs(l.mean()) if len(w) and len(l) else np.nan,
                "Gross exp (R)": t.gross_r.mean(), "Net exp (R)": nr.mean(),
                "Net Sharpe": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan,
                "t": nr.mean() / nr.std(ddof=1) * np.sqrt(n), "Long n": len(lg), "Long net R": lg.net_r.mean() if len(lg) else np.nan,
                "Short n": len(sh), "Short net R": sh.net_r.mean() if len(sh) else np.nan,
                "Random net R": float(np.nanmean(mat)), "Share random >= actual": float((draws >= nr.mean()).mean()),
                "Net EUR": t.eur.sum(), "Days traded": t.d.nunique(), "Mean hold (min)": (t.x - t.j).mean(),
                "Mean stop (% of price)": 100 * (t.dstop / t.entry).mean(), "exits": t.why.value_counts().to_dict(),
                "by_year": t.groupby(full[t.d.to_numpy()].year).net_r.agg(["size", "mean"]).round(3).to_dict(orient="index")})
            t.assign(date=full[t.d.to_numpy()]).to_csv(f"{OUT}/trades_{bot}_{mode}_{bname}_{risk:.0f}.csv", index=False)
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/bots_2015_2020.csv", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_colwidth", 110)
print(res.drop(columns=["exits", "by_year"]).round(3).to_string(index=False))
print(res[["Bot", "Account", "exits"]].iloc[::2].to_string(index=False))
print(res[["Bot", "Account", "by_year"]].iloc[::4].to_string(index=False))
