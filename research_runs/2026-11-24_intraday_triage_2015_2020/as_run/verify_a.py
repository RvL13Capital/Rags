import sys, numpy as np, pandas as pd
src = open(sys.argv[6]).read().split("rng = np.random.default_rng(20201231)")[0]
exec(src)
b = BROKERS["Pepperstone"]; hs, slip = b["hs"], b["slip"]
t, _, _ = run("A", b, 25.0, "both")

# ---- (a) independent minute-by-minute re-implementation of Bot A, written without the vectorised helpers
def slow_a(risk=25.0):
    out = []
    for d in range(D):
        if d < 14 or not np.isfinite(PC[d]) or not np.isfinite(ATR30[d, 0]):
            continue
        prev = [dd for dd in range(d - 14, d)]
        n, pos_, minute = 0, None, 29
        while minute < 390 and (pos_ is not None or n < 2):
            if pos_ is not None:
                k, stop, fill, dstop, u, j0 = pos_
                o, h, l = O[d, minute], H[d, minute], L[d, minute]
                exit_px = None
                if minute == FLAT:
                    exit_px = o * (1 - k * (hs + slip))
                elif pos_exit_signal:
                    exit_px = o * (1 - k * (hs + slip))
                elif (k > 0 and o <= stop) or (k < 0 and o >= stop):
                    exit_px = o * (1 - k * (hs + slip))
                elif (k > 0 and l <= stop) or (k < 0 and h >= stop):
                    exit_px = stop * (1 - k * (hs + slip))
                if exit_px is not None:
                    out.append((d, j0, k, (exit_px - fill) * k / dstop, minute))
                    pos_, n = None, n + 1
                    pos_exit_signal = False
                    # the exit minute is over for entries: a new entry needs a later 30-minute close
            if (minute + 1) % 30 == 0 and minute < 389:                       # a 30-minute bar closes at the end of this minute
                i = (minute + 1) // 30 - 1
                sig = np.mean([abs(C[p, minute] / O[p, 0] - 1) for p in prev])
                up = max(O[d, 0], PC[d]) * (1 + sig); lo = min(O[d, 0], PC[d]) * (1 - sig)
                tp = (H[d, : minute + 1] + L[d, : minute + 1] + C[d, : minute + 1]) / 3
                vol = V[d, : minute + 1] if V[d].sum() > 0 else np.ones(minute + 1)
                vw = (tp * vol).sum() / vol.sum() if vol.sum() > 0 else tp.mean()
                c = C[d, minute]
                if pos_ is not None:
                    k = pos_[0]
                    pos_exit_signal = (c < max(up, vw)) if k > 0 else (c > min(lo, vw))
                elif n < 2 and 29 <= minute <= 299:
                    k = 1 if c > up else (-1 if c < lo else 0)
                    if k:
                        entry = O[d, minute + 1]
                        dstop = max(abs(entry - vw), 1.2 * ATR30[d, i])
                        if dstop / entry >= 0.0020:
                            eff = 0.10 * P_REF / entry
                            u = np.floor(risk * FX[d] / dstop / eff + 1e-9) * eff
                            if u >= eff - 1e-12:
                                pos_ = (k, entry - k * dstop, entry * (1 + k * (hs + slip)), dstop, u, minute + 1)
                                pos_exit_signal = False
            minute += 1
    return pd.DataFrame(out, columns=["d", "j", "k", "net_r", "x"])

s = slow_a()
mrg = t.merge(s, on=["d", "j", "k"], how="outer", suffixes=("", "_slow"), indicator=True)
print("vectorised trades", len(t), "independent re-implementation", len(s), "matched", int((mrg._merge == "both").sum()),
      "max |net R difference| %.2e" % np.nanmax(np.abs(mrg.net_r - mrg.net_r_slow)), "exit minute mismatches", int((mrg.x != mrg.x_slow).sum()))
print("means: vectorised %.5f  independent %.5f" % (t.net_r.mean(), s.net_r.mean()))

# ---- (b) stricter benchmark: random session, same entry minute, direction, stop distance (%) and EXIT MINUTE as the actual trade
rng = np.random.default_rng(1)
mat = np.empty((len(t), D))
for q, r in enumerate(t.itertuples()):
    j, k, x = int(r.j), int(r.k), int(r.x)
    e = O[:, j]; sp = r.dstop / r.entry; stop = e * (1 - k * sp)
    xe = max(x, j + 1)
    adv = L[:, j:xe] if k > 0 else H[:, j:xe]
    hit = (stop[:, None] - adv) * k >= 0
    i_s = np.where(hit.any(1), hit.argmax(1), BIG)
    o_s = O[np.arange(D), np.minimum(i_s + j, 389)]
    stop_fill = np.where((stop - o_s) * k >= 0, o_s, stop) * (1 - k * (hs + slip))
    ex = np.where(i_s < BIG, stop_fill, O[:, min(xe, FLAT)] * (1 - k * (hs + slip)))
    mat[q] = (ex - e * (1 + k * (hs + slip))) * k / (sp * e)
draws = mat[np.arange(len(t))[:, None], rng.integers(0, D, (len(t), 4000))].mean(0)
print("time-matched random benchmark: mean net R %.4f ; share of random draws >= actual %.4f ; actual %.4f" % (mat.mean(), (draws >= t.net_r.mean()).mean(), t.net_r.mean()))
# same-day placebo: same days, same minute, OPPOSITE direction held to the same exit minute (is it the day or the direction?)
print("long/short actual: ", t.groupby("k").net_r.agg(["size", "mean"]).round(4).to_dict(orient="index"))

# ---- (c) the same trades under the earlier fixed-point cost convention (1.0 pt spread + 0.5 pt slippage whatever the index level)
rows = []
for name, hp, sp_, fee in (("Pepperstone fixed points", 0.5, 0.5, 0.0), ("Capital.com fixed points", 0.9, 0.5, 0.007)):
    res = []
    for r in t.itertuples():
        d = int(r.d); p = r.entry
        _, pnl, gross, _ = manage_a(np.array([d]), int(r.key), int(r.k), np.array([r.entry]), np.array([r.dstop]), hp / p, sp_ / p)
        pnl = float(pnl[0]); res.append((pnl - fee * abs(pnl)) / r.dstop)
    res = np.array(res)
    print(name, "net exp R %.4f t %.2f (same %d trades; cost per trade in R %.4f)" % (res.mean(), res.mean() / res.std(ddof=1) * np.sqrt(len(res)), len(res), t.gross_r.mean() - res.mean()))
print("bps convention: gross %.4f net %.4f ; mean index level at entry %.0f ; cost in bps per round trip %.2f" % (t.gross_r.mean(), t.net_r.mean(), t.entry.mean(), 1e4 * 2 * (hs + slip)))
# concentration and shape
nr = t.net_r.to_numpy()
print("top 5%% of trades contribute %.0f%% of total R; without the best 10 trades mean = %.4f; median trade %.3f; skew %.2f" % (
    100 * np.sort(nr)[-len(nr) // 20:].sum() / nr.sum(), np.sort(nr)[:-10].mean(), np.median(nr), pd.Series(nr).skew()))
half = len(t) // 2
print("first half mean %.4f (t %.2f), second half mean %.4f (t %.2f)" % (nr[:half].mean(), nr[:half].mean() / nr[:half].std(ddof=1) * half**.5, nr[half:].mean(), nr[half:].mean() / nr[half:].std(ddof=1) * (len(nr) - half)**.5))
print("by entry time:", t.groupby("j").net_r.agg(["size", "mean"]).round(3).to_dict(orient="index"))
# daily-level t-stat (trades on the same day are not independent)
dly = t.groupby("d").net_r.sum()
print("per trading day: n %d mean R %.4f t %.2f" % (len(dly), dly.mean(), dly.mean() / dly.std(ddof=1) * np.sqrt(len(dly))))
print("---- debug")
only_v = mrg[mrg._merge == "left_only"].sort_values(["d", "j"]); only_s = mrg[mrg._merge == "right_only"].sort_values(["d", "j"])
print(only_v[["d", "j", "k", "net_r", "x"]].head(8).to_string()); print(only_s[["d", "j", "k", "net_r_slow", "x_slow"]].head(8).to_string())
d0 = int(only_v.d.iloc[0])
print("day", d0, "vectorised:", t[t.d == d0][["j", "k", "x", "net_r", "why"]].to_dict("records"), "slow:", s[s.d == d0].to_dict("records"))
for i in range(12):
    cm = 29 + 30 * i
    print(i, cm, "C %.2f UP %.2f LO %.2f VW %.2f" % (C[d0, cm], UP[d0, i], LO[d0, i], VW[d0, cm]))
