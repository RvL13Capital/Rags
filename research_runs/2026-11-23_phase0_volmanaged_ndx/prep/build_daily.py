"""Daily variance decomposition of the NASDAQ-100 from 1-minute regular-session bars.

For every session of the store: open, close, overnight gap, within-session
return and the realised measures on 1-minute and 5-minute log returns.  Also
checks the conservation identity

    sum(1-minute log returns of the session) + overnight gap = close-to-close log return

against the hourly bars the backtests used, and writes the macro inputs.

Usage: build_daily.py <ndx_cash_equiv_1m.parquet> <market.duckdb> <out dir>
"""
import sys

import numpy as np
import pandas as pd

from ndxswing.data.ingest import fetch_fred_csv
from ndxswing.data.store import MarketDataStore
from ndxswing.features.indicators import prepare_hourly_bars
from ndxswing.research.volmanaged import realised_measures

src, db, out = sys.argv[1], sys.argv[2], sys.argv[3]
NY = "America/New_York"

m = pd.read_parquet(src).set_index("ts").sort_index()
m.index = pd.DatetimeIndex(m.index).tz_convert(NY)
with MarketDataStore(db, read_only=True) as store:
    hourly = prepare_hourly_bars(store.load_hourly("NDX"))
    usd_rate, eurusd = store.load_series("USD_RATE"), store.load_series("EURUSD")
h = hourly.copy()
h.index = h.index.tz_convert(NY)
sess = h.groupby("session").agg(first=("open", lambda s: s.index[0]), last=("close", lambda s: s.index[-1]),
                                h_open=("open", "first"), h_close=("close", "last"), bars=("close", "size"))
# the last hourly bar of a session starts at 15:30 (30 minutes) or, on an early close, one hour before 13:00
day = m.index.tz_localize(None).normalize()
rows = []
for d, g in m.groupby(day):
    if d not in sess.index:
        continue                                  # exchange holiday on which the futures traded
    s = sess.loc[d]
    end = s["last"] + pd.Timedelta(minutes=60 if (s["last"].hour, s["last"].minute) != (15, 30) else 30)
    g = g[(g.index >= s["first"]) & (g.index < end)]
    rm = realised_measures(g["open"].to_numpy(), g["close"].to_numpy())
    rows.append({"date": d, "open": g["open"].iloc[0], "close": g["close"].iloc[-1], "n_min": len(g),
                 "h_open": s["h_open"], "h_close": s["h_close"], **rm})
d = pd.DataFrame(rows).set_index("date")
d["r_cc"] = np.log(d["close"]).diff()
d["r_gap"] = np.log(d["open"] / d["close"].shift())

# ---- conservation identity and agreement with the bars the backtests used
resid = (d["sum_r1"] + d["r_gap"] - d["r_cc"]).dropna()
print(f"sessions {len(d)} ({d.index[0].date()} .. {d.index[-1].date()}), store sessions {len(sess)}")
print(f"identity sum(r_1m) + r_gap - r_cc: max |residual| {resid.abs().max():.3e} over {len(resid)} sessions")
print(f"1-minute session open/close against hourly store: max |diff| open {np.abs(d.open - d.h_open).max():.3e}, "
      f"close {np.abs(d.close - d.h_close).max():.3e}")
print(f"minutes per session: median {d.n_min.median():.0f}, min {d.n_min.min()}, sessions under 380: {(d.n_min < 380).sum()}")

# ---- macro inputs on the session calendar (last known value on or before the session date)
ndx, xndx = fetch_fred_csv("NASDAQ100", "2014-12-01"), fetch_fred_csv("NASDAQXNDX", "2014-12-01")
tr = pd.DataFrame({"p": ndx, "t": xndx}).dropna()
div = (np.log(tr.t).diff() - np.log(tr.p).diff()).rename("div_yield_day")     # gross dividend return of the day
estr = fetch_fred_csv("ECBESTRVOLWGTTRMDMNRT", "2014-12-01").dropna() / 100.0
eonia = fetch_fred_csv("EONIARATE", "2014-12-01").dropna() / 100.0 - 0.00085
estr_full = pd.concat([eonia[eonia.index < estr.index[0]], estr]).sort_index()
mac = pd.DataFrame(index=d.index)
mac["usd_rate"] = usd_rate.reindex(usd_rate.index.union(d.index)).ffill().reindex(d.index)
mac["eurusd"] = eurusd.reindex(eurusd.index.union(d.index)).ffill().reindex(d.index)
mac["estr"] = estr_full.reindex(estr_full.index.union(d.index)).ffill().reindex(d.index)
mac["div_yield_day"] = div.reindex(d.index).fillna(0.0)
mac["ndx_official_close"] = ndx.reindex(d.index)
print(f"gross dividend yield, annualised by year:\n{(mac.div_yield_day.groupby(mac.index.year).sum() * 100).round(2).to_dict()}")
cols = ["open", "close", "n_min", "r_cc", "r_gap", "sum_r1", "rv1", "rv5", "bv5", "jv5", "rsm5", "rsp5"]
d[cols].round(10).to_csv(f"{out}/ndx_daily_variance.csv")
mac.round(8).to_csv(f"{out}/macro_daily.csv")
