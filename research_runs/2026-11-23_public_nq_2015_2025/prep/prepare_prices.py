"""Cash-equivalent NASDAQ-100 1-minute bars from a continuous NQ futures tape.

An index CFD is quoted off the future less its fair-value basis.  The basis
is observed once a day (futures at the 16:00 ET close minus the official
index close from FRED), is close to linear in time within one contract and
jumps when the tape rolls to the next contract.  Within each contract segment
a robust line (Theil-Sen) is fitted to the daily basis and subtracted from
every bar of that segment.  This removes the roll jumps and the carry that is
embedded in futures prices, and leaves the level equal to the cash index.

Usage: prepare_prices.py <dir with NQ_1min_*.parquet> <out dir>
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from ndxswing.data.ingest import fetch_fred_csv

NY = "America/New_York"
src, out = sys.argv[1], sys.argv[2]

fut = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(src + "/NQ_1min_*.parquet"))])
fut = fut.set_index("timestamp").sort_index()
fut = fut[~fut.index.duplicated(keep="last")]
fut.index = fut.index.tz_localize("UTC")                     # confirmed by the intraday volume profile
loc = fut.index.tz_convert(NY)
minute = loc.hour * 60 + loc.minute
day = pd.Series(loc.tz_localize(None).normalize(), index=fut.index)

# ---- daily basis at the cash close
rth = (minute >= 570) & (minute < 960) & (loc.dayofweek < 5)
close16 = fut["close"][rth].groupby(day[rth]).last()
last_min = pd.Series(minute[rth], index=fut.index[rth]).groupby(day[rth]).last()
ndx = fetch_fred_csv("NASDAQ100", "2014-12-01")
b = pd.DataFrame({"fut": close16, "ndx": ndx, "last_min": last_min}).dropna()
b = b[b.last_min >= 955]                                      # full sessions only
b["basis"] = b.fut - b.ndx

# ---- roll dates.  The tape switches contract early in each quarterly expiry
# week (Monday to Thursday).  The switch shows up as a jump in the daily basis;
# the first session on the new contract is the day with the largest jump in
# that window.  When carry is tiny the jump drowns in noise and the choice of
# day within the week is immaterial (the basis is then within a few points of zero).
dbasis = b.basis.diff()
expiries = [d for d in pd.date_range("2015-01-01", "2025-12-31", freq="WOM-3FRI") if d.month in (3, 6, 9, 12)]
rolls = []
for exp in expiries:
    if exp > b.index[-1]:
        break
    week = dbasis[(dbasis.index >= exp - pd.Timedelta(days=4)) & (dbasis.index <= exp - pd.Timedelta(days=1))]
    first_session = week.abs().idxmax()
    rolls.append({"expiry": exp.date(), "first_session": first_session, "basis_jump": round(float(week[first_session]), 1),
                  "second_largest_in_week": round(float(week.drop(first_session).abs().max()), 1)})
rolls = pd.DataFrame(rolls)
print(rolls.to_string())

# ---- fit the basis within each segment and subtract it
bounds = [b.index[0]] + list(rolls.first_session) + [b.index[-1] + pd.Timedelta(days=1)]
fit = pd.Series(np.nan, index=b.index)
seg_rows = []
for k in range(len(bounds) - 1):
    seg = b[(b.index >= bounds[k]) & (b.index < bounds[k + 1])]
    if len(seg) < 5:
        continue
    x = (seg.index - seg.index[0]).days.to_numpy(float)
    slope, intercept, _, _ = theilslopes(seg.basis.to_numpy(), x)
    fit.loc[seg.index] = intercept + slope * x
    seg_rows.append({"from": seg.index[0].date(), "to": seg.index[-1].date(), "days": len(seg),
                     "start": round(intercept, 1), "end": round(intercept + slope * x[-1], 1),
                     "resid_sd": round(float((seg.basis - fit.loc[seg.index]).std()), 1)})
segs = pd.DataFrame(seg_rows)
print(segs.to_string())

all_days = pd.DatetimeIndex(sorted(day[rth].unique()))
fit_all = fit.reindex(all_days).interpolate(limit_area="inside").ffill().bfill()   # short sessions take the neighbouring value
# never interpolate across a roll: re-anchor days on the segment they belong to
for r in rolls.first_session:
    prev = fit.index[fit.index < r]
    nxt = fit.index[fit.index >= r]
    if len(prev) and len(nxt):
        gap_days = all_days[(all_days > prev[-1]) & (all_days < nxt[0])]
        fit_all.loc[gap_days[gap_days < r]] = fit.loc[prev[-1]]
        fit_all.loc[gap_days[gap_days >= r]] = fit.loc[nxt[0]]

cash = fut[rth].copy()
adj = day[rth].map(fit_all).to_numpy()
for c in ("open", "high", "low", "close"):
    cash[c] = cash[c] - adj
cash["volume"] = cash["volume"].astype(float)

# ---- validation against the official close
c16 = cash["close"].groupby(day[rth]).last()
v = pd.DataFrame({"cash": c16, "ndx": ndx}).dropna()
v = v[v.index.isin(b.index)]
err = v.cash - v.ndx
rc, ri = np.log(v.cash).diff(), np.log(v.ndx).diff()
print(f"days {len(v)} | return correlation {rc.corr(ri):.5f} | level error pts: median {err.median():.2f}, sd {err.std():.2f}, "
      f"max abs {err.abs().max():.1f} | as % of index: sd {(err / v.ndx).std() * 100:.3f}, max {(err / v.ndx).abs().max() * 100:.2f}")
on_roll = (rc - ri).reindex(rolls.first_session).abs() * 1e4
print(f"return gap on roll days, bp: median {on_roll.median():.1f}, max {on_roll.max():.1f}  (all days: sd {(rc - ri).std() * 1e4:.1f})")
print("worst 6 level errors:", err.abs().nlargest(6).round(1).to_dict())

cash.index = cash.index.rename("ts")
cash.reset_index().to_parquet(out + "/ndx_cash_equiv_1m.parquet")
rolls.to_csv(out + "/rolls.csv", index=False)
segs.to_csv(out + "/basis_segments.csv", index=False)
print("rows written", len(cash), cash.index[0], "->", cash.index[-1])
