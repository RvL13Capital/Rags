"""Additive (Panama) back-adjustment of a futures chain.

The continuous series is used ONLY for returns and indicators, so that a
contract roll does not print an artificial breakout.  The unadjusted price of
the contract that was actually front at each bar is returned alongside
(``raw_close``) and must be used for notional, margin and unit calculations.
"""

from __future__ import annotations

import pandas as pd


def panama_adjust(chain: pd.DataFrame, roll_days_before_expiry: int = 5) -> pd.DataFrame:
    """Build the back-adjusted continuous series.

    ``chain`` columns: contract, expiry, ts, open, high, low, close, volume
    (raw prices, any bar frequency).  A contract is held until the first bar
    whose date is within ``roll_days_before_expiry`` business days of its
    expiry; the roll gap is ``close_next - close_front`` at the last bar both
    contracts traded before the roll, and it is added to all earlier history.
    The most recent contract is therefore unadjusted.

    Returns a frame indexed by ts with adjusted open/high/low/close, volume,
    ``raw_close``, ``contract`` and the cumulative ``adjustment`` applied.
    """
    need = {"contract", "expiry", "ts", "open", "high", "low", "close", "volume"}
    if not need <= set(chain.columns):
        raise ValueError(f"chain is missing columns {sorted(need - set(chain.columns))}")
    df = chain.copy()
    df["expiry"] = pd.to_datetime(df["expiry"])
    contracts = df.drop_duplicates("contract").sort_values("expiry")[["contract", "expiry"]].to_records(index=False)
    by = {c: g.set_index("ts").sort_index() for c, g in df.groupby("contract")}

    segments: list[pd.DataFrame] = []
    gaps: list[float] = []
    seg_start = None
    for k, (name, expiry) in enumerate(contracts):
        cur = by[name]
        if seg_start is not None:
            cur = cur[cur.index >= seg_start]
        if k == len(contracts) - 1:
            segments.append(cur.assign(contract=name))
            break
        roll_date = (pd.Timestamp(expiry) - pd.offsets.BDay(roll_days_before_expiry)).date()
        dates = pd.Index(cur.index.tz_convert("America/New_York").date if cur.index.tz is not None else cur.index.date)
        hold = cur[dates < roll_date]
        nxt = by[contracts[k + 1][0]]
        common = hold.index.intersection(nxt.index)
        if len(hold) == 0 or len(common) == 0:
            raise ValueError(f"contracts {name} and {contracts[k + 1][0]} share no bar before the roll date {roll_date}")
        t_last = common[-1]
        gaps.append(float(nxt.loc[t_last, "close"] - hold.loc[t_last, "close"]))
        segments.append(hold.assign(contract=name))
        seg_start = nxt.index[nxt.index > hold.index[-1]][0] if (nxt.index > hold.index[-1]).any() else None
        if seg_start is None:
            raise ValueError(f"contract {contracts[k + 1][0]} has no bars after the roll out of {name}")

    out = []
    adj = 0.0
    for seg, gap in zip(reversed(segments), [0.0] + list(reversed(gaps))):
        adj += gap
        s = seg.copy()
        s["raw_close"] = s["close"]
        for col in ("open", "high", "low", "close"):
            s[col] = s[col] + adj
        s["adjustment"] = adj
        out.append(s)
    res = pd.concat(reversed(out)).sort_index()
    return res[["open", "high", "low", "close", "volume", "raw_close", "contract", "adjustment"]]
