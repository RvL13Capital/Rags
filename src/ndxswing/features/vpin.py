"""Volume-clock VPIN with bulk volume classification (section 4.D).

Volume is partitioned into fixed-size buckets in *volume time*.  The bucket
size for a session is ``median(daily volume of the previous `median_days`
sessions) / buckets_per_day`` - strictly prior sessions, never a full-sample
statistic and never a clock-time rolling window.  Each bar's volume is split
into buy and sell volume with the Student-t CDF of its standardised price
change and distributed proportionally over the buckets it fills.

Caveat (Andersen & Bondarenko 2015): BVC-VPIN built from hourly bars is
mechanically related to volatility and volume.  The filter is retained only if
the ablation shows out-of-sample lift after spread and GARCH volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .indicators import trailing_pct_rank


def compute_volume_time_vpin(
    hourly: pd.DataFrame,
    n_buckets: int = 50,
    buckets_per_day: int = 50,
    median_days: int = 20,
    sigma_window: int = 140,
    nu: float = 4.0,
) -> pd.Series:
    """VPIN known at each bar close (NaN until ``n_buckets`` buckets completed)."""
    close = hourly["close"].to_numpy(dtype=float)
    vol = hourly["volume"].to_numpy(dtype=float)
    sess = hourly["session"]
    dp = pd.Series(np.diff(close, prepend=close[0]), index=hourly.index)
    sig = dp.rolling(sigma_window, min_periods=sigma_window // 2).std().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        buy_frac = student_t.cdf(dp.to_numpy() / sig, df=nu)
    buy_frac = np.where(np.isfinite(buy_frac), buy_frac, 0.5)

    daily_vol = hourly.groupby("session")["volume"].sum()
    bucket_size_d = daily_vol.rolling(median_days, min_periods=median_days).median().shift(1) / buckets_per_day
    vb = sess.map(bucket_size_d).to_numpy(dtype=float)

    out = np.full(len(close), np.nan)
    imb: list[float] = []   # |buy - sell| of completed buckets
    size: list[float] = []  # their volumes
    cur_b = cur_s = 0.0
    cur_cap = np.nan
    for i in range(len(close)):
        if not np.isfinite(vb[i]) or vb[i] <= 0:
            continue
        if not np.isfinite(cur_cap):
            cur_cap = vb[i]
        rem, bf = vol[i], buy_frac[i]
        while rem > 0:
            room = cur_cap - (cur_b + cur_s)
            take = min(rem, room)
            cur_b += take * bf
            cur_s += take * (1.0 - bf)
            rem -= take
            if cur_b + cur_s >= cur_cap - 1e-9:
                imb.append(abs(cur_b - cur_s))
                size.append(cur_cap)
                cur_b = cur_s = 0.0
                cur_cap = vb[i]  # next bucket uses the size in force now
        if len(imb) >= n_buckets:
            out[i] = sum(imb[-n_buckets:]) / sum(size[-n_buckets:])
    return pd.Series(out, index=hourly.index, name="vpin")


def vpin_percentile(vpin: pd.Series, bars_per_year: int = 252 * 7) -> pd.Series:
    """Causal empirical CDF value of VPIN versus its trailing ~252 sessions."""
    return trailing_pct_rank(vpin, bars_per_year, min_periods=bars_per_year // 4).rename("vpin_pct")
