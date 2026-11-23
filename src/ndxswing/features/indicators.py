"""Bar preparation and strictly causal indicators.

Conventions
-----------
* Hourly bars: ``DatetimeIndex`` in UTC (tz-aware), columns ``open, high, low,
  close, volume`` and optionally ``spread_rel`` (quoted spread / mid).  The
  timestamp is the bar *start*.
* ``session`` is the New York calendar date of the bar (naive, midnight).
* Every function returns a value for row ``t`` that depends on rows ``<= t``
  only.  Anything that must exclude the current bar is shifted explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"
RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60
OHLCV = ["open", "high", "low", "close", "volume"]


def prepare_hourly_bars(df: pd.DataFrame, rth_only: bool = True) -> pd.DataFrame:
    """Validate, sort and annotate hourly bars with their NY session date."""
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"hourly bars are missing columns {missing}")
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx.rename("ts")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    local = out.index.tz_convert(NY)
    minute = np.asarray(local.hour) * 60 + np.asarray(local.minute)
    if rth_only:
        keep = (minute >= RTH_OPEN_MIN) & (minute < RTH_CLOSE_MIN) & (np.asarray(local.dayofweek) < 5)
        out, local = out[keep], local[keep]
    out["session"] = local.tz_localize(None).normalize()
    bad = (out["high"] < out[["open", "close", "low"]].max(axis=1) - 1e-9) | (
        out["low"] > out[["open", "close", "high"]].min(axis=1) + 1e-9
    ) | (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} hourly bars violate OHLC consistency, first at {out.index[bad][0]}")
    if out[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("hourly bars contain NaN prices")
    return out


def daily_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Regular-session daily bars indexed by session date."""
    g = hourly.groupby("session", sort=True)
    daily = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
            "n_bars": g["close"].size(),
        }
    )
    daily.index.name = "session"
    return daily


def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Wilder ATR (includes the overnight gap in the first bar of a session)."""
    return true_range(high, low, close).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def donchian_prev(high: pd.Series, low: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Highest high / lowest low of the ``window`` bars *before* the current one."""
    hi = high.rolling(window, min_periods=window).max().shift(1)
    lo = low.rolling(window, min_periods=window).min().shift(1)
    return hi, lo


def realized_vol(logret: pd.Series, window: int, periods: int = 252) -> pd.Series:
    """sqrt(periods * mean(r^2)) over the trailing window (zero-mean estimator)."""
    return np.sqrt(periods * (logret**2).rolling(window, min_periods=window).mean())


def trailing_pct_rank(x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Empirical CDF value of x_t within the trailing ``window`` *prior* values.

    The current observation is compared against the previous ``window``
    observations only, so the rank at t never uses information after t.
    """
    v = x.to_numpy(dtype=float)
    n = len(v)
    mp = window // 2 if min_periods is None else min_periods
    out = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(v[t]):
            continue
        ref = v[max(0, t - window):t]
        ref = ref[~np.isnan(ref)]
        if len(ref) >= mp:
            out[t] = np.mean(ref <= v[t])
    return pd.Series(out, index=x.index)


def regime_observations(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily observation vector of section 4.A, each row closed at day t.

    X_t = [r_t, RV20_t, Slope20_t, VC_t, Gap_t, Vtilde20_t]
    """
    c, o, v = daily["close"], daily["open"], daily["volume"].astype(float)
    r = np.log(c / c.shift(1))
    rv5 = realized_vol(r, 5)
    rv20 = realized_vol(r, 20)
    ema20 = ema(c, 20)
    atr20 = atr(daily["high"], daily["low"], c, 20)
    slope = (ema20 - ema20.shift(5)) / (5.0 * atr20)
    med_v = v.rolling(20, min_periods=20).median()
    vt = np.log(v.where(v > 0) / med_v.where(med_v > 0))
    return pd.DataFrame(
        {
            "r": r,
            "rv20": rv20,
            "slope20": slope,
            "vc": rv5 / rv20.where(rv20 > 0),
            "gap": np.log(o / c.shift(1)).abs(),
            "vtilde20": vt.fillna(0.0) if (v > 0).sum() == 0 else vt,
            "rv5": rv5,
            "ema20": ema20,
            "ema50": ema(c, 50),
            "atr20": atr20,
        },
        index=daily.index,
    )
