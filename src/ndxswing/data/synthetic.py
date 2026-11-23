"""Synthetic market generator - PIPELINE VERIFICATION ONLY.

Nothing produced here is market data and no statistic computed on it says
anything about the NASDAQ-100.  It exists to (a) exercise every code path of
the backtest / validation stack end to end and (b) run two falsification
checks: with ``edge=0`` the price is a stochastic-volatility martingale and
the release gates must FAIL; with ``edge>0`` a persistent trend component is
planted and the pipeline must be able to see it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from ..features.indicators import NY

BAR_STARTS_MIN = (570, 600, 660, 720, 780, 840, 900)  # 09:30, 10:00, 11:00 ... 15:00 ET
BAR_HOURS = np.array([0.5, 1, 1, 1, 1, 1, 1])
U_SHAPE = np.array([1.8, 1.3, 0.9, 0.75, 0.75, 0.9, 1.4])
SYNTHETIC_SOURCE = "SYNTHETIC-PIPELINE-VERIFICATION"


@dataclass(frozen=True)
class SyntheticSpec:
    start: str = "2012-01-03"
    end: str = "2026-10-30"
    seed: int = 42
    edge: float = 1.0          # 0 = martingale null, 1 = planted trend persistence
    s0: float = 2300.0
    with_spread: bool = True
    #: Intrabar path resolution.  The simulator fills stops and triggers at the
    #: level when a bar's high/low touches it, which is exact for a continuous
    #: path.  On a coarse path the price has already jumped past the level at
    #: the first step that crosses it, so touch fills are too good.  Measured
    #: on no-edge series against a true cost of about 0.08R per trade: 8 steps
    #: per bar flattered expectancy by about 0.08R, 64 steps by about 0.05R,
    #: 256 steps by about 0.015R (inside the noise).  Real bars are built from
    #: ticks and do not have this problem.
    substeps: int = 256


def _sessions(start: str, end: str) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    hol = USFederalHolidayCalendar().holidays(start, end)
    return days.difference(hol)


def generate_synthetic_market(spec: SyntheticSpec = SyntheticSpec()) -> dict[str, object]:
    """Returns dict with ``hourly`` (UTC-indexed OHLCV [+ spread_rel]),
    ``fx`` (EURUSD, USD per EUR), ``rate`` (annual benchmark rate), ``events``
    (UTC timestamps) and ``truth`` (daily latent regime, for tests only)."""
    rng = np.random.default_rng(spec.seed)
    sessions = _sessions(spec.start, spec.end)
    D, K = len(sessions), len(BAR_STARTS_MIN)

    # --- latent daily regime: 0 trend, 1 sideways, 2 stress
    P = np.array([[0.975, 0.020, 0.005], [0.030, 0.960, 0.010], [0.030, 0.050, 0.920]])
    state = np.empty(D, dtype=int)
    direction = np.zeros(D)
    state[0], cur_dir = 1, 1.0
    u = rng.random(D)
    for d in range(1, D):
        state[d] = int(np.searchsorted(np.cumsum(P[state[d - 1]]), u[d]))
        if state[d] == 0 and state[d - 1] != 0:
            cur_dir = 1.0 if rng.random() < 0.65 else -1.0
        direction[d] = cur_dir if state[d] == 0 else 0.0

    # --- daily volatility: regime level times a persistent log-vol factor
    lv = np.zeros(D)
    shocks = rng.normal(0, 0.08, D)
    for d in range(1, D):
        lv[d] = 0.97 * lv[d - 1] + shocks[d]
    sigma_d = np.array([0.009, 0.010, 0.024])[state] * np.exp(lv)
    drift_d = spec.edge * (0.0012 * direction - 0.0015 * (state == 2))
    phi = spec.edge * np.array([0.05, -0.08, 0.0])[state]

    # --- per-bar variance shares: 30% overnight, 70% intraday with a U shape
    share = BAR_HOURS * U_SHAPE
    share = 0.70 * share / share.sum()
    t_scale = np.sqrt(3.0 / 5.0)  # unit variance Student-t(5)
    gap_eps = rng.standard_t(5, D) * t_scale
    bar_eps = rng.standard_t(5, (D, K)) * t_scale
    sub = rng.normal(0.0, 1.0, (D, K, spec.substeps))

    o = np.empty((D, K)); h = np.empty((D, K)); l = np.empty((D, K)); c = np.empty((D, K))
    log_p, prev_r = np.log(spec.s0), 0.0
    steps = np.arange(1, spec.substeps + 1) / spec.substeps
    for d in range(D):
        log_p += 0.3 * drift_d[d] + np.sqrt(0.30) * sigma_d[d] * gap_eps[d]
        for k in range(K):
            sig = sigma_d[d] * np.sqrt(share[k])
            r = 0.7 * drift_d[d] / K + phi[d] * prev_r + sig * bar_eps[d, k]
            noise = np.cumsum(sub[d, k]) * sig / np.sqrt(spec.substeps)
            path = log_p + r * steps + (noise - steps * noise[-1])  # bridge ending at r
            o[d, k] = np.exp(log_p)
            c[d, k] = np.exp(log_p + r)
            h[d, k] = max(o[d, k], c[d, k], float(np.exp(path.max())))
            l[d, k] = min(o[d, k], c[d, k], float(np.exp(path.min())))
            log_p += r
            prev_r = r

    vol_level = (sigma_d / 0.010)[:, None]
    volume = 1.0e6 * (BAR_HOURS * U_SHAPE)[None, :] * vol_level * np.exp(rng.normal(0, 0.35, (D, K)))

    local = (
        np.repeat(sessions.values, K).astype("datetime64[ns]")
        + np.tile(np.array(BAR_STARTS_MIN) * 60_000_000_000, D).astype("timedelta64[ns]")
    )
    ts = pd.DatetimeIndex(local).tz_localize(NY).tz_convert("UTC").rename("ts")
    hourly = pd.DataFrame(
        {"open": o.ravel(), "high": h.ravel(), "low": l.ravel(), "close": c.ravel(), "volume": volume.ravel()},
        index=ts,
    )
    if spec.with_spread:
        widen = np.array([1.6, 1.1, 1, 1, 1, 1, 1.1])[None, :]
        hourly["spread_rel"] = (1.0e-4 * vol_level * widen * np.exp(rng.normal(0, 0.2, (D, K)))).ravel()

    fx = pd.Series(1.30 * np.exp(np.cumsum(rng.normal(0, 0.08 / np.sqrt(252), D))), index=sessions, name="EURUSD")
    years = np.arange(D) / 252.0
    rate = pd.Series(0.03 + 0.02 * np.sin(2 * np.pi * years / 6.0), index=sessions, name="benchmark_rate")

    ev = []
    for i, s in enumerate(sessions):
        if s.dayofweek == 2 and (i // 5) % 6 == 0:
            ev.append((s + pd.Timedelta(hours=14), "SYN-FOMC", 1))
        if s.dayofweek == 4 and s.day <= 7:
            ev.append((s + pd.Timedelta(hours=8, minutes=30), "SYN-NFP", 1))
        if s.dayofweek == 1 and 10 <= s.day <= 16:
            ev.append((s + pd.Timedelta(hours=8, minutes=30), "SYN-CPI", 1))
    events = pd.DataFrame(ev, columns=["ts", "name", "tier"])
    events["ts"] = pd.DatetimeIndex(events["ts"]).tz_localize(NY).tz_convert("UTC")

    truth = pd.DataFrame({"state": state, "direction": direction, "sigma_d": sigma_d}, index=sessions)
    return {"hourly": hourly, "fx": fx, "rate": rate, "events": events, "truth": truth, "source": SYNTHETIC_SOURCE}
