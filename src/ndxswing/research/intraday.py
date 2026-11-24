"""Intraday noise-area momentum ("Bot A"), frozen specification.

After Zarattini, Aziz & Barbon (2024): the session is given a "noise area"
around its open and the previous close; a 30-minute close outside it is a
momentum signal, held while price stays beyond both the boundary and the
session anchor price, and always flat before the close.

Rule, evaluated at the 30-minute closes 10:00 ... 15:30 ET of a regular session:

    sigma(t)  = mean over the 14 previous sessions of |Close(t) / Open(09:30) - 1|
    Upper(t)  = max(Open, PrevClose) (1 + sigma(t))
    Lower(t)  = min(Open, PrevClose) (1 - sigma(t))
    anchor(t) = session VWAP (typical price x volume) or, with no volume feed, session TWAP

    entry  flat at a close 10:00 <= t <= 14:30 and Close > Upper (long) / Close < Lower (short):
           market order at the next minute's open, at most two trades per session
    stop   D = max(|entry - anchor(t)|, 1.2 ATR_30m(14)); skipped if D / entry < 0.20%;
           hard stop at entry -/+ D, checked every minute
    exit   at a later 30-minute close, Close < max(Upper, anchor) (long) or
           Close > min(Lower, anchor) (short): market order at the next minute's open;
           otherwise market order at 15:55 ET

Nothing here is a tuned value: the constants are the ones written down before
the rule was first run.

Fill conventions: bar prices are mid quotes; market and stop fills pay half
the spread plus slippage; a bar that opens beyond the stop fills at the open.
Costs are either fractions of the contemporaneous price or fixed index
points (``IntradayCosts.in_points``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import TRADE_COLUMNS, BacktestResult
from ..config import RiskLimits
from ..risk.engine import CircuitBreakers

MINUTES = 390                 # 09:30 .. 15:59 ET
FLAT_MINUTE = 385             # 15:55 ET
N_CLOSES = 13                 # 30-minute closes 10:00 .. 16:00
LAST_ENTRY_CLOSE = 9          # the 14:30 close
LOT_REFERENCE = 25_288.25     # index level at which the broker's lot size is taken


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class SessionGrid:
    """Full regular sessions on a fixed 390-minute grid."""

    dates: pd.DatetimeIndex
    open: np.ndarray            # (D, 390)
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    prev_close: np.ndarray      # close of the previous exchange session (also when that one was shortened)
    fx: np.ndarray              # quote currency per account currency (USD per EUR)
    source: str = "unknown"

    def __len__(self) -> int:
        return len(self.dates)

    @classmethod
    def from_minute_bars(cls, minutes: pd.DataFrame, sessions: pd.DataFrame, fx: pd.Series,
                         min_minutes: int = 385, source: str = "unknown") -> "SessionGrid":
        """``minutes``: 1-minute bars indexed by tz-aware bar-start time.
        ``sessions``: exchange sessions (index = date) with ``n_min`` and
        ``close``.  Sessions with fewer than ``min_minutes`` bars (early
        closes, broken days) are left out; a missing minute repeats the last
        price with zero volume."""
        m = minutes.sort_index()
        loc = m.index.tz_convert("America/New_York")
        day = loc.tz_localize(None).normalize()
        mi = (loc.hour * 60 + loc.minute - 570).to_numpy()
        full = sessions.index[sessions["n_min"] >= min_minutes]
        d = len(full)
        arr = {k: np.full((d, MINUTES), np.nan) for k in ("open", "high", "low", "close")}
        vol = np.zeros((d, MINUTES))
        pos = pd.Series(np.arange(d), index=full)
        ok = day.isin(full) & (mi >= 0) & (mi < MINUTES)
        r, c = pos.reindex(day[ok]).to_numpy().astype(int), mi[ok]
        for k in arr:
            arr[k][r, c] = m[k].to_numpy()[ok]
        vol[r, c] = np.nan_to_num(m["volume"].to_numpy()[ok])
        for j in range(MINUTES):
            miss = np.isnan(arr["close"][:, j])
            if miss.any():
                fill = arr["close"][miss, j - 1] if j else arr["open"][miss, 1]
                for k in arr:
                    arr[k][miss, j] = fill
        prev = sessions["close"].shift(1).reindex(full).to_numpy()
        fxa = fx.reindex(fx.index.union(full)).ffill().reindex(full).to_numpy()
        return cls(full, arr["open"], arr["high"], arr["low"], arr["close"], vol, prev, fxa, source)


def read_minute_bars(path: str | Path, before: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """1-minute bars from a parquet file.  With ``before`` the filter is applied
    by the reader, so later rows never enter memory (used to keep a hold-out
    period unread)."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    data = ds.dataset(str(path))
    if before is None:
        tbl = data.to_table()
    else:
        cut = pd.Timestamp(before, tz="UTC").to_pydatetime()
        tbl = data.to_table(filter=ds.field("ts") < pa.scalar(cut, type=pa.timestamp("ns", tz="UTC")))
    out = tbl.to_pandas().set_index("ts").sort_index()
    if before is not None and len(out):
        assert out.index.max() < pd.Timestamp(before, tz="UTC")
    return out


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IntradayCosts:
    name: str
    half_spread: float            # fraction of price, or index points when ``in_points``
    slippage: float               # per market / stop leg, same unit
    fx_fee: float = 0.0           # mark-up on converting realised P&L into the account currency
    lot_step: float = 0.001       # at ``LOT_REFERENCE``
    in_points: bool = False
    stress: float = 1.0

    def leg(self, price: float) -> float:
        """Cost of one market or stop leg as a fraction of ``price``."""
        c = (self.half_spread + self.slippage) * self.stress
        return c / price if self.in_points else c

    def stressed(self, factor: float) -> "IntradayCosts":
        return replace(self, stress=self.stress * factor, name=f"{self.name} x{factor:g}")


def cost_schedules(broker: str) -> dict[str, IntradayCosts]:
    """Three schedules per broker: I = today's quote in basis points of price,
    II = the quote in index points whatever the index level, III = 1.5 x II."""
    spec = {"pepperstone": dict(spread_pts=1.0, fx_fee=0.0, lot_step=0.10),
            "capital": dict(spread_pts=1.8, fx_fee=0.007, lot_step=0.001)}[broker]
    slip_pts, sp = 0.5, spec["spread_pts"]
    i = IntradayCosts(f"{broker} I (bps)", 0.5 * round(sp / LOT_REFERENCE * 1e4, 2) * 1e-4, 0.20e-4, spec["fx_fee"], spec["lot_step"])
    ii = IntradayCosts(f"{broker} II (points)", 0.5 * sp, slip_pts, spec["fx_fee"], spec["lot_step"], in_points=True)
    return {"I": i, "II": ii, "III": ii.stressed(1.5)}


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder ATR over a (sessions, bars) grid read row by row; the overnight
    gap is part of the first bar's true range.  Value at a bar uses that bar."""
    hf, lf, cf = high.ravel(), low.ravel(), close.ravel()
    pc = np.r_[np.nan, cf[:-1]]
    tr = np.nanmax(np.c_[hf - lf, np.abs(hf - pc), np.abs(lf - pc)], axis=1)
    atr = np.full(len(tr), np.nan)
    if len(tr) >= n:
        atr[n - 1] = tr[:n].mean()
        for i in range(n, len(tr)):
            atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr.reshape(high.shape)


@dataclass
class NoiseAreaInputs:
    sigma: np.ndarray        # (D, 13) at each 30-minute close
    upper: np.ndarray
    lower: np.ndarray
    anchor: np.ndarray       # (D, 390) session VWAP or TWAP at each minute
    atr30: np.ndarray        # (D, 13)
    anchor_kind: str


def noise_area_inputs(grid: SessionGrid, anchor: str = "vwap", lookback: int = 14, atr_window: int = 14) -> NoiseAreaInputs:
    d = len(grid)
    c30 = grid.close[:, 29::30]
    move = np.abs(c30 / grid.open[:, [0]] - 1.0)
    sigma = np.full(move.shape, np.nan)
    for i in range(lookback, d):
        sigma[i] = move[i - lookback:i].mean(axis=0)              # previous sessions only
    hi = np.fmax(grid.open[:, 0], grid.prev_close)
    lo = np.fmin(grid.open[:, 0], grid.prev_close)
    typ = (grid.high + grid.low + grid.close) / 3.0
    if anchor == "vwap":
        w = np.where(grid.volume.sum(1, keepdims=True) > 0, grid.volume, 1.0)   # a session without volume: equal weights
        cw = np.cumsum(w, 1)
        twap = np.cumsum(typ, 1) / np.arange(1, MINUTES + 1)
        anc = np.where(cw > 0, np.cumsum(typ * w, 1) / np.where(cw > 0, cw, 1.0), twap)
    elif anchor == "twap":
        anc = np.cumsum(typ, 1) / np.arange(1, MINUTES + 1)
    else:
        raise ValueError("anchor must be 'vwap' or 'twap'")
    sh = (d, N_CLOSES, 30)
    atr = wilder_atr(grid.high.reshape(sh).max(2), grid.low.reshape(sh).min(2), c30, atr_window)
    return NoiseAreaInputs(sigma, hi[:, None] * (1 + sigma), lo[:, None] * (1 - sigma), anc, atr, anchor)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


@dataclass
class IntradayResult:
    trades: pd.DataFrame
    equity: pd.Series              # account equity at each session close
    initial_equity: float
    stats: dict

    def as_backtest_result(self) -> BacktestResult:
        """The same run in the shape the tearsheet and metric functions expect."""
        t = self.trades
        cols = TRADE_COLUMNS + ["entry_session"]
        full = pd.DataFrame({c: t[c] if c in t else 0.0 for c in cols}) if len(t) else pd.DataFrame(columns=cols)
        if len(full):
            full["shadow"] = False
        return BacktestResult(self.equity.copy(), full, self.initial_equity, dict(self.stats))

    def window(self, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None) -> "IntradayResult":
        """Sub-period view with equity re-based to the equity entering it."""
        eq = self.equity
        lo = eq.index[0] if start is None else pd.Timestamp(start)
        hi = eq.index[-1] if end is None else pd.Timestamp(end)
        sub = eq[(eq.index >= lo) & (eq.index <= hi)]
        before = eq[eq.index < lo]
        init = float(before.iloc[-1]) if len(before) else self.initial_equity
        t = self.trades
        if len(t):
            t = t[(t["entry_session"] >= lo) & (t["entry_session"] <= hi)]
        return IntradayResult(t, sub, init, dict(self.stats))


def run_noise_area(grid: SessionGrid, inputs: NoiseAreaInputs, costs: IntradayCosts, *, b_risk: float = 0.0025,
                   initial_equity: float = 5000.0, start: str | pd.Timestamp | None = None,
                   end: str | pd.Timestamp | None = None, limits: RiskLimits | None = RiskLimits(),
                   max_leverage: float | None = 1.5, compounding: bool = True, max_trades: int = 2,
                   min_stop: float = 0.0020, atr_mult: float = 1.2, lot_reference: float = LOT_REFERENCE) -> IntradayResult:
    """Bot A over the sessions of ``grid`` in [start, end].

    limits        circuit breakers (1% intraday loss halts entries, 2% rolling five-session loss halves
                  risk, -4% / -6% of the true high-water mark scale risk to 0.75 / 0.25 and cap leverage
                  at 1.0, -8% pauses for 21 sessions, -10% deactivates for good); None switches them off
    max_leverage  hard cap on notional / equity (None: no cap)
    compounding   risk budget is ``b_risk`` of current equity (True) or of ``initial_equity`` (False)

    With ``limits=None, max_leverage=None, compounding=False`` this is the plain
    rule used in the 2015-2020 triage.
    """
    O, H, L, C = grid.open, grid.high, grid.low, grid.close
    i0 = 0 if start is None else int(grid.dates.searchsorted(pd.Timestamp(start)))
    i1 = len(grid) if end is None else int(grid.dates.searchsorted(pd.Timestamp(end), side="right"))
    equity = float(initial_equity)
    brk = CircuitBreakers(limits, equity) if limits is not None else None
    stats = {"signals": 0, "skipped_min_stop": 0, "skipped_min_lot": 0, "leverage_capped": 0, "halted_entries": 0,
             "breaker_flattens": 0, "source": grid.source, "anchor": inputs.anchor_kind, "costs": costs.name,
             "b_risk": b_risk, "cost_stress": costs.stress}
    trades: list[dict] = []
    eq_val: list[float] = []

    for d in range(i0, i1):
        date = grid.dates[d]
        if brk is not None:
            brk.on_session_open(equity, date)
        ready = np.isfinite(inputs.sigma[d, 0]) and np.isfinite(grid.prev_close[d]) and np.isfinite(inputs.atr30[d, 0])
        pos: dict | None = None
        want, leave, n_day = 0, False, 0
        x = grid.fx[d]

        def close_position(minute: int, mid: float, reason: str) -> None:
            nonlocal equity, pos
            k = pos["k"]
            fill = mid * (1 - k * costs.leg(mid))
            pts = (fill - pos["fill"]) * k
            gross_pts = (mid - pos["mid"]) * k
            eur = pos["units"] * pts / x
            fee = costs.fx_fee * abs(eur)
            cost_eur = pos["units"] * (gross_pts - pts) / x
            share_spread = costs.half_spread / (costs.half_spread + costs.slippage) if costs.half_spread + costs.slippage > 0 else 0.0
            risk_eur = pos["units"] * pos["dstop"] / x
            trades.append({
                "trade_id": len(trades), "shadow": False, "segment": 0, "direction": k,
                "entry_ts": date + pd.Timedelta(minutes=570 + pos["minute"]), "exit_ts": date + pd.Timedelta(minutes=570 + minute),
                "entry_px": pos["fill"], "exit_px": fill, "units": pos["units"], "stop_init": pos["stop"], "r_px": pos["dstop"],
                "entry_notional": pos["units"] * pos["mid"] / x, "initial_risk": risk_eur,
                "gross_pnl": pos["units"] * gross_pts / x, "commission": 0.0, "spread_cost": cost_eur * share_spread,
                "slippage_cost": cost_eur * (1 - share_spread), "financing": 0.0, "fx_fee": fee, "net_pnl": eur - fee,
                "r_multiple": (eur - fee) / risk_eur, "underlying_return": (eur - fee) / (pos["units"] * pos["mid"] / x),
                "bars_held": minute - pos["minute"], "sessions_held": 1, "exit_reason": reason, "tp1_hit": False,
                "mfe_r": np.nan, "mae_r": np.nan, "score": 1.0, "leverage": pos["leverage"], "binding": pos["binding"],
                "turnover": pos["units"] * (pos["mid"] + mid) / x, "entry_session": date,
                "gross_r": gross_pts / pos["dstop"], "entry_minute": pos["minute"], "exit_minute": minute,
                "stop_pct": pos["dstop"] / pos["mid"]})
            equity += eur - fee
            pos = None
            if brk is not None:
                brk.on_bar_close(equity, date)

        for i in range(N_CLOSES):
            a = 30 * i
            if pos is not None and leave:
                close_position(a, O[d, a], pos.pop("leave_reason", "signal"))
            if pos is None and want and ready:
                entry = O[d, a]
                dstop = max(abs(entry - inputs.anchor[d, a - 1]), atr_mult * inputs.atr30[d, i - 1])
                if dstop / entry < min_stop:
                    stats["skipped_min_stop"] += 1
                else:
                    m_dd = brk.m_dd if brk is not None else 1.0
                    base = equity if compounding else initial_equity
                    units = b_risk * base * m_dd * x / dstop
                    binding = "risk_budget"
                    cap = max_leverage if max_leverage is not None else np.inf
                    if brk is not None:
                        cap = min(cap, brk.leverage_cap)
                    if np.isfinite(cap) and units * entry / x > cap * equity:
                        units, binding = cap * equity * x / entry, "max_leverage"
                        stats["leverage_capped"] += 1
                    step = costs.lot_step * lot_reference / entry
                    units = np.floor(units / step + 1e-9) * step
                    if units < step - 1e-12:
                        stats["skipped_min_lot"] += 1
                    else:
                        fill = entry * (1 + want * costs.leg(entry))
                        pos = {"k": want, "minute": a, "mid": entry, "fill": fill, "dstop": dstop, "stop": entry - want * dstop,
                               "units": units, "leverage": units * entry / x / equity, "binding": binding}
                        n_day += 1
            want, leave = 0, False
            if pos is not None:                                   # hard stop, minute by minute, to the next decision point
                k, stop = pos["k"], pos["stop"]
                j1 = min(a + 30, FLAT_MINUTE)
                adv = L[d, a:j1] if k > 0 else H[d, a:j1]
                hit = (stop - adv) * k >= 0
                if hit.any():
                    j = a + int(hit.argmax())
                    o = O[d, j]
                    close_position(j, o if (stop - o) * k >= 0 else stop, "stop")
                elif a + 30 > FLAT_MINUTE:
                    close_position(FLAT_MINUTE, O[d, FLAT_MINUTE], "time")
            if i == N_CLOSES - 1 or not ready:
                continue
            cm = a + 29
            c = C[d, cm]
            if pos is not None:
                k = pos["k"]
                if brk is not None:
                    mark = equity + pos["units"] * (c - pos["fill"]) * k / x
                    if brk.on_bar_close(mark, date):               # pause or deactivation: out at the next open
                        leave, pos["leave_reason"] = True, "breaker_flatten"
                        stats["breaker_flattens"] += 1
                thr = max(inputs.upper[d, i], inputs.anchor[d, cm]) if k > 0 else min(inputs.lower[d, i], inputs.anchor[d, cm])
                if (c - thr) * k < 0:
                    leave = True
            elif i <= LAST_ENTRY_CLOSE and n_day < max_trades:
                k = 1 if c > inputs.upper[d, i] else (-1 if c < inputs.lower[d, i] else 0)
                if k:
                    stats["signals"] += 1
                    if brk is not None and not brk.can_enter:
                        stats["halted_entries"] += 1
                    else:
                        want = k
        if brk is not None:
            brk.on_bar_close(equity, date)
            brk.on_session_close(equity)
        eq_val.append(equity)

    eq = pd.Series(eq_val, index=grid.dates[i0:i1].rename("session"), name="equity", dtype=float)
    tdf = pd.DataFrame(trades)
    if brk is not None:
        stats |= {"breaker_events": [(str(pd.Timestamp(s).date()), e, round(float(v), 2)) for s, e, v in brk.events],
                  "n_pauses": sum(1 for ev in brk.events if ev[1] == "pause"),
                  "deactivated_at": next((ev[0] for ev in brk.events if ev[1] == "deactivate"), None),
                  "true_peak": brk.true_peak, "intraday_halts": sum(1 for ev in brk.events if ev[1] == "intraday_halt")}
    else:
        stats |= {"n_pauses": 0, "deactivated_at": None}
    stats["n_sessions"] = len(eq)
    return IntradayResult(tdf, eq, float(initial_equity), stats)
