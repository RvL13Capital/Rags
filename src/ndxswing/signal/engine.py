"""Layer 3/4: conditional entry signals and the net-expectancy hurdle.

The engine never sizes and never talks to a broker.  At the close of hourly
bar ``t`` it may emit one immutable :class:`SignalProposal` describing a
stop-limit entry for bar ``t+1``.

Causality contract
------------------
* hourly inputs at row t use bars <= t (Donchian levels exclude bar t itself);
* daily inputs are lagged one full session: every bar of session d sees the
  daily feature row of session d-1 (regime, Hurst, GARCH, EVT, EMA direction);
* the event calendar is a schedule known in advance.

``tests/test_signal_engine.py`` verifies the contract by truncation: signals
computed on data[:t] equal signals computed on the full sample for all bars
<= t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CostModel, ModuleFlags, StrategyParams
from ..features.indicators import NY, atr, donchian_prev
from ..features.pipeline import FeatureSet

#: daily column each regime model needs
REGIME_INPUT = {"hmm3": "p_trend", "hmm2": "p2_stress", "rv20": "rv20_pct"}

GATES = (
    "breakout", "stop_width", "regime", "hurst", "garch_health", "arma_z", "net_expectancy",
    "spread", "vpin", "event_blackout",
)


@dataclass(frozen=True)
class SignalProposal:
    bar_index: int
    ts: pd.Timestamp
    direction: int            # +1 long, -1 short
    entry_stop: float         # stop trigger P_Entry
    entry_limit: float
    stop_price: float         # initial protective stop (1R below/above P_Entry)
    d_stop: float
    atr: float
    ref_price: float          # close of the signal bar
    score: float              # S_t in [0, 1]
    m_reg: float
    m_hurst: float
    m_liq: float
    sigma_1d: float           # one-day volatility for the vol target
    es99_daily: float         # EVT ES_0.99 for the trade direction (nan if unavailable)
    mu_h: float
    sigma_h: float
    z_h: float
    spread_rel: float         # quoted spread / mid at the signal bar (nan if unknown)


@dataclass
class PreparedSignals:
    """Vectorised per-bar arrays for one (params, flags) pair."""

    params: StrategyParams
    flags: ModuleFlags
    direction: np.ndarray     # 0 = no signal
    entry_stop: np.ndarray
    entry_limit: np.ndarray
    stop_price: np.ndarray
    d_stop: np.ndarray
    atr: np.ndarray
    score: np.ndarray
    m_reg: np.ndarray
    m_hurst: np.ndarray
    m_liq: np.ndarray
    sigma_1d: np.ndarray
    es99: np.ndarray
    mu_h: np.ndarray
    sigma_h: np.ndarray
    z_h: np.ndarray
    spread_rel: np.ndarray
    exit_long: np.ndarray     # daily-state exit flags (trend flip / stress), constant within a session
    exit_short: np.ndarray
    funnel: dict[str, int] = field(default_factory=dict)


class SignalEngine:
    def __init__(
        self,
        features: FeatureSet,
        costs: CostModel = CostModel(),
        events: pd.DataFrame | None = None,
        rate: pd.Series | None = None,
    ) -> None:
        self.features = features
        self.costs = costs
        hb = features.hourly
        self.ts = pd.DatetimeIndex(hb.index)
        self.n = len(hb)
        self.open = hb["open"].to_numpy(float)
        self.high = hb["high"].to_numpy(float)
        self.low = hb["low"].to_numpy(float)
        self.close = hb["close"].to_numpy(float)
        self._hb = hb
        sess = pd.DatetimeIndex(hb["session"])
        daily = features.daily
        self.sessions = pd.DatetimeIndex(daily.index)
        self.sess_idx = self.sessions.get_indexer(sess)
        if (self.sess_idx < 0).any():
            raise ValueError("hourly bars reference sessions missing from the daily feature table")
        self.is_first = np.r_[True, self.sess_idx[1:] != self.sess_idx[:-1]]
        self.is_last = np.r_[self.sess_idx[1:] != self.sess_idx[:-1], True]
        self._lag = self.sess_idx - 1          # daily row visible to each bar
        self._daily = daily
        self._cache: dict[tuple, np.ndarray] = {}
        self.spread_rel = hb["spread_rel"].to_numpy(float) if "spread_rel" in hb else np.full(self.n, np.nan)
        self.spread_med = hb["spread_med"].to_numpy(float) if "spread_med" in hb else np.full(self.n, np.nan)
        self.vpin_pct = hb["vpin_pct"].to_numpy(float) if "vpin_pct" in hb else np.full(self.n, np.nan)
        self.bench_rate = self._lag_series(rate, costs.default_benchmark_rate)
        self.event_block = self._event_block(events)
        self.has_events = events is not None and len(events) > 0

    # ----------------------------------------------------------- helpers
    def _lag_series(self, s: pd.Series | None, default: float) -> np.ndarray:
        """Previous-session value of a daily series for every bar."""
        if s is None:
            return np.full(self.n, default)
        v = s.reindex(self.sessions, method="ffill").to_numpy(float)
        out = np.where(self._lag >= 0, v[np.clip(self._lag, 0, None)], np.nan)
        return np.where(np.isfinite(out), out, default)

    def lagged_daily(self, col: str) -> np.ndarray:
        """Daily feature column as seen by each bar (session d-1 value)."""
        key = ("d", col)
        if key not in self._cache:
            if col not in self._daily.columns:
                raise KeyError(f"daily feature {col!r} was not built; enable its block in FeatureConfig")
            v = self._daily[col].to_numpy(float)
            self._cache[key] = np.where(self._lag >= 0, v[np.clip(self._lag, 0, None)], np.nan)
        return self._cache[key]

    def _atr(self, window: int) -> np.ndarray:
        key = ("atr", window)
        if key not in self._cache:
            hb = self._hb
            self._cache[key] = atr(hb["high"], hb["low"], hb["close"], window).to_numpy(float)
        return self._cache[key]

    def _donchian(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        key = ("don", window)
        if key not in self._cache:
            hi, lo = donchian_prev(self._hb["high"], self._hb["low"], window)
            self._cache[key] = (hi.to_numpy(float), lo.to_numpy(float))
        return self._cache[key]

    def _struct(self, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        key = ("struct", lookback)
        if key not in self._cache:
            lo = self._hb["low"].rolling(lookback, min_periods=lookback).min().to_numpy(float)
            hi = self._hb["high"].rolling(lookback, min_periods=lookback).max().to_numpy(float)
            self._cache[key] = (lo, hi)
        return self._cache[key]

    def _event_block(self, events: pd.DataFrame | None) -> np.ndarray:
        """True when a tier-1 event falls inside the order's live window:
        (close_t, max(close_t + 2h, start_{t+1} + 1h)].  For the last bar of a
        session this spans the overnight gap, so a pre-market release blocks
        an order that would otherwise be working at the next open."""
        if events is None or len(events) == 0:
            return np.zeros(self.n, dtype=bool)
        ts_ns = self.ts.asi8
        unit = self.ts.unit
        per_h = {"s": 3600, "ms": 3_600_000, "us": 3_600_000_000, "ns": 3_600_000_000_000}[unit]
        ev = np.sort(pd.DatetimeIndex(events["ts"]).tz_convert("UTC").as_unit(unit).asi8)
        local_close = (pd.DatetimeIndex(self._hb["session"]) + pd.Timedelta(hours=16)).tz_localize(NY).tz_convert("UTC")
        sess_close = local_close.as_unit(unit).asi8
        nxt = np.r_[ts_ns[1:], ts_ns[-1] + 24 * per_h]
        close_t = np.where(self.is_last, sess_close, nxt)
        end = np.maximum(close_t + 2 * per_h, nxt + per_h)
        return np.searchsorted(ev, end, side="right") > np.searchsorted(ev, close_t, side="right")

    def available_modules(self) -> dict[str, bool]:
        """Which optional modules have the data they need."""
        d = self._daily.columns
        return {
            "regime_gate": any(c in d for c in REGIME_INPUT.values()),
            "hurst_gate": "hurst" in d,
            "garch_vol": "sigma_1d" in d,
            "evt_es_cap": "es99_long" in d,
            "arma_z_filter": "z_1" in d,
            "spread_gate": bool(np.isfinite(self.spread_rel).any() and np.isfinite(self.spread_med).any()),
            "vpin_filter": bool(np.isfinite(self.vpin_pct).any()),
            "event_blackout": self.has_events,
        }

    # ----------------------------------------------------------- prepare
    def prepare(self, params: StrategyParams, flags: ModuleFlags) -> PreparedSignals:
        p, f, n = params, flags, self.n
        avail = self.available_modules()
        for mod, ok in avail.items():
            if getattr(f, mod) and not ok:
                raise ValueError(f"module {mod!r} is enabled but its input data is not available")
        if f.regime_gate and REGIME_INPUT[f.regime_model] not in self._daily.columns:
            raise ValueError(f"regime model {f.regime_model!r} is enabled but its input data is not available")
        c = self.close
        a = self._atr(p.atr_window)
        hi_prev, lo_prev = self._donchian(p.donchian_window)
        lo_s, hi_s = self._struct(p.struct_lookback)
        with np.errstate(invalid="ignore"):
            d_daily = np.sign(self.lagged_daily("ema20") - self.lagged_daily("ema50"))
            d_daily = np.where(np.isfinite(d_daily), d_daily, 0.0)
            long_bo = (d_daily > 0) & (c > hi_prev)
            short_bo = (d_daily < 0) & (c < lo_prev)
            direction = np.where(long_bo, 1, np.where(short_bo, -1, 0))
            direction = np.where(np.isfinite(a) & (a > 0), direction, 0)

            p_long = hi_prev + p.entry_buffer_atr * a
            p_short = lo_prev - p.entry_buffer_atr * a
            entry_stop = np.where(direction > 0, p_long, p_short)
            entry_limit = entry_stop + direction * p.limit_offset_atr * a
            d_struct = np.where(direction > 0, entry_stop - lo_s, hi_s - entry_stop)
            d_stop = np.maximum(p.atr_stop_mult * a, d_struct)
            stop_price = entry_stop - direction * d_stop

            alive = direction != 0
            funnel = {"breakout": int(alive.sum())}

            def gate(name: str, ok: np.ndarray) -> None:
                nonlocal alive
                alive = alive & ok
                funnel[name] = int(alive.sum())

            gate("stop_width", d_stop <= p.max_stop_ratio * p.atr_stop_mult * a)

            # ---- regime
            m_reg = np.ones(n)
            exit_stress = np.zeros(n, dtype=bool)
            s_reg = np.full(n, np.nan)
            if f.regime_gate:
                if f.regime_model == "hmm3":
                    pt, ps, cf = (self.lagged_daily(k) for k in ("p_trend", "p_stress", "conf"))
                    ok = (pt > p.p_trend_min) & (ps < p.p_stress_max) & (cf >= p.conf_min)
                    full = (pt > max(p.p_trend_full, p.p_trend_min)) & (ps < p.p_stress_full)
                    s_reg = np.clip((pt - p.p_trend_min) / 0.25, 0, 1)
                elif f.regime_model == "hmm2":
                    ps = self.lagged_daily("p2_stress")
                    ok = ps < p.p_stress_max
                    full = ps < p.p_stress_full
                    s_reg = np.clip(1.0 - ps / p.p_stress_max, 0, 1)
                else:  # deterministic RV20 percentile filter
                    rk = self.lagged_daily("rv20_pct")
                    ok = rk <= p.rv_filter_pct
                    full = ok
                    ps = np.where(rk >= p.rv_stress_pct, 1.0, 0.0)
                    s_reg = np.clip(1.0 - rk / p.rv_filter_pct, 0, 1)
                m_reg = np.where(ok, np.where(full, 1.0, 0.75), 0.0)
                exit_stress = np.nan_to_num(ps, nan=0.0) > p.stress_exit
                gate("regime", ok)

            # ---- Hurst
            m_hurst = np.ones(n)
            s_hurst = np.full(n, np.nan)
            if f.hurst_gate:
                hh, hl = self.lagged_daily("hurst"), self.lagged_daily("hurst_lcb")
                ok = hh >= p.hurst_min
                if f.hurst_mode == "lcb":
                    ok = ok & (hl > p.hurst_lcb_min)
                m_hurst = np.where(ok, np.where(hh >= max(p.hurst_strong, p.hurst_min), 1.0, 0.8), 0.0)
                s_hurst = np.clip((hh - p.hurst_min) / 0.10, 0, 1)
                gate("hurst", ok)

            # ---- volatility for the daily vol target
            if f.garch_vol:
                sigma_1d = self.lagged_daily("sigma_1d")
                gate("garch_health", (self.lagged_daily("garch_ok") > 0.5) & np.isfinite(sigma_1d) & (sigma_1d > 0))
            else:
                sigma_1d = self.lagged_daily("sigma_rv")
                alive = alive & np.isfinite(sigma_1d) & (sigma_1d > 0)

            # ---- ARMA directional hurdle and net expectancy
            H = p.horizon_days
            mu_h = sig_h = z_h = np.full(n, np.nan)
            s_z = np.full(n, np.nan)
            if f.arma_z_filter:
                mu_h, sig_h, z_h = (self.lagged_daily(f"{k}_{H}") for k in ("mu", "sigma", "z"))
                gate("arma_z", direction * z_h > p.z_hurdle)
                spread = np.where(np.isfinite(self.spread_rel), self.spread_rel * self.costs.stress,
                                  2.0 * self.costs.half_spread_at(c))
                fin_daily = np.where(
                    direction > 0, self.bench_rate + self.costs.financing_markup * self.costs.stress,
                    self.costs.financing_markup * self.costs.stress - self.bench_rate,
                ) / self.costs.financing_day_count
                cost = (2.0 * self.costs.commission_bps * 1e-4 * self.costs.stress + spread
                        + 2.0 * self.costs.slippage_frac + p.expected_hold_days * fin_daily)
                r_frac = d_stop / entry_stop
                e_net = direction * mu_h - cost
                reward_risk = (direction * mu_h + sig_h) / r_frac
                gate("net_expectancy", (e_net > p.expectancy_r_hurdle * r_frac) & (reward_risk >= p.min_reward_risk))
                s_z = np.clip((np.abs(z_h) - p.z_hurdle) / 0.65, 0, 1)

            # ---- liquidity
            m_liq = np.ones(n)
            s_spread = np.full(n, np.nan)
            if f.spread_gate:
                gate("spread", self.spread_rel <= p.spread_ok_mult * self.spread_med)
                s_spread = np.clip(1.0 - self.spread_rel / (p.spread_ok_mult * self.spread_med), 0, 1)
            if f.vpin_filter:
                vp = self.vpin_pct
                gate("vpin", vp <= p.vpin_block_pct)
                m_liq = np.where(vp >= p.vpin_halve_pct, 0.5, 1.0)
            if f.event_blackout:
                gate("event_blackout", ~self.event_block)

            # ---- EVT expected shortfall for the trade direction
            es99 = np.full(n, np.nan)
            if f.evt_es_cap:
                es99 = np.where(direction > 0, self.lagged_daily("es99_long"), self.lagged_daily("es99_short"))
                alive = alive & np.isfinite(es99) & (es99 > 0)

            # ---- bounded signal score over the enabled components
            score = np.ones(n)
            if f.score_sizing:
                comps = [(0.30, s_reg, f.regime_gate), (0.25, s_hurst, f.hurst_gate),
                         (0.25, s_z, f.arma_z_filter), (0.20, s_spread, f.spread_gate)]
                wsum = sum(w for w, _, on in comps if on)
                if wsum > 0:
                    score = sum(w * np.nan_to_num(v, nan=0.0) for w, v, on in comps if on) / wsum

        direction = np.where(alive, direction, 0).astype(np.int8)
        funnel["signals"] = int((direction != 0).sum())
        flip = f.trend_flip_exit
        stress_on = f.regime_gate
        exit_long = (flip & (d_daily < 0)) | (stress_on & exit_stress)
        exit_short = (flip & (d_daily > 0)) | (stress_on & exit_stress)
        return PreparedSignals(
            params=p, flags=f, direction=direction, entry_stop=entry_stop, entry_limit=entry_limit,
            stop_price=stop_price, d_stop=d_stop, atr=a, score=score, m_reg=m_reg, m_hurst=m_hurst, m_liq=m_liq,
            sigma_1d=sigma_1d, es99=es99, mu_h=mu_h, sigma_h=sig_h, z_h=z_h, spread_rel=self.spread_rel,
            exit_long=exit_long, exit_short=exit_short, funnel=funnel,
        )

    # ---------------------------------------------------------- evaluate
    def evaluate(self, prep: PreparedSignals, i: int) -> SignalProposal | None:
        """Proposal generated at the close of bar ``i`` (or None)."""
        d = int(prep.direction[i])
        if d == 0:
            return None
        return SignalProposal(
            bar_index=i, ts=self.ts[i], direction=d,
            entry_stop=float(prep.entry_stop[i]), entry_limit=float(prep.entry_limit[i]),
            stop_price=float(prep.stop_price[i]), d_stop=float(prep.d_stop[i]), atr=float(prep.atr[i]),
            ref_price=float(self.close[i]), score=float(prep.score[i]), m_reg=float(prep.m_reg[i]),
            m_hurst=float(prep.m_hurst[i]), m_liq=float(prep.m_liq[i]), sigma_1d=float(prep.sigma_1d[i]),
            es99_daily=float(prep.es99[i]), mu_h=float(prep.mu_h[i]), sigma_h=float(prep.sigma_h[i]),
            z_h=float(prep.z_h[i]), spread_rel=float(prep.spread_rel[i]),
        )
