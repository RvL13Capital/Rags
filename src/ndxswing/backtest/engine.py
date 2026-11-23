"""Bar-by-bar event-driven backtester (layers 3-6 wired together, no broker).

Order / position state machine, evaluated on regular-session hourly bars:

    signal at close(t) -> RiskApprovedOrder -> WORKING stop-limit on bar t+1
        -> CANCELLED  bar opens > 0.5 ATR beyond the trigger (anti-chase)
        -> EXPIRED    unfilled after ``order_ttl_bars`` bars
        -> FILLED     -> OPEN position
              -> stop (gap-through fills at the adverse open, not the stop price)
              -> +1.5R partial take-profit, stop to breakeven + costs
              -> +2.0R chandelier trail (monotone)
              -> daily trend flip / stress regime exit at the next session open
              -> time exit at the close of trading day ``max_hold_days``
              -> circuit-breaker flatten

Fill conventions (bar prices are treated as mid quotes):

* buys pay mid + half spread, sells receive mid - half spread; stop and market
  fills additionally pay slippage; limit fills (entry limit, take-profit) do not;
* a stop is triggered by the mid touching it; if the bar opens beyond the
  stop the fill is the open (worse than the stop);
* the intrabar path is unknown, so when a bar touches both the stop and a
  target the stop is assumed to come first, and stop adjustments computed at a
  bar's close take effect from the next bar;
* on the entry bar a touch of the initial stop counts as a stop-out
  (``entry_bar_stop_policy="conservative"``) unless configured otherwise;
* financing is charged per calendar night on the previous close notional at
  benchmark +/- markup (weekends and holidays count as multiple nights).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BacktestConfig, ModuleFlags, StrategyParams
from ..risk.engine import AccountState, CircuitBreakers, PortfolioRiskEngine, RiskApprovedOrder, RiskRejection
from ..signal.engine import PreparedSignals, SignalEngine

TRADE_COLUMNS = [
    "trade_id", "shadow", "segment", "direction", "entry_ts", "exit_ts", "entry_px", "exit_px", "units",
    "stop_init", "r_px", "entry_notional", "initial_risk", "gross_pnl", "commission", "spread_cost",
    "slippage_cost", "financing", "net_pnl", "r_multiple", "underlying_return", "bars_held", "sessions_held",
    "exit_reason", "tp1_hit", "mfe_r", "mae_r", "score", "leverage", "binding", "turnover",
]


@dataclass
class _Pending:
    order: RiskApprovedOrder
    placed: int
    params: StrategyParams
    flags: ModuleFlags
    segment: int


@dataclass
class _Position:
    order: RiskApprovedOrder
    params: StrategyParams
    flags: ModuleFlags
    segment: int
    direction: int
    units: float
    units_init: float
    entry_px: float
    entry_i: int
    entry_sess: int
    entry_fx: float
    stop: float
    stop_init: float
    r_px: float
    tp1_px: float
    shadow: bool
    stop_kind: str = "stop"
    tp1_done: bool = False
    best: float = 0.0          # most favourable price since entry
    worst: float = 0.0
    realized: float = 0.0      # account ccy, from fill prices
    commission: float = 0.0
    financing: float = 0.0
    spread_cost: float = 0.0
    slip_cost: float = 0.0
    turnover: float = 0.0
    exit_value: float = 0.0    # sum(units * px) over exit legs
    exit_reason: str = ""
    exit_i: int = -1


@dataclass
class BacktestResult:
    equity: pd.Series                 # session-close equity, account currency
    trades: pd.DataFrame
    initial_equity: float
    stats: dict = field(default_factory=dict)

    def daily_returns(self) -> pd.Series:
        eq = self.equity
        if len(eq) == 0:
            return pd.Series(dtype=float)
        prev = np.r_[self.initial_equity, eq.to_numpy()[:-1]]
        return pd.Series(eq.to_numpy() / prev - 1.0, index=eq.index, name="ret")

    def real_trades(self) -> pd.DataFrame:
        t = self.trades
        return t[~t["shadow"]] if len(t) else t

    def slice(self, start: pd.Timestamp | None, end: pd.Timestamp | None) -> "BacktestResult":
        """Sub-period view: equity re-based to the equity entering the window,
        trades attributed by entry date."""
        eq = self.equity
        lo = eq.index[0] if start is None else pd.Timestamp(start)
        hi = eq.index[-1] if end is None else pd.Timestamp(end)
        sub = eq[(eq.index >= lo) & (eq.index <= hi)]
        before = eq[eq.index < lo]
        init = float(before.iloc[-1]) if len(before) else self.initial_equity
        t = self.trades
        if len(t):
            ent = pd.DatetimeIndex(t["entry_session"])
            t = t[(ent >= lo) & (ent <= hi)]
        return BacktestResult(sub, t, init, dict(self.stats, sliced=True))


class EventDrivenBacktester:
    def __init__(
        self,
        engine: SignalEngine,
        config: BacktestConfig = BacktestConfig(),
        fx: pd.Series | None = None,
    ) -> None:
        """``fx`` is the daily quote of account currency in quote currency
        (EURUSD: USD per EUR) - required when the instrument is quoted in a
        different currency than the account.  X_{quote->acct} = 1 / fx."""
        self.engine = engine
        self.cfg = config
        e, inst = engine, config.instrument
        ratio = inst.price_ratio
        self._ratio = ratio
        self._o = (e.open * ratio).tolist()
        self._h = (e.high * ratio).tolist()
        self._l = (e.low * ratio).tolist()
        self._c = (e.close * ratio).tolist()
        self._first = e.is_first.tolist()
        self._last = e.is_last.tolist()
        self._sess = e.sess_idx.tolist()
        if inst.quote_ccy != config.account_ccy:
            if fx is None:
                raise ValueError(
                    f"instrument is quoted in {inst.quote_ccy} but the account is {config.account_ccy}: "
                    "an fx series (account ccy priced in quote ccy, e.g. EURUSD) is required"
                )
            self._x = (1.0 / e._lag_series(fx, np.nan))
            first_ok = int(np.argmax(np.isfinite(self._x))) if np.isfinite(self._x).any() else 0
            self._x[:first_ok] = self._x[first_ok]
            if not np.isfinite(self._x).all():
                raise ValueError("fx series has gaps inside the bar history")
            self._x = self._x.tolist()
        else:
            self._x = [1.0] * e.n
        costs = config.costs
        hs = np.where(np.isfinite(e.spread_rel), 0.5 * e.spread_rel * costs.stress,
                      costs.half_spread_at(e.close * ratio))
        self._hs = hs.tolist()
        self._rate = e.bench_rate.tolist()
        sess_dates = e.sessions
        gaps = np.r_[1, np.diff(sess_dates.values).astype("timedelta64[D]").astype(int)]
        self._nights = gaps.tolist()

    # ------------------------------------------------------------------ run
    def run(
        self,
        params: StrategyParams | None = None,
        flags: ModuleFlags = ModuleFlags(),
        schedule: list[tuple[pd.Timestamp, StrategyParams]] | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        initial_equity: float | None = None,
    ) -> BacktestResult:
        """Simulate one configuration (``params``) or a walk-forward parameter
        ``schedule`` [(first_session, params), ...] over [start, end]."""
        e, cfg = self.engine, self.cfg
        inst, costs = cfg.instrument, cfg.costs
        if (params is None) == (schedule is None):
            raise ValueError("pass exactly one of params / schedule")
        if schedule is None:
            schedule = [(e.sessions[0], params)]
        schedule = sorted(schedule, key=lambda s: pd.Timestamp(s[0]))
        seg_start = np.array([e.sessions.searchsorted(pd.Timestamp(s)) for s, _ in schedule])
        seg_params = [p for _, p in schedule]
        preps = [e.prepare(p, flags) for p in seg_params]
        seg_of_sess = np.searchsorted(seg_start, np.arange(len(e.sessions)), side="right") - 1
        seg_of_bar = seg_of_sess[e.sess_idx]
        prep = preps[0] if len(preps) == 1 and seg_start[0] == 0 else _combine(preps, seg_of_bar)
        risk_engines = [PortfolioRiskEngine(p, flags, inst, costs, cfg.shadow_trading) for p in seg_params]

        s_lo = 0 if start is None else int(e.sessions.searchsorted(pd.Timestamp(start)))
        s_hi = len(e.sessions) - 1 if end is None else int(e.sessions.searchsorted(pd.Timestamp(end), side="right")) - 1
        if s_hi < s_lo:
            raise ValueError("empty backtest window")
        i0 = int(np.searchsorted(e.sess_idx, s_lo, side="left"))
        i1 = int(np.searchsorted(e.sess_idx, s_hi, side="right"))

        o, h, l, c = self._o, self._h, self._l, self._c
        first, last, sess, X, HS = self._first, self._last, self._sess, self._x, self._hs
        direction = prep.direction.tolist()
        exit_long, exit_short = prep.exit_long.tolist(), prep.exit_short.tolist()
        atr_arr = (prep.atr * self._ratio).tolist()
        seg_l = seg_of_bar.tolist()
        mult = inst.multiplier
        slip_f = costs.slippage_frac
        day_count = costs.financing_day_count
        markup = costs.financing_markup * costs.stress

        init_eq = cfg.initial_equity if initial_equity is None else initial_equity
        cash = init_eq
        breakers = CircuitBreakers(cfg.limits, init_eq)
        pos: _Position | None = None
        pending: _Pending | None = None
        flatten_next = False
        trades: list[dict] = []
        trade_rets: list[float] = []
        eq_dates: list = []
        eq_vals: list[float] = []
        stats = {"orders": 0, "cancel_anti_chase": 0, "expired": 0, "cancel_halt": 0, "rejections": {},
                 "commission": 0.0, "spread_cost": 0.0, "slippage_cost": 0.0, "financing": 0.0, "turnover": 0.0}
        peak_bar, max_dd_bar = init_eq, 0.0
        prev_seg = seg_l[i0] if i1 > i0 else 0

        fx_in_price = inst.fx_mode == "price"

        def trade_pnl(ps: _Position, units: float, px: float, x: float) -> float:
            """P&L in account currency.  A CFD accrues quote-currency P&L that
            is converted at the current rate; an account-currency ETF carries
            the FX move on the whole notional."""
            if fx_in_price:
                return units * mult * ps.direction * (px * x - ps.entry_px * ps.entry_fx)
            return units * mult * (px - ps.entry_px) * ps.direction * x

        # ---- fill helpers -------------------------------------------------
        def exit_fill(i: int, units: float, ref_px: float, slip: bool, reason: str) -> None:
            """Close ``units`` at reference (mid) price ``ref_px``."""
            nonlocal cash, pos
            ps = pos
            hs_px = ref_px * HS[i]
            sl_px = ref_px * slip_f if slip else 0.0
            px = ref_px - ps.direction * (hs_px + sl_px)
            x = X[i]
            pnl = trade_pnl(ps, units, px, x)
            notional = units * mult * px * x
            comm = costs.commission(notional)
            ps.realized += pnl
            ps.commission += comm
            ps.spread_cost += units * mult * hs_px * x
            ps.slip_cost += units * mult * sl_px * x
            ps.turnover += notional
            ps.exit_value += units * px
            ps.units = round(ps.units - units, 10)
            if not ps.shadow:
                cash += pnl - comm
            if ps.units <= 1e-12:
                ps.exit_reason, ps.exit_i = reason, i
                close_trade(ps)
                pos = None

        def exit_limit_fill(i: int, units: float, px: float, reason: str) -> None:
            """Limit (take-profit) fill at exactly ``px``: no spread, no slippage
            beyond the requirement that the bid/ask reached the limit."""
            nonlocal cash, pos
            ps = pos
            x = X[i]
            pnl = trade_pnl(ps, units, px, x)
            notional = units * mult * px * x
            comm = costs.commission(notional)
            ps.realized += pnl
            ps.commission += comm
            ps.turnover += notional
            ps.exit_value += units * px
            ps.units = round(ps.units - units, 10)
            if not ps.shadow:
                cash += pnl - comm
            if ps.units <= 1e-12:
                ps.exit_reason, ps.exit_i = reason, i
                close_trade(ps)
                pos = None

        def close_trade(ps: _Position) -> None:
            net = ps.realized - ps.commission - ps.financing
            risk0 = ps.units_init * mult * ps.r_px * ps.entry_fx
            notional0 = ps.units_init * mult * ps.entry_px * ps.entry_fx
            und = net / notional0
            trade_rets.append(max(und, -0.999))
            pr = ps.order.proposal
            trades.append({
                "trade_id": len(trades), "shadow": ps.shadow, "segment": ps.segment, "direction": ps.direction,
                "entry_ts": e.ts[ps.entry_i], "exit_ts": e.ts[ps.exit_i],
                "entry_session": e.sessions[ps.entry_sess],
                "entry_px": ps.entry_px, "exit_px": ps.exit_value / ps.units_init, "units": ps.units_init,
                "stop_init": ps.stop_init, "r_px": ps.r_px, "entry_notional": notional0, "initial_risk": risk0,
                "gross_pnl": ps.realized + ps.spread_cost + ps.slip_cost, "commission": ps.commission,
                "spread_cost": ps.spread_cost, "slippage_cost": ps.slip_cost, "financing": ps.financing,
                "net_pnl": net, "r_multiple": net / risk0 if risk0 > 0 else np.nan, "underlying_return": und,
                "bars_held": ps.exit_i - ps.entry_i + 1, "sessions_held": sess[ps.exit_i] - ps.entry_sess + 1,
                "exit_reason": ps.exit_reason, "tp1_hit": ps.tp1_done,
                "mfe_r": (ps.best - ps.entry_px) * ps.direction / ps.r_px,
                "mae_r": (ps.worst - ps.entry_px) * ps.direction / ps.r_px,
                "score": pr.score, "leverage": ps.order.leverage, "binding": ps.order.binding,
                "turnover": ps.turnover,
            })
            if not ps.shadow:
                stats["commission"] += ps.commission
                stats["spread_cost"] += ps.spread_cost
                stats["slippage_cost"] += ps.slip_cost
                stats["financing"] += ps.financing
                stats["turnover"] += ps.turnover

        def open_position(i: int, pd_: _Pending, ref_px: float, at_limit: bool) -> None:
            """Entry fill.  ``ref_px`` is the mid at the fill; a fill at the
            limit price pays the limit exactly."""
            nonlocal cash, pos
            od, pr, pp = pd_.order, pd_.order.proposal, pd_.params
            d = pr.direction
            lim = pr.entry_limit * self._ratio
            hs_px, sl_px = ref_px * HS[i], ref_px * slip_f
            if at_limit:
                px, hs_px, sl_px = lim, 0.0, 0.0
            else:
                px = ref_px + d * (hs_px + sl_px)
                if (px - lim) * d > 0:      # a stop-limit never pays through its limit
                    px = lim
            stop = pr.stop_price * self._ratio
            r_px = (px - stop) * d
            x = X[i]
            notional = od.units * mult * px * x
            comm = costs.commission(notional)
            pos = _Position(
                order=od, params=pp, flags=pd_.flags, segment=pd_.segment, direction=d, units=od.units,
                units_init=od.units, entry_px=px, entry_i=i, entry_sess=sess[i], entry_fx=x, stop=stop,
                stop_init=stop, r_px=r_px, tp1_px=px + d * pp.tp1_r * r_px, shadow=od.shadow,
                best=px, worst=px, commission=comm, turnover=notional,
                spread_cost=od.units * mult * hs_px * x, slip_cost=od.units * mult * sl_px * x,
            )
            if not od.shadow:
                cash -= comm

        def take_partial(i: int) -> None:
            """+1.5R: close floor(50%) if size allows, stop to breakeven + costs."""
            ps = pos
            d = ps.direction
            ps.tp1_done = True
            step = ps.order.lot_step or inst.lot_step       # effective (price-level normalised) lot
            lot = ps.order.min_lot or inst.min_lot
            q = round(math.floor(ps.params.tp1_fraction * ps.units_init / step + 1e-9) * step, 10)
            tol = lot * 1e-9
            if ps.units_init >= 2 * lot - tol and q >= lot - tol and ps.units - q >= lot - tol:
                fill = ps.tp1_px
                gap_px = o[i] - d * o[i] * HS[i]          # bid (long) / ask (short) at the open
                if ps.entry_i != i and (gap_px - fill) * d > 0:
                    fill = gap_px                         # opened through the target: price improvement
                exit_limit_fill(i, q, fill, "tp1")
            cost_frac = 2.0 * HS[i] + 2.0 * slip_f + 2.0 * costs.commission_bps * 1e-4 * costs.stress
            be = ps.entry_px * (1.0 + d * cost_frac)
            if (be - ps.stop) * d > 0:
                ps.stop, ps.stop_kind = be, "breakeven_stop"

        def equity_now(i: int) -> float:
            if pos is None or pos.shadow:
                return cash
            return cash + trade_pnl(pos, pos.units, c[i], X[i])

        def account_state() -> AccountState:
            return AccountState(
                equity=cash, fx_quote_to_acct=X[i], m_dd=breakers.m_dd, leverage_cap=breakers.leverage_cap,
                open_exposure=0.0, trade_returns=tuple(trade_rets), live=breakers.live,
            )

        # ---- main loop ------------------------------------------------------
        for i in range(i0, i1):
            is_first = first[i]
            if pos is None and pending is None and not is_first and not last[i] and direction[i] == 0:
                continue  # flat, nothing working, nothing to do: equity unchanged

            if is_first:
                if seg_l[i] != prev_seg:
                    prev_seg = seg_l[i]
                    breakers.on_refit(cash if pos is None or pos.shadow else equity_now(i - 1), e.ts[i])
                if pos is not None:  # overnight financing on the previous close notional
                    if inst.pays_financing:
                        notional = pos.units * mult * c[i - 1] * X[i]
                        rate = (self._rate[i] + markup) if pos.direction > 0 else (markup - self._rate[i])
                        fin = notional * rate / day_count * self._nights[sess[i]]
                        pos.financing += fin
                        if not pos.shadow:
                            cash -= fin
                breakers.on_session_open(equity_now(i - 1) if i > i0 else cash, e.ts[i])

            # ---------------- open position: exits within bar i
            if pos is not None:
                d = pos.direction
                oi, hi_, li, ci = o[i], h[i], l[i], c[i]
                if (pos.stop - oi) * d >= 0:                     # gapped through the stop
                    exit_fill(i, pos.units, oi, True, pos.stop_kind + "_gap")
                elif flatten_next:
                    exit_fill(i, pos.units, oi, True, "breaker_flatten")
                elif is_first and (exit_long[i] if d > 0 else exit_short[i]):
                    exit_fill(i, pos.units, oi, True, "regime_or_trend_exit")
                elif (pos.stop - (li if d > 0 else hi_)) * d >= 0:  # stop touched intrabar
                    exit_fill(i, pos.units, pos.stop, True, pos.stop_kind)
                else:
                    fav = hi_ if d > 0 else li
                    adv = li if d > 0 else hi_
                    if (fav - pos.best) * d > 0:
                        pos.best = fav
                    if (adv - pos.worst) * d < 0:
                        pos.worst = adv
                    if pos.flags.multi_stage_exits and not pos.tp1_done:
                        reach = fav - d * fav * HS[i]            # bid high (long) / ask low (short)
                        if (reach - pos.tp1_px) * d >= 0:
                            take_partial(i)
                            if pos is not None and (pos.stop - ci) * d >= 0:
                                # closed back through the new stop after the target was hit
                                exit_fill(i, pos.units, pos.stop, True, pos.stop_kind)
                    if pos is not None and last[i] and sess[i] - pos.entry_sess + 1 >= pos.params.max_hold_days:
                        exit_fill(i, pos.units, ci, True, "time_exit")
                flatten_next = False

            # ---------------- working entry order
            elif pending is not None:
                pr, pp = pending.order.proposal, pending.params
                d = pr.direction
                P = pr.entry_stop * self._ratio
                Lm = pr.entry_limit * self._ratio
                a_sig = pr.atr * self._ratio
                oi, hi_, li, ci = o[i], h[i], l[i], c[i]
                filled = at_open = False
                if not breakers.can_enter and not pending.order.shadow:
                    stats["cancel_halt"] += 1
                    pending = None
                elif (oi - (P + d * pp.anti_chase_atr * a_sig)) * d > 0:
                    stats["cancel_anti_chase"] += 1
                    pending = None
                else:
                    if (oi - P) * d >= 0:                        # trigger elected at the open
                        quote = oi + d * oi * (HS[i] + slip_f)
                        if (Lm - quote) * d >= 0:
                            open_position(i, pending, oi, False)
                            filled = at_open = True
                        else:                                    # resting limit: needs a pullback to it
                            back = (li + li * HS[i]) if d > 0 else (hi_ - hi_ * HS[i])
                            if (Lm - back) * d >= 0:
                                open_position(i, pending, Lm, True)
                                filled = True
                    elif ((hi_ if d > 0 else li) - P) * d >= 0:  # triggered intrabar
                        open_position(i, pending, P, False)
                        filled = True
                    if filled:
                        pending = None
                        ps = pos
                        adv = li if d > 0 else hi_
                        fav = hi_ if d > 0 else li
                        touched = (ps.stop - adv) * d >= 0
                        closed_through = (ps.stop - ci) * d >= 0
                        if (touched and (at_open or cfg.entry_bar_stop_policy == "conservative")) or closed_through:
                            ps.worst = ps.stop
                            exit_fill(i, ps.units, ps.stop, True, "stop")
                        else:
                            if (fav - ps.best) * d > 0:
                                ps.best = fav
                            ps.worst = adv if at_open else (ci if (ci - ps.entry_px) * d < 0 else ps.entry_px)
                            if ps.flags.multi_stage_exits:
                                reach = fav - d * fav * HS[i]
                                if (reach - ps.tp1_px) * d >= 0:
                                    take_partial(i)
                                    if pos is not None and (pos.stop - ci) * d >= 0:
                                        exit_fill(i, pos.units, pos.stop, True, pos.stop_kind)
                            if pos is not None and last[i] and pos.params.max_hold_days <= 1:
                                exit_fill(i, pos.units, ci, True, "time_exit")
                    elif i - pending.placed >= pp.order_ttl_bars:
                        stats["expired"] += 1
                        pending = None

            # ---------------- bar close: trailing stop, equity, breakers
            if pos is not None and pos.flags.multi_stage_exits:
                d = pos.direction
                if (pos.best - pos.entry_px) * d >= pos.params.trail_activation_r * pos.r_px:
                    trail = pos.best - d * pos.params.trail_atr_mult * atr_arr[i]
                    if (trail - pos.stop) * d > 0:
                        pos.stop, pos.stop_kind = trail, "trail_stop"
            eq = equity_now(i)
            if eq > peak_bar:
                peak_bar = eq
            elif 1.0 - eq / peak_bar > max_dd_bar:
                max_dd_bar = 1.0 - eq / peak_bar
            if breakers.on_bar_close(eq, e.ts[i]) and pos is not None and not pos.shadow:
                flatten_next = True
            if last[i]:
                eq_dates.append(sess[i])
                eq_vals.append(eq)
                breakers.on_session_close(eq)

            # ---------------- new signal at the close of bar i (order for bar i+1)
            if pos is None and pending is None and direction[i] != 0 and i + 1 < i1:
                if last[i] and not cfg.allow_overnight_entry_orders:
                    continue
                if breakers.halted_today and breakers.live:
                    continue
                proposal = e.evaluate(prep, i)
                seg = seg_l[i]
                res = risk_engines[seg].approve(proposal, account_state())
                if isinstance(res, RiskRejection):
                    stats["rejections"][res.reason] = stats["rejections"].get(res.reason, 0) + 1
                else:
                    stats["orders"] += 1
                    pending = _Pending(res, i, seg_params[seg], flags, seg)

        # ---- end of window: mark any open position out at the last close
        if pos is not None and i1 > i0:
            j = i1 - 1
            exit_fill(j, pos.units, c[j], True, "end_of_window")
            if eq_vals:
                eq_vals[-1] = cash

        equity = pd.Series(eq_vals, index=e.sessions[eq_dates].rename("session"), name="equity", dtype=float)
        # sessions skipped by the idle fast-path keep the previous equity
        full_idx = e.sessions[s_lo:s_hi + 1]
        equity = equity.reindex(full_idx).ffill().fillna(init_eq)
        tdf = pd.DataFrame(trades, columns=TRADE_COLUMNS + ["entry_session"]) if trades else pd.DataFrame(
            columns=TRADE_COLUMNS + ["entry_session"]
        )
        if len(tdf):
            tdf["shadow"] = tdf["shadow"].astype(bool)
        stats.update(
            max_dd_bar=max_dd_bar, breaker_events=list(breakers.events), funnel=dict(prep.funnel),
            n_pauses=sum(1 for ev in breakers.events if ev[1] == "pause"),
            deactivated_at=next((ev[0] for ev in breakers.events if ev[1] == "deactivate"), None),
            true_peak=breakers.true_peak, instrument=inst.name,
            n_sessions=len(equity), source=e.features.source, cost_stress=costs.stress,
        )
        return BacktestResult(equity=equity, trades=tdf, initial_equity=init_eq, stats=stats)


def _combine(preps: list[PreparedSignals], seg_of_bar: np.ndarray) -> PreparedSignals:
    """Stitch per-segment prepared arrays into one (walk-forward schedule).
    Bars before the first segment carry no signal."""
    base = preps[0]
    out = {}
    for name in ("direction", "entry_stop", "entry_limit", "stop_price", "d_stop", "atr", "score", "m_reg",
                 "m_hurst", "m_liq", "sigma_1d", "es99", "mu_h", "sigma_h", "z_h", "spread_rel", "exit_long",
                 "exit_short"):
        arr = getattr(base, name).copy()
        for k in range(1, len(preps)):
            m = seg_of_bar == k
            arr[m] = getattr(preps[k], name)[m]
        if name == "direction":
            arr[seg_of_bar < 0] = 0
        out[name] = arr
    funnel = {k: sum(p.funnel.get(k, 0) for p in preps) for k in base.funnel}
    funnel["note"] = "summed over schedule segments (full-sample counts per segment)"
    return PreparedSignals(params=base.params, flags=base.flags, funnel=funnel, **out)
