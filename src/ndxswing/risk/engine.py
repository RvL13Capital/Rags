"""Layer 5: independent position sizing and portfolio risk firewall.

The engine receives an immutable :class:`SignalProposal` and the account
state and either returns a :class:`RiskApprovedOrder` or a
:class:`RiskRejection`.  Size is always floored to the lot step; it is never
rounded up and a size below the minimum lot is a rejection.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import CostModel, InstrumentSpec, ModuleFlags, RiskLimits, StrategyParams
from ..signal.engine import SignalProposal
from .kelly import exact_concave_fractional_kelly

SQRT252 = math.sqrt(252.0)


@dataclass(frozen=True)
class AccountState:
    equity: float                    # account currency, marked to market
    fx_quote_to_acct: float          # X_{USD->EUR} = 1 / EURUSD for a USD instrument and EUR account
    m_dd: float = 1.0                # drawdown risk multiplier from the circuit breakers
    leverage_cap: float = 1.5        # gross leverage cap currently in force
    open_exposure: float = 0.0       # existing gross exposure in account currency
    trade_returns: tuple[float, ...] = ()   # unleveraged net simple returns of closed trades
    live: bool = True                # False while paused / deactivated (paper diagnostics only)


@dataclass(frozen=True)
class RiskApprovedOrder:
    proposal: SignalProposal
    units: float
    risk_budget: float               # R_eff in account currency
    planned_risk: float              # units * (D_eff per unit + costs)
    notional: float                  # account currency at the entry trigger
    leverage: float
    binding: str                     # which constraint set the size
    shadow: bool                     # paper-only order, never sent to the broker
    kelly_f: float
    lot_step: float = 0.0            # effective lot step / minimum for this order (price-level normalised)
    min_lot: float = 0.0


@dataclass(frozen=True)
class RiskRejection:
    reason: str


def floor_to_step(raw: float, step: float) -> float:
    """Largest multiple of ``step`` not exceeding ``raw`` (tolerant of float noise)."""
    if not math.isfinite(raw) or raw <= 0:
        return 0.0
    return round(math.floor(raw / step + 1e-9) * step, 10)


class PortfolioRiskEngine:
    def __init__(self, params: StrategyParams, flags: ModuleFlags, instrument: InstrumentSpec, costs: CostModel,
                 shadow_trading: bool = True) -> None:
        self.p, self.f, self.inst, self.costs = params, flags, instrument, costs
        self.shadow_trading = shadow_trading

    def approve(self, proposal: SignalProposal, state: AccountState) -> RiskApprovedOrder | RiskRejection:
        p, inst, c = self.p, self.inst, self.costs
        if proposal.direction < 0 and not inst.allow_short:
            return RiskRejection("short_not_allowed")
        V = state.equity
        if V <= 0:
            return RiskRejection("no_equity")
        if not (proposal.sigma_1d > 0 and proposal.d_stop > 0 and proposal.entry_stop > 0):
            return RiskRejection("invalid_proposal")

        shadow = not state.live
        vehicle_cap = inst.max_leverage if inst.max_leverage is not None else math.inf
        m_dd, lev_cap = state.m_dd, min(p.max_leverage, state.leverage_cap, vehicle_cap)
        if shadow:
            if not self.shadow_trading:
                return RiskRejection("not_live")
            m_dd, lev_cap = 1.0, min(p.max_leverage, vehicle_cap)  # would-be baseline size for diagnostics

        kelly_f = math.nan
        kelly_cap = math.inf
        if self.f.kelly_cap:
            k = exact_concave_fractional_kelly(np.asarray(state.trade_returns[-p.kelly_window:]), p.kelly_fraction,
                                               min_trades=p.kelly_min_trades)
            if k.active:
                kelly_f = k.f_kelly
                if k.f_kelly <= 0.0:
                    if not self.shadow_trading:
                        return RiskRejection("kelly_zero")
                    shadow = True      # edge not proven: paper trade only
                else:
                    kelly_cap = k.f_kelly * V

        score_mult = 0.5 + 0.5 * proposal.score if self.f.score_sizing else 1.0
        r_eff = V * p.b_risk * proposal.m_reg * proposal.m_hurst * proposal.m_liq * m_dd * score_mult
        if r_eff <= 0:
            return RiskRejection("zero_risk_budget")

        caps = {
            "vol_target": V * (p.vol_target_ann / SQRT252) / proposal.sigma_1d,
            "max_leverage": lev_cap * V,
        }
        if self.f.evt_es_cap:
            if not (proposal.es99_daily > 0):
                return RiskRejection("es_unavailable")
            caps["es_cap"] = (p.es_cap_mult * p.b_risk / proposal.es99_daily) * V
        if not shadow and math.isfinite(kelly_cap):
            caps["kelly"] = kelly_cap
        e_name = min(caps, key=caps.get)
        e_allowed = max(0.0, caps[e_name] - state.open_exposure)

        X = state.fx_quote_to_acct
        px = proposal.entry_stop * inst.price_ratio
        unit_notional = px * inst.multiplier * X
        spread = proposal.spread_rel * c.stress if math.isfinite(proposal.spread_rel) else 2.0 * c.half_spread_at(px)
        var_cost_unit = unit_notional * (spread + 2.0 * c.slippage_frac + 2.0 * c.commission_bps * 1e-4 * c.stress)
        fixed_cost = 2.0 * c.commission_min * c.stress  # conservative: ticket minimum on both legs
        d_eff = p.gap_buffer * proposal.d_stop * inst.price_ratio
        risk_unit = d_eff * inst.multiplier * X + var_cost_unit

        # Leveraged margin products may commit at most 50% of equity as initial
        # margin (they face the ESMA 50% margin close-out).  A fully cash-backed
        # vehicle (margin_rate >= 1) cannot be closed out on margin and may use
        # up to 95% of equity.
        usage_cap = p.margin_usage_cap if inst.margin_rate < 1.0 else p.cash_backed_usage_cap
        cands = {
            "risk_budget": (r_eff - fixed_cost) / risk_unit,
            e_name: e_allowed / unit_notional,
            "margin": usage_cap * V / (unit_notional * inst.margin_rate),
        }
        binding = min(cands, key=cands.get)
        lot_scale = inst.lot_reference_price / px if inst.lot_reference_price else 1.0
        lot_step, min_lot = inst.lot_step * lot_scale, inst.min_lot * lot_scale
        units = floor_to_step(cands[binding], lot_step)
        if units < min_lot * (1.0 - 1e-9):
            return RiskRejection("below_min_lot")
        notional = units * unit_notional
        return RiskApprovedOrder(
            proposal=proposal, units=units, risk_budget=r_eff, planned_risk=units * risk_unit + fixed_cost,
            notional=notional, leverage=notional / V, binding=binding, shadow=shadow, kelly_f=kelly_f,
            lot_step=lot_step, min_lot=min_lot,
        )


# --------------------------------------------------------------------------- #
# Multi-tier circuit breakers (6.3)
# --------------------------------------------------------------------------- #


@dataclass
class CircuitBreakers:
    """Portfolio drawdown state machine.

    Two references are kept apart on purpose:

    * ``true_peak`` - the all-time high-water mark.  It is NEVER re-based.
      The risk tiers (-4% / -6%) and the hard deactivation at
      ``equity <= (1 - dd_deactivate) * true_peak`` are measured against it,
      so a system that resumes after a pause still trades at reduced risk
      until it has actually recovered, and cumulative losses can never step
      down past -10% of the peak.  Deactivation is terminal within a run:
      restarting needs re-estimation and re-validation, which is a decision
      taken outside the simulation.
    * ``pause_ref`` - the reference for the -8% pause.  It is re-based to
      current equity when a pause ends (cool-down or walk-forward refit);
      otherwise a flat, paused system could never trade again.
    """

    limits: RiskLimits
    initial_equity: float
    pause_ref: float = field(init=False)
    true_peak: float = field(init=False)
    session_open_equity: float = field(init=False)
    halted_today: bool = False
    paused: bool = False
    deactivated: bool = False
    resume_in: int = 0
    events: list = field(default_factory=list)
    _closes: deque = field(default_factory=lambda: deque(maxlen=6))
    _m_dd: float = 1.0
    _lev_cap: float = math.inf

    #: thresholds are inclusive ("drawdown >= 8%"); guard against 1 - 4600/5000 = 0.07999999999999996
    EPS = 1e-9

    def __post_init__(self) -> None:
        self.pause_ref = self.true_peak = self.session_open_equity = self.initial_equity
        self._closes.append(self.initial_equity)

    # -- queries
    @property
    def live(self) -> bool:
        return not (self.paused or self.deactivated)

    @property
    def can_enter(self) -> bool:
        return self.live and not self.halted_today

    @property
    def m_dd(self) -> float:
        return self._m_dd

    @property
    def leverage_cap(self) -> float:
        return self._lev_cap

    @property
    def hard_kill_equity(self) -> float:
        return (1.0 - self.limits.dd_deactivate) * self.true_peak

    # -- transitions
    def _resume(self, equity: float, why: str, stamp: object) -> None:
        self.paused = False
        self.resume_in = 0
        self.pause_ref = equity
        self._closes.clear()
        self._closes.append(equity)
        self.events.append((stamp, f"resume:{why}", equity))

    def on_refit(self, equity: float, stamp: object = None) -> None:
        """Walk-forward re-estimation boundary: a PAUSED system may restart.
        A deactivated system stays off."""
        if self.paused and not self.deactivated:
            self._resume(equity, "refit", stamp)
            self._update_tiers(equity)

    def on_session_open(self, equity: float, stamp: object = None) -> None:
        self.halted_today = False
        self.session_open_equity = equity
        if self.paused and not self.deactivated:
            self.resume_in -= 1
            if self.resume_in <= 0:
                self._resume(equity, "cooldown", stamp)
        self._update_tiers(equity)

    def on_session_close(self, equity: float) -> None:
        self._closes.append(equity)

    def _update_tiers(self, equity: float) -> None:
        L = self.limits
        dd = 1.0 - equity / self.true_peak if self.true_peak > 0 else 1.0
        m, cap = 1.0, math.inf
        if dd >= L.dd_tier2 - self.EPS:
            m = 0.25
            cap = L.tier1_leverage_cap
        elif dd >= L.dd_tier1 - self.EPS:
            m, cap = 0.75, L.tier1_leverage_cap
        # rolling loss over the last five completed sessions plus the current one
        base = self._closes[0]
        if base > 0 and 1.0 - equity / base >= L.rolling5_loss_halve - self.EPS:
            m = min(m, 0.50)
        self._m_dd, self._lev_cap = m, cap

    def on_bar_close(self, equity: float, stamp: object = None) -> bool:
        """Update with marked-to-market equity.  Returns True when open
        positions must be flattened (pause or deactivation just triggered)."""
        L = self.limits
        self.true_peak = max(self.true_peak, equity)
        if self.live:
            self.pause_ref = max(self.pause_ref, equity)
        flatten = False
        if not self.deactivated and equity <= self.hard_kill_equity * (1.0 + self.EPS):
            self.deactivated, flatten = True, True
            self.events.append((stamp, "deactivate", equity))
        elif self.live and 1.0 - equity / self.pause_ref >= L.dd_pause - self.EPS:
            self.paused, self.resume_in, flatten = True, L.pause_cooldown_days, True
            self.events.append((stamp, "pause", equity))
        if not self.halted_today and self.session_open_equity > 0:
            if 1.0 - equity / self.session_open_equity >= L.intraday_loss_halt - self.EPS:
                self.halted_today = True
                self.events.append((stamp, "intraday_halt", equity))
        self._update_tiers(equity)
        return flatten
