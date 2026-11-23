"""Immutable configuration objects.

Every number that is a broker/market assumption (spread, slippage, financing
markup, lot step, margin) lives here so that it can be audited and stressed in
one place.  Defaults marked ``ASSUMPTION`` must be replaced by the actual broker
schedule before any result is used for a capital decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

# --------------------------------------------------------------------------- #
# Instrument & costs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InstrumentSpec:
    """Tradable vehicle.

    ``multiplier`` is the quote-currency P&L of one unit for a one point move of
    the *instrument* price; ``price_ratio`` maps the signal series onto the
    instrument price (1.0 for an index CFD quoted in index points).
    """

    name: str
    kind: str  # "cfd" | "etf"
    quote_ccy: str = "USD"
    multiplier: float = 1.0
    price_ratio: float = 1.0
    lot_step: float = 0.01
    min_lot: float = 0.01
    margin_rate: float = 0.05  # ESMA retail limit 20:1 on major indices
    allow_short: bool = True
    pays_financing: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("cfd", "etf"):
            raise ValueError(f"unknown instrument kind {self.kind!r}")
        if self.lot_step <= 0 or self.min_lot <= 0 or self.multiplier <= 0:
            raise ValueError("lot_step, min_lot and multiplier must be positive")


def ndx_fractional_cfd(lot_step: float = 0.01) -> InstrumentSpec:
    """Regulated fractional NASDAQ-100 index CFD, USD 1 per point per unit.

    ASSUMPTION: lot step / minimum 0.01 index units.  Verify with the broker.
    """
    return InstrumentSpec(
        name="NDX-CFD", kind="cfd", multiplier=1.0, lot_step=lot_step, min_lot=lot_step,
        margin_rate=0.05, allow_short=True, pays_financing=True,
    )


def ucits_ndx_etf(price_ratio: float = 0.04) -> InstrumentSpec:
    """Unleveraged UCITS NASDAQ-100 ETF (long only, cash funded, whole shares).

    ASSUMPTION: ``price_ratio`` (ETF price / index level) is a placeholder scale
    and must be set from the actual share class.  Note that a Xetra listing
    only overlaps the US regular session for about two hours, so stops placed
    on US-afternoon bars cannot be worked there.
    """
    return InstrumentSpec(
        name="NDX-UCITS-ETF", kind="etf", multiplier=1.0, price_ratio=price_ratio,
        lot_step=1.0, min_lot=1.0, margin_rate=1.0, allow_short=False, pays_financing=False,
    )


@dataclass(frozen=True)
class CostModel:
    """Execution and carry costs.  All defaults are ASSUMPTIONS to be verified.

    spread_bps            full quoted spread relative to price, used whenever a
                          bar carries no ``spread_rel`` observation
    slippage_bps          adverse slippage per side on stop and market fills
    commission_bps/min    per order, on notional / floor in account currency
    financing_markup      annual spread over (longs) / under (shorts) the benchmark
    stress                multiplies spread, slippage, commission and markup
    """

    spread_bps: float = 1.5
    slippage_bps: float = 1.0
    commission_bps: float = 0.0
    commission_min: float = 0.0
    financing_markup: float = 0.03
    financing_day_count: int = 360
    default_benchmark_rate: float = 0.03
    stress: float = 1.0

    def stressed(self, factor: float) -> "CostModel":
        return replace(self, stress=self.stress * factor)

    @property
    def half_spread_frac(self) -> float:
        return 0.5 * self.spread_bps * 1e-4 * self.stress

    @property
    def slippage_frac(self) -> float:
        return self.slippage_bps * 1e-4 * self.stress

    def commission(self, notional_acct: float) -> float:
        """Commission for one order of ``notional_acct`` (account currency)."""
        if notional_acct <= 0:
            return 0.0
        return max(self.commission_min, notional_acct * self.commission_bps * 1e-4) * self.stress


# --------------------------------------------------------------------------- #
# Strategy parameters and module switches
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyParams:
    # --- entry structure (section 5.2 / 5.4)
    ema_fast: int = 20
    ema_slow: int = 50
    donchian_window: int = 20
    atr_window: int = 14
    entry_buffer_atr: float = 0.10
    limit_offset_atr: float = 0.15
    anti_chase_atr: float = 0.50
    order_ttl_bars: int = 2
    # --- stop & exits
    atr_stop_mult: float = 1.50
    struct_lookback: int = 5
    max_stop_ratio: float = 1.50  # skip if D_stop > ratio * atr_stop_mult * ATR (2.25 ATR at defaults)
    tp1_r: float = 1.50
    tp1_fraction: float = 0.50
    trail_activation_r: float = 2.00
    trail_atr_mult: float = 1.50
    max_hold_days: int = 5
    # --- regime gate (4.A)
    p_trend_min: float = 0.65
    p_trend_full: float = 0.75
    p_stress_max: float = 0.20
    p_stress_full: float = 0.10
    conf_min: float = 0.35
    stress_exit: float = 0.30
    rv_filter_pct: float = 0.80  # deterministic RV20 benchmark: eligible below this percentile
    rv_stress_pct: float = 0.95  # ... and treated as stress (exit) above this one
    # --- Hurst gate (4.B)
    hurst_min: float = 0.55
    hurst_strong: float = 0.60
    hurst_lcb_min: float = 0.50
    # --- ARMA-GARCH (4.C)
    z_hurdle: float = 0.35
    horizon_days: int = 3
    lb_pvalue_min: float = 0.05
    # --- liquidity (4.D)
    spread_ok_mult: float = 1.5   # 5.1 gate 6: spread <= 1.5 x trailing median, else NO_TRADE
    vpin_halve_pct: float = 0.90
    vpin_block_pct: float = 0.95
    # --- expectancy hurdle (5.3)
    expectancy_r_hurdle: float = 0.25
    min_reward_risk: float = 1.50
    expected_hold_days: float = 3.0
    # --- sizing (6)
    b_risk: float = 0.0025
    vol_target_ann: float = 0.11
    max_leverage: float = 1.50
    gap_buffer: float = 1.25
    es_cap_mult: float = 2.5
    margin_usage_cap: float = 0.50
    kelly_fraction: float = 0.20
    kelly_min_trades: int = 30
    kelly_window: int = 250

    def __post_init__(self) -> None:
        if not 0.10 <= self.kelly_fraction <= 0.25:
            raise ValueError("kelly_fraction must lie in [0.10, 0.25]")
        if self.b_risk > 0.0075 + 1e-12:
            raise ValueError("b_risk above the 0.75% hard ceiling")
        if self.max_leverage > 1.5 + 1e-12:
            raise ValueError("max_leverage above the 1.5x hard cap")
        if not 1 <= self.max_hold_days <= 5:
            raise ValueError("max_hold_days must be within 1..5 trading days")
        if self.horizon_days not in (1, 2, 3, 5):
            raise ValueError("horizon_days must be one of 1, 2, 3, 5")

    def with_(self, **kw: Any) -> "StrategyParams":
        return replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Ablation order.  The baseline has every flag off.
MODULE_ORDER: tuple[str, ...] = (
    "multi_stage_exits",   # +1.5R partial, breakeven, +2R chandelier
    "trend_flip_exit",     # exit when daily EMA20/50 flips against the trade
    "regime_gate",         # causal Markov regime gate, M_reg and stress exit
    "hurst_gate",          # DFA Hurst + bootstrap LCB gate, M_Hurst
    "garch_vol",           # GJR-GARCH-t sigma_1d in the vol target + Ljung-Box gate
    "evt_es_cap",          # EVT/GPD ES_0.99 exposure cap
    "arma_z_filter",       # ARMA Z_{t,H} directional hurdle + net expectancy gate
    "spread_gate",         # relative spread gate (needs bid/ask history)
    "vpin_filter",         # volume-clock VPIN toxicity filter
    "event_blackout",      # tier-1 macro / earnings blackout
    "score_sizing",        # risk scaled by (0.5 + 0.5 * S_t)
    "kelly_cap",           # exact concave LCB fractional Kelly exposure cap
)


@dataclass(frozen=True)
class ModuleFlags:
    multi_stage_exits: bool = False
    trend_flip_exit: bool = False
    regime_gate: bool = False
    regime_model: str = "hmm3"  # "hmm3" | "hmm2" | "rv20"
    hurst_gate: bool = False
    garch_vol: bool = False
    evt_es_cap: bool = False
    arma_z_filter: bool = False
    spread_gate: bool = False
    vpin_filter: bool = False
    event_blackout: bool = False
    score_sizing: bool = False
    kelly_cap: bool = False

    def __post_init__(self) -> None:
        if self.regime_model not in ("hmm3", "hmm2", "rv20"):
            raise ValueError(f"unknown regime_model {self.regime_model!r}")

    def with_(self, **kw: Any) -> "ModuleFlags":
        return replace(self, **kw)

    def enabled(self) -> list[str]:
        return [m for m in MODULE_ORDER if getattr(self, m)]

    @classmethod
    def baseline(cls) -> "ModuleFlags":
        return cls()

    @classmethod
    def full(cls, **kw: Any) -> "ModuleFlags":
        return cls(**{m: True for m in MODULE_ORDER} | kw)


# --------------------------------------------------------------------------- #
# Portfolio risk limits (6.3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RiskLimits:
    intraday_loss_halt: float = 0.01
    rolling5_loss_halve: float = 0.02
    dd_tier1: float = 0.04   # M_DD 0.75, gross leverage <= 1.0x
    dd_tier2: float = 0.06   # M_DD 0.25
    dd_pause: float = 0.08   # flatten, paper diagnostics
    dd_deactivate: float = 0.10
    tier1_leverage_cap: float = 1.0
    #: Research-mode resume policy.  Live trading never resumes automatically.
    #: In a backtest a paused system resumes after ``pause_cooldown_days``
    #: sessions (or at the next walk-forward refit) with the drawdown reference
    #: re-based to current equity; a deactivated system resumes only at a refit.
    pause_cooldown_days: int = 21
    deactivate_cooldown_days: int = 63


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 5_000.0
    account_ccy: str = "EUR"
    instrument: InstrumentSpec = field(default_factory=ndx_fractional_cfd)
    costs: CostModel = field(default_factory=CostModel)
    limits: RiskLimits = field(default_factory=RiskLimits)
    #: "conservative": a bar that fills the entry and trades through the stop
    #: is booked as a stop-out; "close": only when the bar closes beyond it.
    entry_bar_stop_policy: str = "conservative"
    allow_overnight_entry_orders: bool = True
    #: When the Kelly cap or a pause sets size to zero the trade is simulated
    #: off-book ("paper diagnostics") so that the edge estimate keeps updating.
    shadow_trading: bool = True

    def __post_init__(self) -> None:
        if self.entry_bar_stop_policy not in ("conservative", "close"):
            raise ValueError("entry_bar_stop_policy must be 'conservative' or 'close'")
