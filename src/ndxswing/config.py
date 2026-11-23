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
    #: "pnl"   - P&L accrues in the quote currency and is converted (CFD);
    #: "price" - the instrument itself is priced in the account currency and
    #:           carries the FX move on its full notional (unhedged EUR ETF).
    fx_mode: str = "pnl"
    #: Price level at which ``lot_step`` / ``min_lot`` apply.  When set, the
    #: effective step at price P is ``lot_step * reference / P`` so that one lot
    #: is the same NOTIONAL throughout history.  Without it a 2012 backtest at
    #: index 2,300 sees lots eleven times finer, relative to a fixed 5,000
    #: account, than the lots that account can actually trade today.
    lot_reference_price: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("cfd", "etf"):
            raise ValueError(f"unknown instrument kind {self.kind!r}")
        if self.fx_mode not in ("pnl", "price"):
            raise ValueError(f"unknown fx_mode {self.fx_mode!r}")
        if self.lot_step <= 0 or self.min_lot <= 0 or self.multiplier <= 0:
            raise ValueError("lot_step, min_lot and multiplier must be positive")


def ndx_fractional_cfd(lot_step: float = 0.01, lot_reference_price: float | None = None) -> InstrumentSpec:
    """Micro-lot NASDAQ-100 index CFD, USD 1 per point per unit, as offered by
    EU retail CFD brokers with an API (not IBKR).

    ASSUMPTION: lot step / minimum 0.01 index units (some platforms use 0.1,
    which sizes to zero at 0.25% risk with the index near 25,000 - see README).
    """
    return InstrumentSpec(
        name=f"NDX-microCFD-{lot_step:g}", kind="cfd", multiplier=1.0, lot_step=lot_step, min_lot=lot_step,
        margin_rate=0.05, allow_short=True, pays_financing=True, lot_reference_price=lot_reference_price,
    )


def ibkr_ibust100(lot_reference_price: float | None = None) -> InstrumentSpec:
    """IBKR US Tech 100 index CFD (IBUST100): multiplier 1 x index, minimum
    order ONE whole unit (IBKR index CFD schedule, checked Nov 2026).  One unit
    at index 25,000 is about 4.7x a EUR 5,000 account, so it sizes to zero."""
    return InstrumentSpec(
        name="IBKR-IBUST100", kind="cfd", multiplier=1.0, lot_step=1.0, min_lot=1.0, margin_rate=0.05,
        allow_short=True, pays_financing=True, lot_reference_price=lot_reference_price,
    )


def ucits_ndx_etf(price_ratio: float = 0.0022, lot_reference_price: float | None = None) -> InstrumentSpec:
    """Unleveraged EUR-quoted UCITS NASDAQ-100 ETF: long only, cash funded,
    whole shares, no financing, unhedged (carries EURUSD on the notional).

    ASSUMPTION: ``price_ratio`` (share price in USD-equivalent / index level)
    is a placeholder (about USD 55 per share at index 25,000) and must be set
    from the actual share class.  Trading through the US close needs a venue
    with retail hours to 22:00 CET (Tradegate, gettex, or Xetra's extended
    retail service since December 2025, broker permitting); stops are assumed
    to be managed client-side and executed as market orders.
    """
    return InstrumentSpec(
        name="NDX-UCITS-ETF", kind="etf", quote_ccy="USD", multiplier=1.0, price_ratio=price_ratio,
        lot_step=1.0, min_lot=1.0, margin_rate=1.0, allow_short=False, pays_financing=False, fx_mode="price",
        lot_reference_price=None if lot_reference_price is None else lot_reference_price * price_ratio,
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
    #: full quoted spread in instrument PRICE POINTS; overrides spread_bps when set
    spread_points: float | None = None
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

    def half_spread_at(self, price):  # noqa: ANN001, ANN201 - float or ndarray
        """Half spread as a fraction of ``price`` (points-based when configured)."""
        if self.spread_points is not None:
            return 0.5 * self.spread_points * self.stress / price
        return self.half_spread_frac + 0.0 * price

    @property
    def slippage_frac(self) -> float:
        return self.slippage_bps * 1e-4 * self.stress

    def commission(self, notional_acct: float) -> float:
        """Commission for one order of ``notional_acct`` (account currency)."""
        if notional_acct <= 0:
            return 0.0
        return max(self.commission_min, notional_acct * self.commission_bps * 1e-4) * self.stress


def micro_cfd_costs() -> CostModel:
    """ASSUMPTION (schedule supplied for the EU API CFD brokers, not verified
    against a published rate card): 1.2 index points spread, no ticket fee,
    financing at the USD overnight rate +/- 2.5% p.a., ACT/360."""
    return CostModel(spread_points=1.2, slippage_bps=1.0, commission_bps=0.0, commission_min=0.0,
                     financing_markup=0.025)


def ibkr_cfd_costs() -> CostModel:
    """IBKR index CFD schedule: 0.010% per trade, USD 1.00 minimum per order
    (booked here as 1.00 in account currency), financing benchmark +/- 1.5%.
    ASSUMPTION: 1.0 index point quoted spread."""
    return CostModel(spread_points=1.0, slippage_bps=1.0, commission_bps=1.0, commission_min=1.0,
                     financing_markup=0.015)


def ucits_etf_costs() -> CostModel:
    """ASSUMPTION: 6 bp quoted spread in extended retail hours, 2 bp slippage
    on client-side stop/market orders, EUR 1.00 flat ticket, no financing."""
    return CostModel(spread_bps=6.0, slippage_bps=2.0, commission_bps=0.0, commission_min=1.0, financing_markup=0.0)


# --------------------------------------------------------------------------- #
# Broker presets
# --------------------------------------------------------------------------- #
#
# Only brokers that offer EU retail clients both a fractional index CFD and a
# programmatic API are useful for a EUR 5,000 account:
#
# * OANDA is NOT one of them: OANDA's v20 REST API documentation excludes the
#   EU entity (OANDA TMS Brokers S.A.), so there is no API for EU retail
#   accounts.
# * IBKR's IBUST100 trades in whole units of 1 x index and sizes to zero.
#
# The two CFD presets below carry the schedule supplied for this project.
# Nothing in them has been confirmed against the brokers' instrument
# specifications; see ``BrokerPreset.note`` for what a public page showed.


def pepperstone_eu_ctrader(lot_step: float = 0.01, lot_reference_price: float | None = None) -> InstrumentSpec:
    """Pepperstone EU via the cTrader Open API: USD 1 per point per lot."""
    return replace(ndx_fractional_cfd(lot_step, lot_reference_price), name=f"Pepperstone-EU-cTrader-NAS100-{lot_step:g}")


def capital_com_eu_rest(lot_step: float = 0.10, lot_reference_price: float | None = None) -> InstrumentSpec:
    """Capital.com EU via its REST API: USD 1 per point per contract."""
    return replace(ndx_fractional_cfd(lot_step, lot_reference_price), name=f"Capital.com-EU-REST-US100-{lot_step:g}")


@dataclass(frozen=True)
class BrokerPreset:
    instrument: InstrumentSpec
    costs: CostModel
    note: str


def broker_preset(name: str, lot_step: float | None = None, lot_reference_price: float | None = None,
                  etf_price_ratio: float = 0.0022) -> BrokerPreset:
    """Instrument + cost schedule for one execution venue.  Financing is the
    USD overnight rate (EFFR / SOFR) plus or minus the markup, ACT/360."""
    if name == "pepperstone_eu_ctrader":
        return BrokerPreset(
            pepperstone_eu_ctrader(0.01 if lot_step is None else lot_step, lot_reference_price), micro_cfd_costs(),
            "Supplied schedule: 1.2 pt spread, no index commission, ARR +/- 2.5%, minimum 0.01 lot. Pepperstone "
            "states it charges no commission on index CFDs and advertises NAS100 spreads from about 1.0 pt; the "
            "only minimum-size figure found (a third-party page) says 0.1 lot. Confirm the minimum in the cTrader "
            "symbol specification and rerun with --lot-step 0.1 if it applies.")
    if name == "capital_com_eu_rest":
        return BrokerPreset(
            capital_com_eu_rest(0.10 if lot_step is None else lot_step, lot_reference_price), micro_cfd_costs(),
            "Supplied schedule: 1.2 pt spread, no commission, ARR +/- 2.5%, minimum and step 0.10. Capital.com's own "
            "EU market page lists a minimum traded quantity of 0.001 and a long overnight adjustment of -0.02226% "
            "of position value per day, about 8.0% a year or roughly the overnight rate + 4.1%. To model the page "
            "instead: --lot-step 0.001 --financing-markup 0.041. With a 0.10 lot, 0.25% risk is clipped to zero "
            "for typical stops; --b-risk 0.0035 is the smallest budget that carries a 1.5 ATR stop of about 135 "
            "points.")
    if name == "ibkr_cfd":
        return BrokerPreset(ibkr_ibust100(lot_reference_price), ibkr_cfd_costs(),
                            "IBKR schedule (0.010%, USD 1.00 minimum, benchmark +/- 1.5%, whole units) checked "
                            "Nov 2026; 1.0 pt spread assumed. Sizes to zero on EUR 5,000.")
    if name == "ucits_etf":
        return BrokerPreset(ucits_ndx_etf(etf_price_ratio, lot_reference_price), ucits_etf_costs(),
                            "Assumptions: 6 bp spread, 2 bp slippage, EUR 1.00 ticket, no financing, long only, "
                            "client-side stops. Set the price ratio for the actual share class.")
    raise ValueError(f"unknown broker preset {name!r}; choose from {BROKER_PRESETS}")


BROKER_PRESETS: tuple[str, ...] = ("pepperstone_eu_ctrader", "capital_com_eu_rest", "ibkr_cfd", "ucits_etf")


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


#: Ablation order.  The baseline has every flag off.  The Hurst gate and the
#: ARMA Z filter are optional challengers and come last: both are expected to
#: starve the trade count and must not sit in front of the modules that are
#: needed for sizing.
MODULE_ORDER: tuple[str, ...] = (
    "multi_stage_exits",   # +1.5R partial, breakeven, +2R chandelier
    "trend_flip_exit",     # exit when daily EMA20/50 flips against the trade
    "regime_gate",         # causal Markov regime gate, M_reg and stress exit
    "garch_vol",           # GJR-GARCH-t sigma_1d in the vol target + Ljung-Box gate
    "evt_es_cap",          # EVT/GPD ES_0.99 exposure cap
    "spread_gate",         # relative spread gate (needs bid/ask history)
    "vpin_filter",         # volume-clock VPIN toxicity filter
    "event_blackout",      # tier-1 macro / earnings blackout
    "score_sizing",        # risk scaled by (0.5 + 0.5 * S_t)
    "kelly_cap",           # exact concave LCB fractional Kelly exposure cap
    "hurst_gate",          # challenger: DFA Hurst gate, M_Hurst
    "arma_z_filter",       # challenger: ARMA Z_{t,H} hurdle + net expectancy gate
)


@dataclass(frozen=True)
class ModuleFlags:
    multi_stage_exits: bool = False
    trend_flip_exit: bool = False
    regime_gate: bool = False
    regime_model: str = "hmm3"  # "hmm3" | "hmm2" | "rv20"
    hurst_gate: bool = False
    hurst_mode: str = "lcb"     # "lcb": H >= min AND bootstrap LCB > 0.50 (spec); "point": H >= min only
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
        if self.hurst_mode not in ("lcb", "point"):
            raise ValueError(f"unknown hurst_mode {self.hurst_mode!r}")

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
    #: An 8% pause ends after ``pause_cooldown_days`` sessions or at the next
    #: walk-forward refit (backtests only; live trading never resumes on its
    #: own).  The -10% deactivation is measured against the all-time
    #: high-water mark, is unconditional and is TERMINAL inside a backtest.
    pause_cooldown_days: int = 21


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
