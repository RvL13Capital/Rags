"""Risk firewall: sizing formula, discrete floor, FX orientation, Kelly, EVT, breakers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from ndxswing.config import CostModel, InstrumentSpec, ModuleFlags, RiskLimits, StrategyParams, ndx_fractional_cfd
from ndxswing.risk import tail
from ndxswing.risk.engine import (
    AccountState, CircuitBreakers, PortfolioRiskEngine, RiskApprovedOrder, RiskRejection, floor_to_step,
)
from ndxswing.risk.kelly import exact_concave_fractional_kelly
from ndxswing.signal.engine import SignalProposal

NOCOST = CostModel(spread_bps=0, slippage_bps=0, commission_bps=0, financing_markup=0)


def proposal(**kw) -> SignalProposal:
    base = dict(bar_index=0, ts=pd.Timestamp("2026-01-05", tz="UTC"), direction=1, entry_stop=25_000.0,
                entry_limit=25_015.0, stop_price=24_900.0, d_stop=100.0, atr=66.0, ref_price=24_990.0, score=1.0,
                m_reg=1.0, m_hurst=1.0, m_liq=1.0, sigma_1d=0.012, es99_daily=float("nan"), mu_h=float("nan"),
                sigma_h=float("nan"), z_h=float("nan"), spread_rel=float("nan"))
    return SignalProposal(**(base | kw))


def engine(params=StrategyParams(), flags=ModuleFlags(), inst=None, costs=NOCOST) -> PortfolioRiskEngine:
    if isinstance(params, ModuleFlags):                      # engine(flags) shorthand
        params, flags = StrategyParams(), params
    return PortfolioRiskEngine(params, flags, inst or ndx_fractional_cfd(), costs)


def test_floor_never_rounds_up():
    assert floor_to_step(0.3, 0.1) == pytest.approx(0.3)       # 0.3 / 0.1 = 2.9999999999999996 in floats
    assert floor_to_step(0.2999, 0.1) == pytest.approx(0.2)
    assert floor_to_step(0.0999, 0.1) == 0.0
    assert floor_to_step(float("nan"), 0.1) == 0.0 and floor_to_step(-1.0, 0.1) == 0.0


def test_unit_formula_with_fx_orientation():
    """EUR 5,000, 0.25% risk, 100 pt stop, 1.25x gap buffer, EURUSD 1.08:
    X = 1/1.08 EUR per USD -> 12.5 / (125 * 0.9259) = 0.108 -> 0.10 units."""
    X = 1 / 1.08
    res = engine().approve(proposal(), AccountState(equity=5000.0, fx_quote_to_acct=X))
    assert isinstance(res, RiskApprovedOrder)
    assert res.units == pytest.approx(0.10) and res.binding == "risk_budget"
    assert res.risk_budget == pytest.approx(12.5)
    assert res.planned_risk <= 12.5
    assert res.leverage == pytest.approx(0.10 * 25_000 * X / 5000)
    # using EURUSD the wrong way round (X = 1.08) would have produced a smaller, wrong size
    wrong = engine().approve(proposal(), AccountState(equity=5000.0, fx_quote_to_acct=1.08))
    assert wrong.units == pytest.approx(0.09)


def test_mnq_style_contract_is_blocked_at_5k():
    """One MNQ ($2/pt, whole contracts) with a 100 pt stop risks $200: N must be 0."""
    mnq = InstrumentSpec(name="MNQ", kind="cfd", multiplier=2.0, lot_step=1.0, min_lot=1.0, margin_rate=0.08)
    res = engine(inst=mnq).approve(proposal(), AccountState(equity=5000.0, fx_quote_to_acct=0.93))
    assert isinstance(res, RiskRejection) and res.reason == "below_min_lot"


def test_exposure_caps_bind_in_the_right_order():
    st = AccountState(equity=5000.0, fx_quote_to_acct=1.0)
    tight = proposal(d_stop=5.0, stop_price=24_995.0)                     # tiny stop -> exposure caps bind
    r = engine().approve(tight, st)
    assert r.binding == "vol_target"
    assert r.notional <= 5000 * (0.11 / math.sqrt(252)) / 0.012 + 1e-6
    r = engine().approve(proposal(d_stop=5.0, sigma_1d=0.002), st)        # calm market -> leverage cap
    assert r.binding == "max_leverage" and r.leverage <= 1.5
    r = engine().approve(proposal(d_stop=5.0, sigma_1d=0.002), AccountState(5000.0, 1.0, leverage_cap=1.0))
    assert r.leverage <= 1.0
    es = engine(ModuleFlags(evt_es_cap=True)).approve(proposal(d_stop=5.0, sigma_1d=0.002, es99_daily=0.05), st)
    assert es.binding == "es_cap" and es.notional <= 2.5 * 0.0025 / 0.05 * 5000 + 1e-9
    no_es = engine(ModuleFlags(evt_es_cap=True)).approve(proposal(), st)
    assert isinstance(no_es, RiskRejection) and no_es.reason == "es_unavailable"


def test_multipliers_and_score_scale_the_risk_budget():
    st = AccountState(equity=5000.0, fx_quote_to_acct=1.0, m_dd=0.5)
    r = engine(ModuleFlags(score_sizing=True)).approve(proposal(m_reg=0.75, m_hurst=0.8, m_liq=0.5, score=0.0), st)
    assert isinstance(r, RiskRejection) or r.risk_budget == pytest.approx(12.5 * 0.75 * 0.8 * 0.5 * 0.5 * 0.5)
    r = engine().approve(proposal(m_reg=0.0), st)
    assert isinstance(r, RiskRejection) and r.reason == "zero_risk_budget"
    with pytest.raises(ValueError):
        StrategyParams(b_risk=0.01)
    with pytest.raises(ValueError):
        StrategyParams(max_leverage=2.0)


def test_kelly_gate_in_engine():
    eng = engine(ModuleFlags(kelly_cap=True))
    few = AccountState(5000.0, 1.0, trade_returns=tuple([0.01] * 10))
    assert not eng.approve(proposal(), few).shadow                         # < 30 trades: Kelly bypassed
    losing = AccountState(5000.0, 1.0, trade_returns=tuple(np.random.default_rng(0).normal(-0.002, 0.01, 60)))
    r = eng.approve(proposal(), losing)
    assert r.shadow and r.kelly_f == 0.0                                   # unproven edge -> paper only
    strict = PortfolioRiskEngine(StrategyParams(), ModuleFlags(kelly_cap=True), ndx_fractional_cfd(), NOCOST,
                                 shadow_trading=False)
    assert strict.approve(proposal(), losing).reason == "kelly_zero"
    paused = eng.approve(proposal(), AccountState(5000.0, 1.0, live=False))
    assert paused.shadow


# ----------------------------------------------------------------------- Kelly


def test_kelly_bypass_zero_and_exact_optimum():
    rng = np.random.default_rng(1)
    assert not exact_concave_fractional_kelly(rng.normal(0.01, 0.02, 29)).active
    assert exact_concave_fractional_kelly(rng.normal(-0.001, 0.02, 200)).f_kelly == 0.0
    weak = np.tile([0.0210, -0.0200], 50)                                   # positive mean, LCB below zero
    k = exact_concave_fractional_kelly(weak)
    assert k.mu_hat > 0 >= k.mu_lcb and k.f_kelly == 0.0

    r = np.r_[rng.normal(0.006, 0.02, 300), [-0.08, -0.11]]
    k = exact_concave_fractional_kelly(r, c_kelly=0.20, f_max=50.0)
    rt = np.maximum(r - r.mean() + k.mu_lcb, -0.95)
    grid = np.linspace(0, min(50.0, 0.95 / abs(rt.min())), 20001)
    growth = np.array([np.mean(np.log1p(f * rt)) for f in grid])
    assert k.f_star == pytest.approx(grid[growth.argmax()], abs=grid[1] * 2)
    assert k.f_kelly == pytest.approx(0.20 * k.f_star)
    assert k.f_star <= 0.95 / abs(rt.min()) + 1e-12
    with pytest.raises(ValueError):
        exact_concave_fractional_kelly(r, c_kelly=0.5)


def test_kelly_differs_from_gaussian_shortcut_under_fat_tails():
    rng = np.random.default_rng(3)
    r = np.clip(0.004 + 0.02 * rng.standard_t(2.5, 500), -0.5, 0.5)
    k = exact_concave_fractional_kelly(r, f_max=100.0)
    if k.f_star > 0:
        rt = r - r.mean() + k.mu_lcb
        gauss = rt.mean() / rt.var()
        assert k.f_star < gauss                                             # mu/sigma^2 over-bets heavy tails


# ------------------------------------------------------------------------- EVT


def test_gpd_var_es_formulas_against_known_tail():
    rng = np.random.default_rng(5)
    xi, beta, u, n = 0.2, 0.01, 0.02, 200_000
    body = rng.uniform(0, u, int(n * 0.95))
    exc = genpareto.rvs(xi, scale=beta, size=n - len(body), random_state=rng)
    losses = np.r_[body, u + exc]
    fit = tail.fit_gpd_pot(losses, 0.95)
    assert fit.xi == pytest.approx(xi, abs=0.03) and fit.beta == pytest.approx(beta, rel=0.05)
    for a in (0.975, 0.99):
        var_true = u + beta / xi * (((1 - a) / 0.05) ** (-xi) - 1)
        es_true = (var_true + beta - xi * u) / (1 - xi)
        assert fit.var(a) == pytest.approx(var_true, rel=0.03)
        assert fit.es(a) == pytest.approx(es_true, rel=0.04)
        assert fit.es(a) > fit.var(a)
    assert fit.es(0.99) == pytest.approx(np.sort(losses)[int(0.99 * n):].mean(), rel=0.05)
    assert tail.fit_gpd_pot(losses[:30]) is None


def test_tsallis_student_equivalence_and_kurtosis_boundary():
    rng = np.random.default_rng(2)
    q, nu = tail.tsallis_q_from_returns(rng.standard_t(6, 20_000) * 0.01)
    assert nu == pytest.approx(6, rel=0.2) and q == pytest.approx(1 + 2 / (nu + 1))
    assert 2 / (tail.Q_KURTOSIS_BOUNDARY - 1) - 1 == pytest.approx(4.0)     # q = 1.40 <=> nu = 4
    df = tail.rolling_tsallis_q(pd.Series(rng.standard_t(3, 700) * 0.01), 504, 50)
    assert df["tail_alert"].iloc[-1]


def test_var_backtests():
    rng = np.random.default_rng(8)
    n = 3000
    ok = np.zeros(n, dtype=bool); ok[::100] = True                          # exactly 1% violations, unclustered
    assert tail.kupiec_pof(ok, 0.99)[1] > 0.05
    assert tail.kupiec_pof(rng.random(n) < 0.04, 0.99)[1] < 1e-6
    clustered = np.zeros(n, dtype=bool); clustered[100:130] = True
    assert tail.christoffersen_independence(clustered)[1] < 1e-6
    assert tail.christoffersen_independence(ok)[1] > 0.01
    loss = rng.standard_t(5, n) * 0.01
    var = np.full(n, np.quantile(loss, 0.975)); es = np.full(n, loss[loss > var[0]].mean())
    z_ok, _ = tail.acerbi_szekely_z2(loss, var, es, 0.975)
    z_bad, p_bad = tail.acerbi_szekely_z2(loss, var / 2, es / 2, 0.975)
    assert abs(z_ok) < 0.2 and z_bad > 0.7 and p_bad < 0.01


def test_rolling_evt_is_causal_and_two_sided():
    rng = np.random.default_rng(4)
    r = pd.Series(np.where(rng.random(900) < 0.5, 1, -1) * np.abs(rng.standard_t(4, 900)) * 0.01)
    r = r - np.abs(r) * 0.3 * (r < 0)                                       # fatter left tail
    a = tail.rolling_evt(r, 504, 5)
    b = tail.rolling_evt(r.iloc[:700], 504, 5)
    assert np.allclose(a.iloc[:700].to_numpy(), b.to_numpy(), equal_nan=True)
    last = a.dropna().iloc[-1]
    assert last["es99_long"] > last["var99_long"] > 0 and last["es99_long"] > last["es99_short"]


# -------------------------------------------------------------------- breakers


def test_drawdown_tiers():
    cb = CircuitBreakers(RiskLimits(), 5000.0)
    cb.on_session_open(5000.0)
    assert cb.m_dd == 1.0 and cb.can_enter and math.isinf(cb.leverage_cap)
    assert not cb.on_bar_close(4960.0)                                      # -0.8% on the day
    assert cb.can_enter
    assert not cb.on_bar_close(4950.0) and not cb.can_enter                 # -1.0% intraday: no new entries today
    cb.on_session_close(4950.0); cb.on_session_open(4950.0)
    assert cb.can_enter                                                     # a new day re-arms entries

    # slow bleed of 0.2% of initial equity per day: the rolling five-day rule stays quiet,
    # so the drawdown tiers can be observed in isolation at their exact euro levels
    cb = CircuitBreakers(RiskLimits(), 5000.0)
    seen = {}
    eq = 5000.0
    while cb.live:
        cb.on_session_open(eq)
        eq -= 10.0
        flatten = cb.on_bar_close(eq)
        cb.on_session_close(eq)
        seen[round(eq)] = (cb.m_dd, cb.leverage_cap, flatten)
    assert seen[4810] == (1.0, math.inf, False)
    assert seen[4800] == (0.75, 1.0, False)                                 # exactly -4.0% (EUR 4,800)
    assert seen[4710] == (0.75, 1.0, False)
    assert seen[4700] == (0.25, 1.0, False)                                 # exactly -6.0%
    assert seen[4600][2] is True and cb.paused and not cb.can_enter         # exactly -8.0%: flatten and pause
    assert not cb.on_bar_close(4600.0)                                      # already paused: no second flatten


def test_fast_loss_triggers_rolling_rule_before_the_drawdown_tier():
    cb = CircuitBreakers(RiskLimits(), 5000.0)
    cb.on_session_open(5000.0)
    cb.on_bar_close(4890.0)                                                 # -2.2% in one day
    assert cb.m_dd == 0.5 and math.isinf(cb.leverage_cap) and not cb.can_enter


def test_hard_deactivation_is_anchored_to_the_all_time_peak_and_is_terminal():
    cb = CircuitBreakers(RiskLimits(pause_cooldown_days=3), 5000.0)
    cb.on_session_open(5000.0)
    assert cb.hard_kill_equity == pytest.approx(4500.0)
    assert cb.on_bar_close(4490.0) and cb.deactivated                        # straight through -10%
    for _ in range(300):                                                     # no cool-down ever revives it
        cb.on_session_close(4490.0); cb.on_session_open(4490.0)
    cb.on_refit(4490.0)                                                      # ... and neither does a refit
    assert cb.deactivated and not cb.live and cb.true_peak == 5000.0
    # a new equity high lifts the kill level with it
    cb = CircuitBreakers(RiskLimits(), 5000.0)
    cb.on_session_open(5000.0); cb.on_bar_close(5200.0)
    assert cb.hard_kill_equity == pytest.approx(4680.0)


def test_pause_resume_cannot_step_down_past_ten_percent():
    """The loophole: pause at -8%, re-base, lose another 8% from the lower
    base, ... and never meet the -10% deactivation.  The pause reference is
    re-based, the high-water mark is not."""
    cb = CircuitBreakers(RiskLimits(pause_cooldown_days=3), 5000.0)
    cb.on_session_open(5000.0)
    assert cb.on_bar_close(4590.0) and cb.paused and not cb.deactivated      # -8.2%: pause
    cb.on_session_close(4590.0)
    cb.on_refit(4590.0)                                                      # walk-forward refit restarts it
    assert cb.live and cb.pause_ref == 4590.0 and cb.true_peak == 5000.0
    assert cb.m_dd == 0.25 and cb.leverage_cap == 1.0                        # still 8.2% under the real peak
    cb.on_session_open(4590.0)
    assert not cb.on_bar_close(4560.0)                                       # -0.7% vs the re-based reference
    cb.on_session_close(4560.0); cb.on_session_open(4560.0)
    assert cb.on_bar_close(4500.0)                                           # only -2% vs 4,590 but -10% vs 5,000
    assert cb.deactivated and [e[1] for e in cb.events].count("deactivate") == 1
    # risk comes back only with a real recovery
    cb = CircuitBreakers(RiskLimits(pause_cooldown_days=1), 5000.0)
    cb.on_session_open(5000.0); cb.on_bar_close(4590.0); cb.on_session_close(4590.0)
    cb.on_session_open(4590.0)
    assert cb.live and cb.m_dd == 0.25
    for eq in (4640.0, 4700.0, 4760.0, 4820.0):
        cb.on_bar_close(eq); cb.on_session_close(eq); cb.on_session_open(eq)
    assert cb.m_dd == 1.0 and math.isinf(cb.leverage_cap)                    # back within 4% of the peak


def test_lot_rules_are_normalised_to_a_reference_price():
    """At index 2,500 one IBUST100 unit is USD 2,500 and looks tradable on EUR
    5,000; at 25,000 it is not.  With a reference price the backtest applies
    today's lot notional throughout history."""
    from ndxswing.config import ibkr_ibust100

    st = AccountState(equity=5000.0, fx_quote_to_acct=0.93)
    old = proposal(entry_stop=2500.0, entry_limit=2501.5, stop_price=2490.0, d_stop=10.0, atr=6.6, ref_price=2499.0)
    naive = engine(inst=ibkr_ibust100()).approve(old, st)
    assert isinstance(naive, RiskApprovedOrder) and naive.units == 1.0       # the historical illusion
    honest = engine(inst=ibkr_ibust100(lot_reference_price=25_000.0)).approve(old, st)
    assert isinstance(honest, RiskRejection) and honest.reason == "below_min_lot"
    today = engine(inst=ibkr_ibust100(lot_reference_price=25_000.0)).approve(proposal(), st)
    assert isinstance(today, RiskRejection)                                  # 1 unit = 4.65x equity
    micro = engine(inst=ndx_fractional_cfd(0.01, 25_000.0)).approve(old, st)
    assert micro.lot_step == pytest.approx(0.1) and micro.units / 0.1 == pytest.approx(round(micro.units / 0.1))
    assert micro.units * 2500.0 == pytest.approx(engine().approve(proposal(), st).units * 25_000.0, rel=0.25)
    # a 0.1 lot cannot carry a EUR 12.50 budget with a 135 point stop at 25,000 ...
    wide = proposal(d_stop=135.0, stop_price=24_865.0)
    assert isinstance(engine(inst=ndx_fractional_cfd(0.1)).approve(wide, st), RiskRejection)
    # ... but it can carry EUR 25.00, and a 0.01 lot carries both
    assert engine(StrategyParams(b_risk=0.005), inst=ndx_fractional_cfd(0.1)).approve(wide, st).units == pytest.approx(0.1)
    assert engine(inst=ndx_fractional_cfd(0.01)).approve(wide, st).units == pytest.approx(0.07)


def test_points_spread_and_broker_presets():
    from ndxswing.config import ibkr_cfd_costs, pepperstone_costs, ucits_etf_costs, ucits_ndx_etf

    c = pepperstone_costs("standard")
    assert c.half_spread_at(25_000.0) == pytest.approx(0.6 / 25_000.0)
    assert c.stressed(1.5).half_spread_at(25_000.0) == pytest.approx(0.9 / 25_000.0)
    assert pepperstone_costs().spread_points == 1.0                               # Razor
    assert CostModel(spread_bps=2.0).half_spread_at(123.0) == pytest.approx(1e-4)
    ib = ibkr_cfd_costs()
    assert ib.commission(100.0) == 1.0 and ib.commission(50_000.0) == pytest.approx(5.0)   # 0.010%, minimum 1.00
    assert ib.financing_markup == 0.015 and c.financing_markup == 0.025
    etf = ucits_ndx_etf()
    assert not etf.allow_short and not etf.pays_financing and etf.fx_mode == "price" and etf.lot_step == 1.0
    assert ucits_etf_costs().financing_markup == 0.0
    with pytest.raises(ValueError):
        pepperstone_costs("vip")


def test_eu_broker_presets_match_the_supplied_schedules():
    from ndxswing.config import BROKER_PRESETS, broker_preset

    assert BROKER_PRESETS == ("capital_com_eu_leveraged", "capital_com_eu_unleveraged_1to1",
                              "pepperstone_eu_ctrader", "ibkr_cfd", "ucits_etf")
    pep = broker_preset("pepperstone_eu_ctrader", lot_reference_price=25_288.25)
    assert (pep.instrument.lot_step, pep.instrument.min_lot, pep.instrument.multiplier) == (0.10, 0.10, 1.0)
    assert (pep.costs.spread_points, pep.costs.commission_bps, pep.costs.commission_min) == (1.0, 0.0, 0.0)
    assert (pep.costs.financing_markup, pep.costs.financing_day_count, pep.costs.fx_conversion_fee) == (0.025, 360, 0.0)

    lev = broker_preset("capital_com_eu_leveraged")
    assert (lev.instrument.lot_step, lev.instrument.min_lot, lev.instrument.multiplier) == (0.001, 0.001, 1.0)
    assert (lev.costs.spread_points, lev.costs.commission_bps, lev.costs.commission_min) == (1.8, 0.0, 0.0)
    assert (lev.costs.financing_markup, lev.costs.financing_day_count) == (0.040, 360)
    assert lev.costs.financing_markup / 360 == pytest.approx(0.0001111, rel=1e-3)       # 0.01111% a day
    assert lev.costs.fx_conversion_fee == 0.007 and lev.instrument.pays_financing
    assert lev.instrument.max_leverage is None and lev.instrument.margin_rate == 0.05

    one = broker_preset("capital_com_eu_unleveraged_1to1")
    assert (one.instrument.lot_step, one.instrument.min_lot, one.costs.spread_points) == (0.001, 0.001, 1.8)
    assert one.instrument.max_leverage == 1.0 and not one.instrument.pays_financing
    assert one.costs.financing_markup == 0.0 and one.costs.fx_conversion_fee == 0.0
    assert one.instrument.allow_short                                               # still a CFD

    for gone in ("oanda_eu_v20", "capital_com_eu_rest", "micro_cfd"):
        with pytest.raises(ValueError):
            broker_preset(gone)


def test_lot_size_decides_which_broker_can_carry_the_burn_in_budget():
    from ndxswing.config import broker_preset

    st = AccountState(equity=5000.0, fx_quote_to_acct=0.93)
    wide = proposal(d_stop=135.0, stop_price=24_865.0)                               # ~1.5 ATR at index 25,000
    pep, lev = broker_preset("pepperstone_eu_ctrader"), broker_preset("capital_com_eu_leveraged")
    # 0.10 lot: EUR 12.50 is clipped to zero, EUR 17.50 carries exactly one lot
    assert isinstance(engine(inst=pep.instrument, costs=pep.costs).approve(wide, st), RiskRejection)
    assert isinstance(engine(StrategyParams(b_risk=0.0030), inst=pep.instrument, costs=pep.costs).approve(wide, st),
                      RiskRejection)
    ok = engine(StrategyParams(b_risk=0.0035), inst=pep.instrument, costs=pep.costs).approve(wide, st)
    assert ok.units == pytest.approx(0.10) and ok.planned_risk <= 17.5
    # 0.001 lot: the burn-in budget is used almost in full
    fine = engine(inst=lev.instrument, costs=lev.costs).approve(wide, st)
    # 12.5 / (1.25 * 135 * 0.93 + 23,250 * (1.8 / 25,000 + 2 bp)) = 0.0766 -> 0.076
    assert fine.units == pytest.approx(0.076) and 12.3 < fine.planned_risk <= 12.5


def test_vehicle_leverage_cap_and_one_to_one_margin():
    from ndxswing.config import broker_preset

    st = AccountState(equity=5000.0, fx_quote_to_acct=1.0)
    calm = proposal(d_stop=5.0, stop_price=24_995.0, sigma_1d=0.002)                  # exposure caps bind
    lev = broker_preset("capital_com_eu_leveraged")
    one = broker_preset("capital_com_eu_unleveraged_1to1")
    r_lev = engine(inst=lev.instrument).approve(calm, st)
    assert r_lev.binding == "max_leverage" and r_lev.leverage == pytest.approx(1.5, abs=0.01)
    r_one = engine(inst=one.instrument).approve(calm, st)
    assert r_one.leverage <= 0.5 + 1e-9 and r_one.binding == "margin"                 # full margin, 50% usage rule
    # with the margin-usage rule out of the way the vehicle's own 1.0 cap is what binds
    r_cap = engine(StrategyParams(margin_usage_cap=5.0), inst=one.instrument).approve(calm, st)
    assert r_cap.binding == "max_leverage" and r_cap.leverage == pytest.approx(1.0, abs=0.01)
    # the drawdown tier can only tighten it further
    tier = engine(StrategyParams(margin_usage_cap=5.0), inst=one.instrument).approve(
        calm, AccountState(5000.0, 1.0, leverage_cap=0.6))
    assert tier.leverage <= 0.6 + 1e-9


def test_rolling_five_day_loss_halves_risk():
    cb = CircuitBreakers(RiskLimits(), 5000.0)
    eq = 5000.0
    for _ in range(5):
        cb.on_session_open(eq)
        eq -= 22.0                                                            # 0.44% a day: no intraday halt
        cb.on_bar_close(eq)
        cb.on_session_close(eq)
    cb.on_session_open(eq)
    assert 1 - eq / 5000.0 >= 0.02 and cb.m_dd == 0.5 and cb.can_enter
