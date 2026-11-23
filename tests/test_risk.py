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


def engine(flags=ModuleFlags(), params=StrategyParams(), inst=None, costs=NOCOST) -> PortfolioRiskEngine:
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


def test_deactivation_and_resume_rebases_reference_but_not_true_peak():
    lim = RiskLimits(pause_cooldown_days=3, deactivate_cooldown_days=5)
    cb = CircuitBreakers(lim, 5000.0)
    cb.on_session_open(5000.0)
    assert cb.on_bar_close(4490.0) and cb.deactivated                        # straight through -10%
    for _ in range(4):
        cb.on_session_close(4490.0); cb.on_session_open(4490.0)
        assert not cb.live
    cb.on_session_close(4490.0); cb.on_session_open(4490.0)
    assert cb.live and cb.ref_peak == 4490.0 and cb.true_peak == 5000.0 and cb.m_dd == 1.0
    cb2 = CircuitBreakers(lim, 5000.0)
    cb2.on_session_open(5000.0); cb2.on_bar_close(4590.0)
    assert cb2.paused
    cb2.on_refit(4590.0)                                                     # walk-forward re-estimation restarts it
    assert cb2.live and cb2.ref_peak == 4590.0


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
