"""Volatility-managed exposure: measurement, weights, retail accounting and the
benchmark-relative validator.  Synthetic or hand-computed inputs, except the
last block, which checks invariances on the committed (spent) NASDAQ-100 daily
table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ndxswing.analytics import relative as rel
from ndxswing.research import volmanaged as vm
from ndxswing.research.volmanaged import CashSweep, GermanTax, MarketDays, RetailCosts

FREE = RetailCosts(spread_points=0.0, fx_fee=0.0, funding_markup=0.0, dividend_credit=1.0)
NO_TAX = GermanTax(enabled=False)


def market(n: int = 600, seed: int = 4, start: str = "2019-01-02", drift: float = 0.0005, fx_vol: float = 0.004,
           div: float = 0.00003, usd_rate: float = 0.03, eur_rate: float = 0.02) -> MarketDays:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    gap = rng.normal(0.0, 0.004, n)
    intra = rng.normal(drift, 0.011, n)
    close = 10_000.0 * np.exp(np.cumsum(gap + intra))
    open_ = close * np.exp(-intra)
    fx = 1.10 * np.exp(np.cumsum(rng.normal(0.0, fx_vol, n)))
    return MarketDays(idx, open_, close, fx, np.full(n, usd_rate), np.full(n, eur_rate), np.full(n, div))


def hold(n: int, w0: float) -> np.ndarray:
    t = np.full(n, np.nan)
    t[0] = w0
    return t


# ------------------------------------------------------------------ measurement


def test_realised_measures_identities_and_jump_detection():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.0006, 390)
    close = 100.0 * np.exp(np.cumsum(r))
    open_ = np.r_[100.0, close[:-1]]
    m = vm.realised_measures(open_, close)
    assert m["sum_r1"] == pytest.approx(np.log(close[-1] / 100.0), abs=1e-12)
    assert m["rv1"] == pytest.approx(np.sum(r**2))
    assert m["rsm5"] + m["rsp5"] == pytest.approx(m["rv5"])
    # 78 five-minute returns of a diffusion: bipower variation estimates the same variance
    assert m["bv5"] == pytest.approx(m["rv5"], rel=0.35) and m["jv5"] >= 0.0
    # one 1% jump: realised variance takes all of it, bipower variation almost none
    rj = r.copy()
    rj[200] += 0.01
    cj = 100.0 * np.exp(np.cumsum(rj))
    mj = vm.realised_measures(np.r_[100.0, cj[:-1]], cj)
    assert mj["rv5"] - m["rv5"] > 0.8e-4 and mj["bv5"] - m["bv5"] < 0.3e-4
    assert mj["jv5"] > 0.6e-4
    # a session that ends off the five-minute grid still telescopes to the close
    m17 = vm.realised_measures(open_[:17], close[:17])
    assert m17["sum_r1"] == pytest.approx(np.log(close[16] / 100.0), abs=1e-12)


def test_variance_signals_are_causal():
    rng = np.random.default_rng(2)
    n = 700
    idx = pd.bdate_range("2016-01-04", periods=n)
    d = pd.DataFrame({"r_cc": rng.normal(0, 0.012, n), "r_gap": rng.normal(0, 0.004, n),
                      "rv5": rng.gamma(2.0, 4e-5, n)}, index=idx)
    d["bv5"], d["rsm5"] = d["rv5"] * 0.95, d["rv5"] * 0.5
    full = vm.variance_signals(d)
    cut = vm.variance_signals(d.iloc[:500])
    pd.testing.assert_frame_equal(full.iloc[:500], cut)                # nothing after t enters row t
    assert full["rv20"].iloc[40] == pytest.approx((d["r_cc"].iloc[21:41] ** 2).mean())
    assert full["har"].notna().sum() > 300 and (full["har"].dropna() > 0).all()
    assert full["har"].iloc[:252].isna().all()


def test_scalar_choices_real_time_against_in_sample():
    rng = np.random.default_rng(3)
    n = 1500
    idx = pd.bdate_range("2015-01-02", periods=n)
    vol = 0.01 * np.exp(0.5 * np.sin(np.arange(n) / 60.0))
    r = pd.Series(rng.normal(0.0004, 1.0, n) * vol, index=idx)
    f = (r**2).rolling(20).mean()
    ins = vm.managed_weights(f, r, "insample")
    cau = vm.managed_weights(f, r, "causal")
    m_ins = (ins["w"].shift(1) * r).dropna()
    assert m_ins.std(ddof=1) == pytest.approx(r[m_ins.index].std(ddof=1), rel=1e-12)   # in-sample c equalises volatility
    assert ins["c"].nunique() == 1 and cau["c"].iloc[:251].isna().all()
    # the causal scalar at t does not change when the future is removed
    cut = vm.managed_weights(f.iloc[:900], r.iloc[:900], "causal")
    pd.testing.assert_series_equal(cau["c"].iloc[:900], cut["c"])
    # frictionless appraisal ratio does not depend on the scalar: only the size of the bet does
    base = rel.spanning_regression(m_ins.to_numpy(), r[m_ins.index].to_numpy())
    twice = rel.spanning_regression(2.5 * m_ins.to_numpy(), r[m_ins.index].to_numpy())
    assert twice["appraisal_ratio"] == pytest.approx(base["appraisal_ratio"], rel=1e-10)
    assert twice["alpha"] == pytest.approx(2.5 * base["alpha"], rel=1e-10) and twice["t_alpha"] == pytest.approx(base["t_alpha"])


# ------------------------------------------------------------------- accounting


def test_conservation_weight_one_never_rebalanced_is_buy_and_hold():
    mkt = market()
    sim = vm.simulate_cfd_account(mkt, hold(len(mkt), 1.0), deadband=np.inf, costs=FREE, tax=NO_TAX, rebalance=False)
    exact = vm.buy_and_hold_cfd_closed_form(mkt, 5000.0, 1.0)
    assert np.max(np.abs(sim.frame["equity"].to_numpy() / exact - 1.0)) < 1e-12
    assert np.max(np.abs(sim.frame["liq_post"].to_numpy() / exact - 1.0)) < 1e-12     # no costs, no tax: same thing
    assert sim.totals["rebalances"] == 1 and sim.totals["tax"] == 0.0
    # an infinite deadband does the same as switching rebalancing off, whatever the targets say
    noisy = np.r_[1.0, np.random.default_rng(0).uniform(0.0, 1.0, len(mkt) - 1)]
    same = vm.simulate_cfd_account(mkt, noisy, deadband=np.inf, costs=FREE, tax=NO_TAX)
    assert np.array_equal(same.frame["equity"].to_numpy(), sim.frame["equity"].to_numpy())


def test_conservation_weight_zero_is_idle_cash():
    mkt = market()
    flat = vm.simulate_cfd_account(mkt, hold(len(mkt), 0.0), deadband=np.inf, costs=FREE, tax=NO_TAX)
    assert np.array_equal(flat.frame["liq_post"].to_numpy(), np.full(len(mkt), 5000.0))   # broker cash earns nothing
    free_sweep = CashSweep(index_spread=0.0, ter=0.0, ticket=0.0, half_spread=0.0)
    swept = vm.simulate_cfd_account(mkt, hold(len(mkt), 0.0), deadband=np.inf, costs=FREE, tax=NO_TAX, sweep=free_sweep)
    exact = 5000.0 * np.cumprod(1.0 + np.r_[0.0, 0.02 * mkt.nights[1:] / 360.0])
    assert np.max(np.abs(swept.frame["liq_post"].to_numpy() / exact - 1.0)) < 1e-12
    # the real sweep pays a ticket and earns the index spread less the fund's fee
    real = vm.simulate_cfd_account(mkt, hold(len(mkt), 0.0), deadband=np.inf, costs=FREE, tax=NO_TAX, sweep=CashSweep())
    assert 0 < real.terminal < swept.terminal
    assert CashSweep().net_rate(-0.004) == 0.0 and CashSweep().net_rate(0.03) == pytest.approx(0.02985)


def test_spread_conversion_and_realised_gain_by_hand():
    idx = pd.bdate_range("2021-03-01", periods=3)
    mkt = MarketDays(idx, np.array([10_000.0, 10_100.0, 10_300.0]), np.array([10_050.0, 10_200.0, 10_400.0]),
                     np.full(3, 1.25), np.zeros(3), np.zeros(3), np.zeros(3))
    costs = RetailCosts(spread_points=2.0, fx_fee=0.01, funding_markup=0.0)
    # day 0: buy 0.5 units at 10,001 (weight 1 of EUR 4,000 -> 5,000 USD / 10,000); day 2: target 0 -> sell at 10,299
    sim = vm.simulate_cfd_account(mkt, np.array([1.0, np.nan, 0.0]), initial=4000.0, deadband=0.1, costs=costs, tax=NO_TAX)
    f = sim.frame
    assert f["units"].tolist() == [0.5, 0.5, 0.0]
    assert f["equity"].iloc[0] == pytest.approx(4000.0 + 0.5 * (10_050.0 - 10_001.0) / 1.25)
    gain_eur = 0.5 * (10_299.0 - 10_001.0) / 1.25
    assert f["equity"].iloc[2] == pytest.approx(4000.0 + gain_eur * 0.99)
    assert sim.totals["fx_fee"] == pytest.approx(0.01 * gain_eur) and sim.totals["spread"] == pytest.approx(2 * 0.5 * 1.0 / 1.25)
    # liquidation value on day 1 deducts the closing half-spread and the conversion mark-up on the gain
    assert f["liq_pre"].iloc[1] == pytest.approx(4000.0 + 0.5 * (10_199.0 - 10_001.0) / 1.25 * 0.99)


def test_deadband_suppresses_small_rebalances():
    mkt = market(300)
    wobble = 0.6 + 0.04 * np.sin(np.arange(300))
    tight = vm.simulate_cfd_account(mkt, wobble, deadband=0.0, costs=RetailCosts(), tax=NO_TAX)
    wide = vm.simulate_cfd_account(mkt, wobble, deadband=0.10, costs=RetailCosts(), tax=NO_TAX)
    assert tight.totals["rebalances"] > 250 and wide.totals["rebalances"] < 40
    assert wide.totals["spread"] < 0.2 * tight.totals["spread"]
    assert (wide.frame["weight"] <= 1.0 + 0.15).all()


def test_leverage_cliff_charges_the_whole_notional_only_above_one():
    idx = pd.bdate_range("2022-06-06", periods=6)          # Monday .. Monday: one weekend
    flat = np.full(6, 10_000.0)
    mkt = MarketDays(idx, flat, flat, np.ones(6), np.full(6, 0.05), np.zeros(6), np.zeros(6))
    costs = RetailCosts(spread_points=0.0, fx_fee=0.0, funding_markup=0.04)
    levered = vm.simulate_cfd_account(mkt, hold(6, 1.2), initial=5000.0, w_max=1.5, costs=costs, tax=NO_TAX,
                                      funding="cliff", rebalance=False)
    nights = 7                                              # Mon->Mon
    per_night = 0.6 * 10_000.0 * 0.09 / 360.0               # 0.6 units, the WHOLE notional, benchmark + 4%
    # weight creeps above 1.2 as funding eats equity, but the charge is always on 6,000 of notional
    assert levered.totals["funding"] == pytest.approx(nights * per_night)
    assert levered.frame["equity"].iloc[-1] == pytest.approx(5000.0 - nights * per_night)
    assert levered.totals["nights_levered"] == nights
    at_one = vm.simulate_cfd_account(mkt, hold(6, 1.0), initial=5000.0, w_max=1.5, costs=costs, tax=NO_TAX,
                                     funding="cliff", rebalance=False)
    assert at_one.totals["funding"] == 0.0 and at_one.frame["equity"].iloc[-1] == 5000.0
    with pytest.raises(ValueError):
        vm.simulate_cfd_account(mkt, hold(6, 1.2), w_max=1.5, costs=costs, tax=NO_TAX, funding="none")


def test_dividends_are_credited_and_taxed_as_income():
    idx = pd.bdate_range("2021-11-01", periods=3)
    flat = np.full(3, 8_000.0)
    mkt = MarketDays(idx, flat, flat, np.ones(3), np.zeros(3), np.zeros(3), np.array([0.0, 0.001, 0.0]))
    sim = vm.simulate_cfd_account(mkt, hold(3, 1.0), initial=4000.0, costs=FREE, tax=NO_TAX, rebalance=False)
    assert sim.totals["dividends"] == pytest.approx(0.5 * 8.0) and sim.frame["equity"].iloc[-1] == pytest.approx(4004.0)
    part = vm.simulate_cfd_account(mkt, hold(3, 1.0), initial=4000.0, tax=NO_TAX, rebalance=False,
                                   costs=RetailCosts(spread_points=0.0, fx_fee=0.0, dividend_credit=0.7))
    assert part.totals["dividends"] == pytest.approx(0.7 * 4.0)
    taxed = vm.simulate_cfd_account(mkt, hold(3, 1.0), initial=4000.0, costs=FREE, tax=GermanTax(), rebalance=False)
    assert taxed.terminal == pytest.approx(4000.0 + 4.0 * (1 - 0.26375)) and taxed.totals["tax"] == pytest.approx(4.0 * 0.26375)


def test_cfd_tax_is_annual_with_loss_carry_forward_and_allowance():
    # three calendar years, one session each: +200, -300, +250 realised by going flat and re-entering
    idx = pd.DatetimeIndex(["2021-06-01", "2021-06-02", "2022-06-01", "2022-06-02", "2023-06-01", "2023-06-02"])
    o = np.array([10_000.0, 10_400.0, 10_000.0, 9_400.0, 10_000.0, 10_500.0])
    mkt = MarketDays(idx, o, o, np.ones(6), np.zeros(6), np.zeros(6), np.zeros(6))
    tgt = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    sim = vm.simulate_cfd_account(mkt, tgt, initial=5000.0, deadband=0.0, costs=FREE, tax=GermanTax())
    g1 = 0.5 * 400.0                                           # 2021: +200, taxed
    e1 = 5000.0 + g1 * (1 - 0.26375)
    u2 = np.floor(e1 / 10_000.0 / 0.001 + 1e-9) * 0.001
    l2 = u2 * -600.0                                           # 2022: loss, carried forward, no refund
    e2 = e1 + l2
    u3 = np.floor(e2 / 10_000.0 / 0.001 + 1e-9) * 0.001
    g3 = u3 * 500.0                                            # 2023: gain, reduced by the loss carried
    tax3 = 0.26375 * max(g3 + l2, 0.0)
    assert sim.terminal == pytest.approx(e2 + g3 - tax3)
    assert sim.totals["tax"] == pytest.approx(g1 * 0.26375 + tax3)
    # the saver's allowance shelters the first EUR 801 (1,000 from 2023) of each year's income
    allow = vm.simulate_cfd_account(mkt, tgt, initial=5000.0, deadband=0.0, costs=FREE,
                                    tax=GermanTax(allowance_to_2022=801.0, allowance_from_2023=1000.0))
    assert allow.totals["tax"] == 0.0 and allow.terminal > sim.terminal
    # liquidation-basis value before year end already deducts the tax the open gain would trigger
    held = vm.simulate_cfd_account(mkt.__class__(idx[:2], o[:2], o[:2], np.ones(2), np.zeros(2), np.zeros(2), np.zeros(2)),
                                   hold(2, 1.0), initial=5000.0, costs=FREE, tax=GermanTax(), rebalance=False)
    assert held.frame["liq_post"].iloc[1] == pytest.approx(5000.0 + 200.0 * (1 - 0.26375))


def test_etf_buy_and_hold_fee_dividends_fx_and_deferred_tax():
    idx = pd.bdate_range("2020-01-02", periods=253)
    n = len(idx)
    close = 10_000.0 * np.cumprod(np.r_[1.0, np.full(n - 1, 1.0005)])
    mkt = MarketDays(idx, np.r_[10_000.0, close[:-1]], close, np.linspace(1.10, 1.20, n), np.zeros(n), np.zeros(n),
                     np.full(n, 0.00004))
    etf = vm.simulate_etf_buy_and_hold(mkt, initial=5000.0, ter=0.003, half_spread=0.0, ticket=0.0, tax=NO_TAX)
    growth = np.prod((close[1:] / close[:-1] + 0.00004 * 0.85) * (1 - 0.003 * mkt.nights[1:] / 365.0))
    assert etf.terminal == pytest.approx(5000.0 * growth * 1.10 / 1.20)        # a stronger euro shrinks the whole holding
    taxed = vm.simulate_etf_buy_and_hold(mkt, initial=5000.0, ter=0.003, half_spread=0.0, ticket=0.0, tax=GermanTax())
    gain = etf.terminal - 5000.0
    assert taxed.terminal == pytest.approx(etf.terminal - 0.26375 * 0.70 * gain)   # 18.4625% of the gain, once, at the end
    assert taxed.totals["tax"] == pytest.approx(0.184625 * gain)
    # nothing is paid before the sale: mid value identical with and without tax
    assert np.array_equal(taxed.frame["equity"].to_numpy(), etf.frame["equity"].to_numpy())
    # a loss is not taxed and not refunded
    down = MarketDays(idx, mkt.open[::-1].copy(), mkt.close[::-1].copy(), np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n))
    lost = vm.simulate_etf_buy_and_hold(down, initial=5000.0, tax=GermanTax())
    assert lost.totals["tax"] == 0.0 and lost.terminal < 5000.0
    # ticket and spread on both legs
    costly = vm.simulate_etf_buy_and_hold(mkt, initial=5000.0, ter=0.003, half_spread=0.0003, ticket=1.0, tax=NO_TAX)
    assert costly.terminal == pytest.approx((5000.0 - 1.0) / 1.0003 * growth * 1.10 / 1.20 * (1 - 0.0003) - 1.0)


def test_vorabpauschale_prepays_part_of_the_final_tax():
    idx = pd.bdate_range("2023-01-02", "2025-06-30")
    n = len(idx)
    close = 10_000.0 * np.cumprod(np.r_[1.0, np.full(n - 1, 1.0006)])
    mkt = MarketDays(idx, np.r_[10_000.0, close[:-1]], close, np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n))
    plain = vm.simulate_etf_buy_and_hold(mkt, ter=0.0, tax=GermanTax())
    vp = vm.simulate_etf_buy_and_hold(mkt, ter=0.0, tax=GermanTax(vorabpauschale=True))
    assert vp.frame["units"].iloc[-1] < plain.frame["units"].iloc[-1]            # units were sold to pay it
    assert 0 < plain.terminal - vp.terminal < 0.01 * plain.terminal               # a timing cost, not a second tax
    first_jan = np.flatnonzero(idx.year == 2024)[0]
    start_value = plain.frame["units"].iloc[0] * 10_000.0
    paid_2024 = 0.26375 * 0.7 * (start_value * 0.0255 * 0.7)                      # full year held, Basiszins 2.55%
    sold = (plain.frame["units"].iloc[0] - vp.frame["units"].iloc[first_jan]) * close[first_jan - 1] * (1 - 0.0003)
    assert sold == pytest.approx(paid_2024, rel=1e-9)


# ----------------------------------------------------------------- the validator


def test_spanning_regression_matches_statsmodels_hac():
    import statsmodels.api as sm

    rng = np.random.default_rng(5)
    n = 1200
    x = rng.normal(0.0005, 0.012, n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.3 * e[t - 1] + rng.normal(0, 0.004)
    y = 0.0002 + 0.7 * x + e
    out = rel.spanning_regression(y, x, lags=8)
    ref = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 8, "use_correction": True})
    assert out["alpha"] == pytest.approx(ref.params[0]) and out["beta"] == pytest.approx(ref.params[1])
    assert out["se_alpha"] == pytest.approx(ref.bse[0], rel=1e-9) and out["t_alpha"] == pytest.approx(ref.tvalues[0], rel=1e-9)
    assert out["appraisal_ratio"] == pytest.approx(out["alpha"] / np.std(y - ref.fittedvalues, ddof=2) * np.sqrt(252))
    assert rel.newey_west_lags(2400) == 8


def test_paired_difference_test_point_estimates_and_standard_error():
    rng = np.random.default_rng(6)
    n = 2000
    b = rng.normal(0.0004, 0.01, n)
    a = 0.5 * b + rng.normal(0.0006, 0.006, n)
    d = rel.paired_difference_test(a, b, "sharpe", n_boot=600, seed=1)
    assert d["delta_ann"] == pytest.approx(rel.sharpe_ann(a) - rel.sharpe_ann(b), rel=1e-3)   # population vs sample std
    c = rel.paired_difference_test(a, b, "cer", gamma=3.0, n_boot=600, seed=1)
    assert c["delta_ann"] == pytest.approx(rel.cer_ann(a, 3.0) - rel.cer_ann(b, 3.0), rel=1e-9)
    # the delta-method standard error agrees with the sampling spread of the estimate
    draws = []
    for s in range(300):
        g = np.random.default_rng(100 + s)
        bb = g.normal(0.0004, 0.01, n)
        aa = 0.5 * bb + g.normal(0.0006, 0.006, n)
        v = rel._moments(aa, bb)
        draws.append(rel._delta_and_gradient(v, "sharpe", 3.0)[0])
    assert d["se"] == pytest.approx(np.std(draws), rel=0.2)
    # identical series: no difference, nothing to reject
    same = rel.paired_difference_test(b, b.copy(), "sharpe", n_boot=300)
    assert same["delta"] == 0.0 and same["p_two_sided"] == 1.0 and same["p_one_sided"] == 1.0
    # a large true difference is found, in the right direction only
    strong = rel.paired_difference_test(b + 0.002, b, "sharpe", n_boot=600)
    assert strong["p_one_sided"] < 0.01 and strong["ci95_ann"][0] > 0
    assert rel.paired_difference_test(b, b + 0.002, "sharpe", n_boot=600)["p_one_sided"] > 0.95


def test_tests_hold_their_size_when_timing_has_no_value():
    """Premium proportional to variance: the optimal weight is constant, so the
    managed series has no alpha and no higher Sharpe ratio.  Small version of
    the calibration in the Phase 0 run (looser bound for Monte Carlo error)."""
    r, _ = vm.simulate_gjr_returns(160, 1500, premium="variance", seed=11)
    rej_alpha = rej_sharpe = 0
    for i in range(len(r)):
        m, b = vm.managed_from_returns(r[i])
        rej_alpha += rel.spanning_regression(m, b)["t_alpha"] > 1.645
        rej_sharpe += rel.paired_difference_test(m, b, "sharpe", n_boot=199, seed=i)["p_one_sided"] < 0.05
    assert rej_alpha / len(r) <= 0.11 and rej_sharpe / len(r) <= 0.11


def test_gjr_simulator_premium_structures():
    r, v = vm.simulate_gjr_returns(40, 4000, premium="variance", seed=3)
    assert np.sqrt(252 * r.var()) == pytest.approx(0.21, rel=0.15)
    # premium proportional to variance: mean return per unit of variance is the same in calm and turbulent halves
    lo, hi = v < np.median(v), v >= np.median(v)
    assert r[hi].mean() > 2.0 * r[lo].mean() and r[lo].mean() / v[lo].mean() == pytest.approx(r[hi].mean() / v[hi].mean(), rel=0.5)
    with pytest.raises(ValueError):
        vm.simulate_gjr_returns(2, 10, beta=0.99)


def test_participation_ratio_limits():
    assert rel.participation_ratio(np.eye(6)) == pytest.approx(6.0)
    assert rel.participation_ratio(np.ones((6, 6))) == pytest.approx(1.0)
    c = np.full((4, 4), 0.8) + 0.2 * np.eye(4)
    lam = np.r_[1 + 3 * 0.8, np.full(3, 0.2)]
    assert rel.participation_ratio(c) == pytest.approx(lam.sum() ** 2 / (lam**2).sum())


# ------------------------------------------ invariances on the committed daily table

DATA = Path(__file__).resolve().parents[1] / "research_runs" / "2026-11-23_phase0_volmanaged_ndx" / "data"


@pytest.mark.skipif(not (DATA / "ndx_daily_variance.csv").exists(), reason="Phase 0 daily table not present")
def test_committed_daily_table_conserves_returns_and_scalar_invariance_holds():
    d = pd.read_csv(DATA / "ndx_daily_variance.csv", index_col=0, parse_dates=True)
    mac = pd.read_csv(DATA / "macro_daily.csv", index_col=0, parse_dates=True)
    resid = (d["sum_r1"] + d["r_gap"] - d["r_cc"]).dropna()
    assert resid.abs().max() < 1e-9 and len(d) == 2653
    assert (d["rsm5"] + d["rsp5"] - d["rv5"]).abs().max() < 1e-9 and (d["jv5"] >= 0).all()
    f = vm.frictionless_excess(d, mac)
    sig = vm.variance_signals(d)["rv20"]
    ins = vm.managed_weights(sig, f, "insample")
    m = (ins["w"].shift(1) * f).dropna()
    assert m.std(ddof=1) == pytest.approx(f[m.index].std(ddof=1), rel=1e-12)
    a1 = rel.spanning_regression(m.to_numpy(), f[m.index].to_numpy())
    a2 = rel.spanning_regression(0.4 * m.to_numpy(), f[m.index].to_numpy())
    assert a2["appraisal_ratio"] == pytest.approx(a1["appraisal_ratio"], rel=1e-10)
    # same weights, costs switched on: wealth can only fall
    mkt = MarketDays.from_frames(d, mac, "2016-01-05")
    w = vm.managed_weights(sig, d["r_cc"], "causal")["w"].shift(1).reindex(mkt.index).clip(upper=1.0).to_numpy()
    free = vm.simulate_cfd_account(mkt, w, costs=FREE, tax=NO_TAX)
    paid = vm.simulate_cfd_account(mkt, w, costs=RetailCosts(), tax=NO_TAX)
    taxed = vm.simulate_cfd_account(mkt, w, costs=RetailCosts(), tax=GermanTax())
    assert free.terminal > paid.terminal > taxed.terminal
