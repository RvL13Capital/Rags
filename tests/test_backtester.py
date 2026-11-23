"""Order / position state machine on hand-scripted bars with hand-computed P&L.

Common set-up: price pinned at 100, a scripted LONG proposal with stop trigger
101, limit 101.3, protective stop 99 (1R = 2 points), ATR 2.  With 5,000 equity,
0.25% risk and the 1.25x gap buffer the size is 12.5 / (1.25 * 2) = 5.00 units.
"""

from __future__ import annotations

import numpy as np
import pytest

from ndxswing.config import CostModel, ModuleFlags, StrategyParams, ucits_ndx_etf
from tests.conftest import LONG, SHORT, flat_market, scripted_backtester, set_bar

S = 14  # signal bar: first bar of the third session (a Wednesday)
P = StrategyParams()
BASE = ModuleFlags()
EXITS = ModuleFlags(multi_stage_exits=True)


def run(bars, script, flags=BASE, params=P, **kw):
    return scripted_backtester(bars, script, **kw).run(params, flags)


def one_trade(res):
    assert len(res.trades) == 1
    return res.trades.iloc[0]


def test_intrabar_trigger_fill_then_stop_is_minus_one_r():
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)   # trades up through the trigger
    set_bar(bars, S + 2, 101.2, 101.4, 98.5, 98.8)     # trades down through the stop
    res = run(bars, {S: LONG})
    t = one_trade(res)
    assert t["units"] == pytest.approx(5.0)
    assert t["entry_px"] == pytest.approx(101.0)
    assert t["exit_px"] == pytest.approx(99.0)
    assert t["net_pnl"] == pytest.approx(-10.0)
    assert t["r_multiple"] == pytest.approx(-1.0)
    assert t["exit_reason"] == "stop"
    assert res.equity.iloc[-1] == pytest.approx(4990.0)


def test_gap_through_stop_fills_at_adverse_open_not_stop_price():
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    set_bar(bars, S + 2, 97.0, 97.5, 96.5, 97.2)       # opens 2 points below the 99 stop
    t = one_trade(run(bars, {S: LONG}))
    assert t["exit_px"] == pytest.approx(97.0)
    assert t["net_pnl"] == pytest.approx(5.0 * (97.0 - 101.0))
    assert t["r_multiple"] == pytest.approx(-2.0)
    assert t["exit_reason"] == "stop_gap"


def test_anti_chase_cancels_when_open_gaps_half_atr_beyond_trigger():
    bars = flat_market()
    set_bar(bars, S + 1, 102.1, 102.5, 102.0, 102.3)   # 102.1 > 101 + 0.5 * 2
    res = run(bars, {S: LONG})
    assert len(res.trades) == 0
    assert res.stats["cancel_anti_chase"] == 1
    assert res.equity.iloc[-1] == pytest.approx(5000.0)


def test_open_exactly_at_anti_chase_bound_is_not_cancelled():
    bars = flat_market()
    set_bar(bars, S + 1, 102.0, 102.2, 101.0, 101.1)   # open == 102.0, above the limit: rests, pulls back to 101.3
    t = one_trade(run(bars, {S: LONG}))
    assert t["entry_px"] == pytest.approx(101.3)


def test_unfilled_order_expires_after_two_bars():
    bars = flat_market()
    set_bar(bars, S + 3, 100.5, 101.5, 100.4, 101.2)   # would trigger, but the order is gone
    res = run(bars, {S: LONG})
    assert len(res.trades) == 0
    assert res.stats["expired"] == 1


def test_fill_on_second_live_bar():
    bars = flat_market()
    set_bar(bars, S + 2, 100.5, 101.5, 100.4, 101.2)
    t = one_trade(run(bars, {S: LONG}))
    assert t["entry_px"] == pytest.approx(101.0)


def test_open_between_trigger_and_limit_fills_at_the_open():
    bars = flat_market()
    set_bar(bars, S + 1, 101.2, 101.6, 101.1, 101.5)
    t = one_trade(run(bars, {S: LONG}))
    assert t["entry_px"] == pytest.approx(101.2)


def test_open_above_limit_without_pullback_does_not_fill():
    bars = flat_market()
    set_bar(bars, S + 1, 101.6, 101.9, 101.5, 101.8)   # never back to the 101.3 limit
    set_bar(bars, S + 2, 101.8, 101.9, 101.7, 101.8)
    res = run(bars, {S: LONG})
    assert len(res.trades) == 0
    assert res.stats["expired"] == 1


def test_partial_take_profit_then_breakeven_stop():
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)   # fill 101, 1R = 2, TP1 = 104
    set_bar(bars, S + 2, 101.2, 104.5, 101.1, 104.2)   # +1.5R reached: sell 2.5 at 104, stop -> 101
    set_bar(bars, S + 3, 104.2, 104.3, 100.5, 100.8)   # remainder stopped at breakeven
    t = one_trade(run(bars, {S: LONG}, EXITS))
    assert t["tp1_hit"]
    assert t["net_pnl"] == pytest.approx(2.5 * 3.0 + 2.5 * 0.0)
    assert t["exit_reason"] == "breakeven_stop"
    assert t["r_multiple"] == pytest.approx(0.75)


def test_chandelier_trail_after_two_r():
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    set_bar(bars, S + 2, 101.2, 104.6, 101.1, 104.5)   # TP1; best 104.6 < entry + 2R = 105
    set_bar(bars, S + 3, 104.5, 105.5, 104.4, 105.4)   # best 105.5 -> trail 105.5 - 1.5 * 2 = 102.5
    set_bar(bars, S + 4, 105.4, 105.4, 102.0, 102.2)   # trail hit
    t = one_trade(run(bars, {S: LONG}, EXITS))
    assert t["exit_reason"] == "trail_stop"
    assert t["net_pnl"] == pytest.approx(2.5 * 3.0 + 2.5 * 1.5)
    assert t["mfe_r"] == pytest.approx((105.5 - 101.0) / 2.0)


def test_stop_and_target_in_same_bar_assumes_stop_first():
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    set_bar(bars, S + 2, 101.2, 104.5, 98.9, 103.0)    # touches both 104 and 99
    t = one_trade(run(bars, {S: LONG}, EXITS))
    assert t["exit_reason"] == "stop"
    assert t["net_pnl"] == pytest.approx(-10.0)


def test_stops_are_monotone_trail_never_loosens():
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    set_bar(bars, S + 2, 101.2, 106.0, 101.1, 105.9)   # TP1 and trail to 103.0
    set_bar(bars, S + 3, 105.9, 105.9, 103.5, 103.6)   # lower bar: trail must stay at 103.0
    set_bar(bars, S + 4, 103.6, 103.7, 102.9, 103.2)   # hits 103.0
    t = one_trade(run(bars, {S: LONG}, EXITS))
    assert t["exit_reason"] == "trail_stop"
    assert t["net_pnl"] == pytest.approx(2.5 * 3.0 + 2.5 * 2.0)


def test_time_exit_at_session_close_of_last_holding_day():
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    res = run(bars, {S: LONG}, params=P.with_(max_hold_days=2))
    t = one_trade(res)
    assert t["exit_reason"] == "time_exit"
    assert t["sessions_held"] == 2
    assert res.trades["exit_ts"].iloc[0] == bars.index[3 * 7 + 6]   # last bar of the following session
    assert t["net_pnl"] == pytest.approx(5.0 * (100.0 - 101.0))


def test_five_day_time_exit_spans_about_35_bars():
    bars = flat_market(n_days=14)
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    t = one_trade(run(bars, {S: LONG}))
    assert t["exit_reason"] == "time_exit" and t["sessions_held"] == 5
    assert t["bars_held"] == 34   # entered on the 2nd bar of day 1, out on the last bar of day 5


def test_overnight_financing_one_night_and_weekend():
    costs = CostModel(spread_bps=0, slippage_bps=0, commission_bps=0, financing_markup=0.03,
                      default_benchmark_rate=0.03)
    for sig_bar, nights in ((S, 1), (4 * 7, 3)):       # Wednesday -> Thursday, Friday -> Monday
        bars = flat_market()
        set_bar(bars, sig_bar + 1, 100.5, 101.5, 100.4, 101.2)
        set_bar(bars, sig_bar + 6, 100.0, 100.0, 100.0, 100.0)
        t = one_trade(run(bars, {sig_bar: LONG}, params=P.with_(max_hold_days=2), costs=costs))
        # sizing is unchanged by financing; notional = 5 units * 100 (previous close)
        assert t["financing"] == pytest.approx(5.0 * 100.0 * 0.06 / 360.0 * nights)
        assert t["net_pnl"] == pytest.approx(-5.0 - t["financing"])


def test_short_receives_benchmark_minus_markup():
    costs = CostModel(spread_bps=0, slippage_bps=0, commission_bps=0, financing_markup=0.01,
                      default_benchmark_rate=0.04)
    bars = flat_market()
    set_bar(bars, S + 1, 99.5, 99.6, 98.5, 98.8)
    t = one_trade(run(bars, {S: SHORT}, params=P.with_(max_hold_days=2), costs=costs))
    assert t["financing"] == pytest.approx(5.0 * 100.0 * (0.01 - 0.04) / 360.0)   # a credit


def test_short_mirror_stop_and_gap():
    bars = flat_market()
    set_bar(bars, S + 1, 99.5, 99.6, 98.5, 98.8)       # fill 99
    set_bar(bars, S + 2, 98.8, 101.5, 98.7, 101.2)     # stop 101
    t = one_trade(run(bars, {S: SHORT}))
    assert (t["entry_px"], t["exit_px"]) == (pytest.approx(99.0), pytest.approx(101.0))
    assert t["net_pnl"] == pytest.approx(-10.0)
    bars = flat_market()
    set_bar(bars, S + 1, 99.5, 99.6, 98.5, 98.8)
    set_bar(bars, S + 2, 103.0, 103.5, 102.5, 103.2)   # gap over the stop
    t = one_trade(run(bars, {S: SHORT}))
    assert t["exit_reason"] == "stop_gap" and t["net_pnl"] == pytest.approx(-20.0)


def test_short_anti_chase_and_partial():
    bars = flat_market()
    set_bar(bars, S + 1, 97.9, 98.0, 97.5, 97.8)       # 97.9 < 99 - 1.0
    assert len(run(bars, {S: SHORT}).trades) == 0
    bars = flat_market()
    set_bar(bars, S + 1, 99.5, 99.6, 98.5, 98.8)       # fill 99, TP1 = 96
    set_bar(bars, S + 2, 98.8, 98.9, 95.5, 95.8)       # buy back 2.5 at 96, stop -> 99
    set_bar(bars, S + 3, 95.8, 99.5, 95.7, 99.2)       # breakeven
    t = one_trade(run(bars, {S: SHORT}, EXITS))
    assert t["net_pnl"] == pytest.approx(2.5 * 3.0)


def test_entry_bar_stop_policy():
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 98.9, 100.0)    # fills, touches the stop, closes above it
    t = one_trade(run(bars, {S: LONG}))
    assert t["exit_reason"] == "stop" and t["bars_held"] == 1 and t["net_pnl"] == pytest.approx(-10.0)
    res = run(bars, {S: LONG}, entry_bar_stop_policy="close")
    assert one_trade(res)["exit_reason"] == "time_exit"


def test_spread_and_slippage_enter_fill_prices_and_sizing():
    costs = CostModel(spread_bps=2.0, slippage_bps=1.0, commission_bps=0.0, financing_markup=0.0,
                      default_benchmark_rate=0.0)
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    set_bar(bars, S + 2, 101.2, 101.4, 98.5, 98.8)
    t = one_trade(run(bars, {S: LONG}, costs=costs))
    # risk per unit = 1.25 * 2 + 101 * (2e-4 spread + 2e-4 slippage) -> floor(12.5 / 2.5404 / 0.01) * 0.01
    assert t["units"] == pytest.approx(4.92)
    assert t["entry_px"] == pytest.approx(101.0 * (1 + 2e-4))
    assert t["exit_px"] == pytest.approx(99.0 * (1 - 2e-4))
    assert t["net_pnl"] == pytest.approx(4.92 * (99.0 * (1 - 2e-4) - 101.0 * (1 + 2e-4)))
    assert t["spread_cost"] == pytest.approx(4.92 * 1e-4 * (101.0 + 99.0))
    assert t["gross_pnl"] - t["spread_cost"] - t["slippage_cost"] == pytest.approx(t["net_pnl"])


def test_stressed_costs_scale_fills():
    costs = CostModel(spread_bps=2.0, slippage_bps=1.0, commission_bps=0.0, financing_markup=0.0,
                      default_benchmark_rate=0.0).stressed(1.5)
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    t = one_trade(run(bars, {S: LONG}, costs=costs))
    assert t["entry_px"] == pytest.approx(101.0 * (1 + 3e-4))


def test_commission_minimum_is_charged_per_order():
    costs = CostModel(spread_bps=0, slippage_bps=0, commission_bps=0.5, commission_min=1.0, financing_markup=0.0,
                      default_benchmark_rate=0.0)
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    set_bar(bars, S + 2, 101.2, 101.4, 98.5, 98.8)
    t = one_trade(run(bars, {S: LONG}, costs=costs))
    assert t["commission"] == pytest.approx(2.0)       # entry + exit, each at the ticket minimum
    assert t["net_pnl"] == pytest.approx(t["units"] * -2.0 - 2.0)


def test_no_new_signal_is_taken_while_in_position_or_order_working():
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    res = run(bars, {S: LONG, S + 1: LONG, S + 2: LONG})
    assert len(res.trades) == 1 and res.stats["orders"] == 1


def test_long_only_instrument_rejects_shorts():
    bars = flat_market()
    set_bar(bars, S + 1, 99.5, 99.6, 98.5, 98.8)
    from dataclasses import replace
    etf = replace(ucits_ndx_etf(price_ratio=1.0), quote_ccy="EUR")
    res = run(bars, {S: SHORT}, instrument=etf)
    assert len(res.trades) == 0 and res.stats["rejections"] == {"short_not_allowed": 1}


def test_fx_conversion_uses_usd_to_eur_orientation():
    """EURUSD = 1.25 USD per EUR  ->  X = 0.8 EUR per USD: a USD instrument
    needs 1/0.8 times the units for the same EUR risk and USD P&L converts at 0.8."""
    import pandas as pd
    from ndxswing.config import BacktestConfig, ndx_fractional_cfd
    from ndxswing.backtest.engine import EventDrivenBacktester
    from ndxswing.features.pipeline import build_feature_set
    from tests.conftest import FAST, ZERO_COSTS, ScriptedEngine

    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    set_bar(bars, S + 2, 101.2, 101.4, 98.5, 98.8)
    fs = build_feature_set(bars, FAST, n_jobs=1)
    fx = pd.Series(1.25, index=fs.sessions)
    bt = EventDrivenBacktester(ScriptedEngine(fs, ZERO_COSTS, {S: LONG}),
                               BacktestConfig(instrument=ndx_fractional_cfd(), costs=ZERO_COSTS), fx=fx)
    t = one_trade(bt.run(P, BASE))
    assert t["units"] == pytest.approx(6.25)            # 12.5 / (2.5 * 0.8)
    assert t["net_pnl"] == pytest.approx(6.25 * -2.0 * 0.8)
    with pytest.raises(ValueError, match="fx series"):
        EventDrivenBacktester(ScriptedEngine(fs, ZERO_COSTS, {}), BacktestConfig(instrument=ndx_fractional_cfd()))


def test_shadow_trades_pay_the_same_costs_as_real_ones(monkeypatch):
    """Paper (shadow) trades feed the Kelly sample, so they must carry spread,
    slippage, commission and financing exactly like live trades."""
    from dataclasses import replace
    from ndxswing.risk.engine import PortfolioRiskEngine

    costs = CostModel(spread_bps=2.0, slippage_bps=1.0, commission_bps=0.5, commission_min=0.2, financing_markup=0.03,
                      default_benchmark_rate=0.03)

    def scenario():
        bars = flat_market()
        set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
        return run(bars, {S: LONG}, params=P.with_(max_hold_days=3), costs=costs)

    live = one_trade(scenario())
    real_approve = PortfolioRiskEngine.approve
    monkeypatch.setattr(PortfolioRiskEngine, "approve", lambda self, pr, st: replace(real_approve(self, pr, st), shadow=True))
    res = scenario()
    paper = one_trade(res)
    assert paper["shadow"] and not live["shadow"]
    for col in ("entry_px", "exit_px", "units", "commission", "spread_cost", "slippage_cost", "financing", "net_pnl",
                "underlying_return", "r_multiple"):
        assert paper[col] == pytest.approx(live[col]), col
    assert live["financing"] > 0 and live["commission"] > 0 and live["spread_cost"] > 0
    assert (res.equity == 5000.0).all() and len(res.real_trades()) == 0      # nothing touches the account


def test_points_spread_fills():
    costs = CostModel(spread_points=0.2, slippage_bps=0.0, commission_bps=0.0, financing_markup=0.0,
                      default_benchmark_rate=0.0)
    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    set_bar(bars, S + 2, 101.2, 101.4, 98.5, 98.8)
    t = one_trade(run(bars, {S: LONG}, costs=costs))
    assert t["entry_px"] == pytest.approx(101.0 + 0.1 * 101.0 / 101.2)       # half of 0.2 points, quoted off the bar close
    assert t["exit_px"] == pytest.approx(99.0 - 0.1 * 99.0 / 98.8)


def test_unhedged_eur_etf_carries_fx_on_the_notional():
    """EUR-quoted ETF on a USD index: P&L = shares * (P1 * X1 - P0 * X0).
    Index falls 101 -> 99 while the dollar rallies from 0.80 to 1.00 EUR."""
    import pandas as pd
    from dataclasses import replace
    from ndxswing.backtest.engine import EventDrivenBacktester
    from ndxswing.config import BacktestConfig, ndx_fractional_cfd, ucits_ndx_etf
    from ndxswing.features.pipeline import build_feature_set
    from tests.conftest import FAST, ZERO_COSTS, ScriptedEngine

    bars = flat_market()
    set_bar(bars, S + 1, 100.5, 101.5, 100.4, 101.2)
    set_bar(bars, 5 * 7 + 1, 100.0, 100.2, 98.5, 98.8)                       # stop at 99 three sessions later
    fs = build_feature_set(bars, FAST, n_jobs=1)
    eurusd = pd.Series([1.25] * 3 + [1.0] * (len(fs.sessions) - 3), index=fs.sessions)
    etf = ucits_ndx_etf(price_ratio=1.0)
    bt = EventDrivenBacktester(ScriptedEngine(fs, ZERO_COSTS, {S: LONG}), BacktestConfig(instrument=etf, costs=ZERO_COSTS),
                               fx=eurusd)
    t = one_trade(bt.run(P, BASE))
    assert t["units"] == 6.0                                                 # floor(12.5 / (2.5 * 0.8)) whole shares
    assert t["net_pnl"] == pytest.approx(6.0 * (99.0 * 1.0 - 101.0 * 0.8))
    assert t["financing"] == 0.0 and t["sessions_held"] == 4
    cfd = EventDrivenBacktester(ScriptedEngine(fs, ZERO_COSTS, {S: LONG}),
                                BacktestConfig(instrument=replace(ndx_fractional_cfd(), lot_step=1.0, min_lot=1.0),
                                               costs=ZERO_COSTS), fx=eurusd)
    assert one_trade(cfd.run(P, BASE))["net_pnl"] == pytest.approx(6.0 * (99.0 - 101.0) * 1.0)


def test_drawdown_pause_flattens_and_blocks_real_entries():
    """Repeated gap losses push the drawdown through 8%: the system must stop
    sending real orders and later signals may only be paper (shadow) trades."""
    bars = flat_market(n_days=40)
    script = {}
    big = dict(direction=1, entry_stop=101.0, entry_limit=101.3, stop_price=100.8, d_stop=0.2)
    for k in range(12):
        s = 7 * (2 + 3 * k)
        script[s] = big
        set_bar(bars, s + 1, 100.95, 101.5, 100.9, 101.2)
        set_bar(bars, s + 2, 99.0, 99.2, 98.8, 99.0)   # gap far through a very tight stop
        set_bar(bars, s + 3, 100.0, 100.0, 100.0, 100.0)
    res = run(bars, script)
    real = res.real_trades()
    events = [e[1] for e in res.stats["breaker_events"]]
    assert "pause" in events or "deactivate" in events
    dd = 1 - res.equity.min() / 5000.0
    assert dd < 0.13                                    # losses stop accumulating once halted
    assert res.trades["shadow"].any()                   # diagnostics continue off-book
    assert len(real) < 12
    # shadow trades never touch equity
    assert res.equity.iloc[-1] == pytest.approx(5000.0 + real["net_pnl"].sum())


def test_equity_identity_and_trade_integrity_on_synthetic(synth_features, synth_market):
    from ndxswing.backtest.engine import EventDrivenBacktester
    from ndxswing.config import BacktestConfig
    from ndxswing.signal.engine import SignalEngine

    bt = EventDrivenBacktester(SignalEngine(synth_features, CostModel(), synth_market["events"], synth_market["rate"]),
                               BacktestConfig(), fx=synth_market["fx"])
    res = bt.run(P, ModuleFlags(multi_stage_exits=True, trend_flip_exit=True))
    t = res.real_trades()
    assert len(t) > 20
    assert res.equity.iloc[-1] - 5000.0 == pytest.approx(t["net_pnl"].sum(), abs=1e-6)
    assert (t["entry_ts"].to_numpy()[1:] >= t["exit_ts"].to_numpy()[:-1]).all()      # never two positions
    assert (t["units"] * 100 % 1 < 1e-6).all() or (np.abs(t["units"] * 100 - np.round(t["units"] * 100)) < 1e-6).all()
    assert (t["leverage"] <= 1.5 + 1e-9).all()
    assert (t["sessions_held"] <= 5).all()
    assert (t["initial_risk"] <= 5000 * 0.0025 * 1.2).all()
    # determinism and schedule equivalence
    res2 = bt.run(schedule=[(synth_features.sessions[0], P)], flags=ModuleFlags(multi_stage_exits=True, trend_flip_exit=True))
    assert np.allclose(res.equity.to_numpy(), res2.equity.to_numpy())
    # stressed costs can only hurt
    from dataclasses import replace
    bt15 = EventDrivenBacktester(SignalEngine(synth_features, CostModel().stressed(1.5), None, synth_market["rate"]),
                                 replace(BacktestConfig(), costs=CostModel().stressed(1.5)), fx=synth_market["fx"])
    res15 = bt15.run(P, ModuleFlags())
    base = bt.run(P, ModuleFlags())
    cost = lambda r: r.real_trades()[["spread_cost", "slippage_cost"]].to_numpy().sum() / max(len(r.real_trades()), 1)
    assert cost(res15) > cost(base)
