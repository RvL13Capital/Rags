"""Intraday noise-area momentum (Bot A): indicators, order handling, sizing and breakers on hand-built sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ndxswing.config import RiskLimits
from ndxswing.research import intraday as it
from ndxswing.research.intraday import IntradayCosts, SessionGrid

FREE = IntradayCosts("free", 0.0, 0.0, lot_step=1e-6)
PLAIN = dict(limits=None, max_leverage=None, compounding=False)
P0 = 10_000.0


def make_grid(n_prior: int = 20, fx: float = 1.25) -> SessionGrid:
    """``n_prior`` sessions drifting +/-0.2% from open to close, then one flat session to script."""
    d = n_prior + 1
    m = np.arange(390)
    o = np.full((d, 390), P0)
    c = o.copy()
    for i in range(n_prior):
        s = 1.0 if i % 2 == 0 else -1.0
        o[i], c[i] = P0 * (1 + s * 0.002 * m / 390), P0 * (1 + s * 0.002 * (m + 1) / 390)
    dates = pd.bdate_range("2022-03-01", periods=d)
    return SessionGrid(dates, o, np.maximum(o, c), np.minimum(o, c), c, np.ones((d, 390)), np.full(d, P0), np.full(d, fx), "unit-test")


def script(grid: SessionGrid, closes: np.ndarray, day: int = -1) -> None:
    c = np.asarray(closes, dtype=float)
    o = np.r_[P0, c[:-1]]
    grid.open[day], grid.close[day], grid.high[day], grid.low[day] = o, c, np.maximum(o, c), np.minimum(o, c)


def breakout(after: float = 10_080.0) -> np.ndarray:
    """Ramps to 10,060 by the 10:00 close (far above the noise area), then sits at ``after``."""
    return np.r_[np.linspace(10_002.0, 10_060.0, 30), np.full(360, after)]


# -------------------------------------------------------------------- indicators


def test_noise_area_inputs_by_hand_and_causal():
    g = make_grid()
    inp = it.noise_area_inputs(g, "twap")
    move = np.abs(g.close[:, 29::30] / g.open[:, [0]] - 1)
    assert np.isnan(inp.sigma[:14]).all() and inp.sigma.shape == (21, 13)
    assert inp.sigma[20] == pytest.approx(move[6:20].mean(axis=0))            # the 14 sessions BEFORE the day
    assert inp.upper[20, 0] == pytest.approx(P0 * (1 + inp.sigma[20, 0])) and inp.lower[20, 0] == pytest.approx(P0 * (1 - inp.sigma[20, 0]))
    g.prev_close[20] = 10_050.0                                               # gap down: the band hangs from the higher of the two
    wide = it.noise_area_inputs(g, "twap")
    assert wide.upper[20, 0] == pytest.approx(10_050.0 * (1 + wide.sigma[20, 0])) and wide.lower[20, 0] == pytest.approx(P0 * (1 - wide.sigma[20, 0]))
    # nothing about a later session reaches an earlier row
    h = make_grid()
    script(h, breakout())
    late = it.noise_area_inputs(h, "twap")
    for a, b in ((inp.sigma, late.sigma), (inp.anchor, late.anchor), (inp.atr30, late.atr30)):
        np.testing.assert_array_equal(a[:20], b[:20])
    # ATR: Wilder smoothing of the 30-minute true range, the overnight gap in the first bar
    tr_first = max(g.high[1, :30].max() - g.low[1, :30].min(), abs(g.high[1, :30].max() - g.close[0, -1]), abs(g.low[1, :30].min() - g.close[0, -1]))
    assert tr_first == pytest.approx(abs(g.low[1, :30].min() - g.close[0, -1])) and tr_first > 15.0
    assert np.isnan(inp.atr30.ravel()[:13]).all() and np.isfinite(inp.atr30.ravel()[13:]).all()


def test_vwap_follows_volume_and_twap_does_not():
    g = make_grid()
    script(g, breakout())
    same = it.noise_area_inputs(g, "vwap").anchor
    np.testing.assert_allclose(same, it.noise_area_inputs(g, "twap").anchor)   # constant volume: identical
    g.volume[20, 30:] = 50.0                                                  # heavy volume after the ramp, at 10,080
    v, t = it.noise_area_inputs(g, "vwap").anchor[20], it.noise_area_inputs(g, "twap").anchor[20]
    assert v[100] > t[100] + 5.0 and v[29] == pytest.approx(t[29])
    typ = (g.high[20] + g.low[20] + g.close[20]) / 3
    assert t[100] == pytest.approx(typ[:101].mean()) and v[100] == pytest.approx((typ[:101] * g.volume[20, :101]).sum() / g.volume[20, :101].sum())
    g.volume[20] = 0.0                                                        # a session without volume falls back to equal weights
    np.testing.assert_allclose(it.noise_area_inputs(g, "vwap").anchor[20], t)
    with pytest.raises(ValueError):
        it.noise_area_inputs(g, "median")


# ------------------------------------------------------------------ order handling


def one(res: it.IntradayResult) -> pd.Series:
    assert len(res.trades) == 1, res.trades
    return res.trades.iloc[0]


def test_long_entry_stop_distance_and_time_exit_by_hand():
    g = make_grid()
    script(g, breakout())
    inp = it.noise_area_inputs(g, "twap")
    assert g.close[20, 29] > inp.upper[20, 0]
    t = one(it.run_noise_area(g, inp, FREE, b_risk=0.005, **PLAIN))
    d_stop = max(abs(10_060.0 - inp.anchor[20, 29]), 1.2 * inp.atr30[20, 0])
    assert d_stop == pytest.approx(10_060.0 - inp.anchor[20, 29]) and 25 < d_stop < 35
    assert t["entry_minute"] == 30 and t["entry_px"] == 10_060.0 and t["r_px"] == pytest.approx(d_stop)
    assert t["units"] == pytest.approx(0.005 * 5000 * 1.25 / d_stop, abs=1e-5)        # EUR 25 of risk, USD per point
    assert t["exit_minute"] == 385 and t["exit_reason"] == "time" and t["exit_px"] == 10_080.0
    assert t["net_pnl"] == pytest.approx(t["units"] * 20.0 / 1.25) and t["r_multiple"] == pytest.approx(20.0 / d_stop)
    assert t["entry_ts"] == g.dates[20] + pd.Timedelta(hours=10) and t["exit_ts"] == g.dates[20] + pd.Timedelta(hours=15, minutes=55)


def test_exit_when_the_close_falls_back_under_the_anchor():
    g = make_grid()
    script(g, np.r_[breakout()[:150], np.full(240, 10_058.0)])                 # sits at 10,080, then 10,058 from 12:00
    inp = it.noise_area_inputs(g, "twap")
    assert inp.anchor[20, 179] > 10_058.0 > 10_060.0 - (10_060.0 - inp.anchor[20, 29])
    t = one(it.run_noise_area(g, inp, FREE, b_risk=0.005, **PLAIN))
    assert t["exit_reason"] == "signal" and t["exit_minute"] == 180 and t["exit_px"] == 10_058.0   # decided at the 12:30 close
    assert t["r_multiple"] == pytest.approx(-2.0 / t["r_px"])


def test_hard_stop_fills_at_the_stop_or_at_a_worse_open():
    g = make_grid()
    script(g, np.r_[breakout()[:60], np.full(330, P0)])                        # trades down through the stop inside 10:30
    inp = it.noise_area_inputs(g, "twap")
    stop = 10_060.0 - (10_060.0 - inp.anchor[20, 29])
    t = one(it.run_noise_area(g, inp, FREE, b_risk=0.005, **PLAIN))
    assert t["exit_reason"] == "stop" and t["exit_minute"] == 60 and t["exit_px"] == pytest.approx(stop) and t["r_multiple"] == pytest.approx(-1.0)
    g.open[20, 60], g.high[20, 60] = 10_010.0, 10_010.0                        # the minute OPENS below the stop
    t = one(it.run_noise_area(g, inp, FREE, b_risk=0.005, **PLAIN))
    assert t["exit_px"] == 10_010.0 and t["r_multiple"] < -1.5


def test_short_mirror_and_costs_in_both_units():
    g = make_grid()
    script(g, 2 * P0 - breakout())                                            # mirror image: down to 9,940, then 9,920
    inp = it.noise_area_inputs(g, "twap")
    bps = IntradayCosts("bps", 0.20e-4, 0.20e-4, lot_step=1e-6)
    t = one(it.run_noise_area(g, inp, bps, b_risk=0.005, **PLAIN))
    assert t["direction"] == -1 and t["entry_px"] == pytest.approx(9_940.0 * (1 - 0.40e-4)) and t["exit_px"] == pytest.approx(9_920.0 * (1 + 0.40e-4))
    assert t["gross_r"] == pytest.approx(20.0 / t["r_px"]) and t["r_multiple"] == pytest.approx((20.0 - 0.40e-4 * (9_940.0 + 9_920.0)) / t["r_px"])
    pts = IntradayCosts("points", 0.5, 0.5, fx_fee=0.007, lot_step=1e-6, in_points=True)
    t = one(it.run_noise_area(g, inp, pts, b_risk=0.005, **PLAIN))
    assert t["entry_px"] == pytest.approx(9_939.0) and t["exit_px"] == pytest.approx(9_921.0)          # one point a leg
    eur = t["units"] * 18.0 / 1.25
    assert t["fx_fee"] == pytest.approx(0.007 * eur) and t["net_pnl"] == pytest.approx(eur * 0.993)
    assert pts.stressed(1.5).leg(10_000.0) == pytest.approx(1.5e-4) and pts.leg(20_000.0) == pytest.approx(0.5e-4)


def test_cost_schedules_match_the_specification():
    p, c = it.cost_schedules("pepperstone"), it.cost_schedules("capital")
    assert p["I"].half_spread == pytest.approx(0.20e-4) and p["I"].slippage == 0.20e-4 and not p["I"].in_points and p["I"].lot_step == 0.10
    assert c["I"].half_spread == pytest.approx(0.355e-4) and c["I"].fx_fee == 0.007 and c["I"].lot_step == 0.001
    assert (p["II"].half_spread, p["II"].slippage, p["II"].in_points) == (0.5, 0.5, True) and c["II"].half_spread == 0.9
    assert p["III"].stress == 1.5 and p["III"].leg(20_000.0) == pytest.approx(1.5 * 1.0 / 20_000.0)
    assert p["I"].leg(12_345.0) == pytest.approx(0.40e-4)


def test_two_trades_a_day_no_entry_after_1430_and_skips():
    g = make_grid()
    c = np.r_[breakout()[:60], np.full(30, P0), np.full(60, 10_090.0), np.full(30, P0), np.full(210, 10_095.0)]
    script(g, c)                                    # stop-out, re-entry at 11:30, stop-out again, a third breakout at 12:30
    inp = it.noise_area_inputs(g, "twap")
    res = it.run_noise_area(g, inp, FREE, b_risk=0.005, **PLAIN)
    assert len(res.trades) == 2 and res.trades["entry_minute"].tolist() == [30, 120]
    assert res.trades["exit_reason"].tolist() == ["stop", "stop"] and g.close[20, 209] > inp.upper[20, 6]   # the third one is not taken
    assert len(it.run_noise_area(g, inp, FREE, b_risk=0.005, max_trades=1, **PLAIN).trades) == 1
    late = make_grid()
    script(late, np.r_[np.full(301, P0), np.full(89, 10_090.0)])              # breaks out at 14:31: first seen at the 15:00 close
    assert len(it.run_noise_area(late, it.noise_area_inputs(late, "twap"), FREE, b_risk=0.005, **PLAIN).trades) == 0
    edge = make_grid()
    script(edge, np.r_[np.full(299, P0), np.full(91, 10_090.0)])              # ... and at 14:29: taken at 14:30
    assert one(it.run_noise_area(edge, it.noise_area_inputs(edge, "twap"), FREE, b_risk=0.005, **PLAIN))["entry_minute"] == 300
    # stop distance under 0.20% of price: no trade
    tight = make_grid()
    script(tight, np.r_[np.linspace(10_001.0, 10_012.0, 30), np.full(360, 10_014.0)])
    r = it.run_noise_area(tight, it.noise_area_inputs(tight, "twap"), FREE, b_risk=0.005, **PLAIN)
    assert len(r.trades) == 0 and r.stats["skipped_min_stop"] >= 1


def test_lot_pinned_to_reference_level_and_leverage_cap():
    g = make_grid()
    script(g, breakout())
    inp = it.noise_area_inputs(g, "twap")
    d_stop = 10_060.0 - inp.anchor[20, 29]
    pep = IntradayCosts("pepperstone", 0.0, 0.0, lot_step=0.10)
    step = 0.10 * it.LOT_REFERENCE / 10_060.0                                 # a 0.10 lot at 25,288 is 0.251 units here
    t = one(it.run_noise_area(g, inp, pep, b_risk=0.005, **PLAIN))
    assert t["units"] == pytest.approx(np.floor(0.005 * 5000 * 1.25 / d_stop / step) * step) and t["units"] / step == pytest.approx(round(t["units"] / step))
    small = it.run_noise_area(g, inp, pep, b_risk=0.001, **PLAIN)             # EUR 5 of risk cannot carry one lot
    assert len(small.trades) == 0 and small.stats["skipped_min_lot"] >= 1
    # EUR 25 of risk on a 0.29% stop is 1.73x the account: capped at 1.5x, so less than the budget is at risk
    full = one(it.run_noise_area(g, inp, FREE, b_risk=0.005, limits=None, max_leverage=None))
    capped_run = it.run_noise_area(g, inp, FREE, b_risk=0.005, limits=None, max_leverage=1.5)
    capped = one(capped_run)
    assert full["leverage"] == pytest.approx(0.005 * 10_060.0 / d_stop, rel=1e-4) and full["leverage"] > 1.5
    assert capped["leverage"] == pytest.approx(1.5, abs=1e-4) and capped["binding"] == "max_leverage" and capped_run.stats["leverage_capped"] == 1
    assert capped["initial_risk"] < 25.0 and capped["r_multiple"] == pytest.approx(full["r_multiple"])


def test_breakers_halt_entries_after_a_one_percent_day_and_compound_risk():
    g = make_grid()
    c = np.r_[breakout()[:60], np.full(30, 10_020.0), np.full(300, 10_090.0)]
    script(g, c)
    g.open[20, 60], g.high[20, 60] = 9_990.0, 10_020.0                         # first trade gaps far through its stop: about -2.4R
    inp = it.noise_area_inputs(g, "twap")
    off = it.run_noise_area(g, inp, FREE, b_risk=0.005, limits=None, max_leverage=None)
    assert len(off.trades) == 2 and off.trades["r_multiple"].iloc[0] < -2.0
    on = it.run_noise_area(g, inp, FREE, b_risk=0.005, limits=RiskLimits(), max_leverage=None)
    assert len(on.trades) == 1 and on.stats["halted_entries"] >= 1 and on.stats["intraday_halts"] == 1
    assert on.equity.iloc[-1] == pytest.approx(5000.0 + on.trades["net_pnl"].sum()) and on.equity.iloc[-1] < 4950.0
    # compounding: the second trade risks 0.5% of the equity left after the first
    assert off.trades["initial_risk"].iloc[1] == pytest.approx(0.005 * (5000.0 + off.trades["net_pnl"].iloc[0]), rel=1e-4)
    flat = it.run_noise_area(g, inp, FREE, b_risk=0.005, **PLAIN)
    assert flat.trades["initial_risk"].iloc[1] == pytest.approx(25.0, rel=1e-4)
    # the result converts into the shape the tearsheet expects, and a window re-bases equity
    bt = on.as_backtest_result()
    assert len(bt.equity) == 21 and bt.daily_returns().iloc[-1] == pytest.approx(on.equity.iloc[-1] / 5000.0 - 1)
    assert on.window(g.dates[20]).initial_equity == 5000.0 and len(on.window(g.dates[20]).equity) == 1


def test_drawdown_tiers_scale_risk_against_the_true_high_water_mark():
    g = make_grid(n_prior=40)
    losing = np.r_[breakout()[:60], np.full(330, P0)]
    for day in range(25, 41):
        script(g, losing, day=day)
        g.open[day, 60], g.high[day, 60] = 9_950.0, 9_950.0                   # every day one trade that loses about 3.7R
    inp = it.noise_area_inputs(g, "twap")
    r = it.run_noise_area(g, inp, FREE, b_risk=0.0025, limits=RiskLimits(), max_leverage=None)
    t = r.trades
    assert len(t) >= 12 and (t["entry_minute"] == 30).all()
    budget = (t["initial_risk"] / r.equity.shift(1).fillna(5000.0).reindex(t["entry_session"]).to_numpy()).to_numpy()
    assert budget[0] == pytest.approx(0.0025, rel=1e-3)
    assert 0.5 * 0.0025 == pytest.approx(budget[3], rel=1e-3)                 # more than 2% lost over five sessions: half risk
    assert budget.min() == pytest.approx(0.25 * 0.0025, rel=1e-3)             # 6% below the high-water mark: a quarter
    assert {float(v) for v in np.round(budget / 0.0025, 2)} == {1.0, 0.5, 0.25} and (np.diff(budget) <= 1e-7).all()
    assert 0.06 < 1 - r.equity.min() / 5000.0 < 0.08


# --------------------------------------------------------------------------- data


def test_session_grid_from_minute_bars_and_holdout_reader(tmp_path):
    days = pd.bdate_range("2020-12-28", "2021-01-06")
    rows = []
    for d in days:
        n = 210 if d == days[2] else 390                                      # one shortened session
        ts = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=n, freq="min", tz="America/New_York")
        px = 100.0 + np.arange(n) * 0.01
        rows.append(pd.DataFrame({"ts": ts.tz_convert("UTC"), "open": px, "high": px + 0.02, "low": px - 0.02, "close": px + 0.01, "volume": 10.0}))
    bars = pd.concat(rows).reset_index(drop=True)
    bars = bars.drop(index=5)                                                 # one missing minute on the first day
    bars["ts"] = bars["ts"].astype("datetime64[ns, UTC]")
    f = tmp_path / "bars.parquet"
    bars.to_parquet(f)
    early = it.read_minute_bars(f, before="2021-01-01")
    assert early.index.max() < pd.Timestamp("2021-01-01", tz="UTC") and len(it.read_minute_bars(f)) == len(bars)
    m = it.read_minute_bars(f)
    n_min = m.groupby(m.index.tz_convert("America/New_York").tz_localize(None).normalize()).size()
    sessions = pd.DataFrame({"n_min": n_min, "close": m["close"].groupby(m.index.tz_convert("America/New_York").tz_localize(None).normalize()).last()})
    g = SessionGrid.from_minute_bars(m, sessions, pd.Series(1.2, index=days))
    assert len(g) == len(days) - 1 and days[2] not in g.dates                  # the short session is left out ...
    k = g.dates.get_loc(days[3])
    assert g.prev_close[k] == pytest.approx(sessions.loc[days[2], "close"])    # ... but still supplies the previous close
    assert g.close[0, 5] == g.close[0, 4] and g.volume[0, 5] == 0.0 and g.open.shape == (len(g), 390)
    assert np.isnan(g.prev_close[0]) and (g.fx == 1.2).all()
