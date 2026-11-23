"""Signal engine: spec formulas, gates and the zero look-ahead contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ndxswing.config import CostModel, ModuleFlags, StrategyParams
from ndxswing.features.pipeline import FeatureConfig, FeatureSet, build_feature_set
from ndxswing.signal.engine import SignalEngine

P = StrategyParams()


def test_entry_levels_follow_the_spec(synth_features):
    eng = SignalEngine(synth_features)
    prep = eng.prepare(P, ModuleFlags())
    hb = synth_features.hourly
    idx = np.flatnonzero(prep.direction != 0)
    assert len(idx) > 50
    for i in idx[:: max(1, len(idx) // 40)]:
        d, a = int(prep.direction[i]), prep.atr[i]
        hi20 = hb["high"].iloc[i - 20:i].max()
        lo20 = hb["low"].iloc[i - 20:i].min()
        day = synth_features.daily.iloc[eng.sess_idx[i] - 1]             # previous session's daily row
        assert d == (1 if day["ema20"] > day["ema50"] else -1)
        if d > 0:
            assert hb["close"].iloc[i] > hi20
            assert prep.entry_stop[i] == pytest.approx(hi20 + 0.10 * a)
            assert prep.entry_limit[i] == pytest.approx(prep.entry_stop[i] + 0.15 * a)
            d_struct = prep.entry_stop[i] - hb["low"].iloc[i - 4:i + 1].min()
        else:
            assert hb["close"].iloc[i] < lo20
            assert prep.entry_stop[i] == pytest.approx(lo20 - 0.10 * a)
            d_struct = hb["high"].iloc[i - 4:i + 1].max() - prep.entry_stop[i]
        assert prep.d_stop[i] == pytest.approx(max(1.5 * a, d_struct))
        assert prep.d_stop[i] <= 2.25 * a + 1e-9                        # wider stops are skipped, never tightened
        assert prep.stop_price[i] == pytest.approx(prep.entry_stop[i] - d * prep.d_stop[i])
        prop = eng.evaluate(prep, int(i))
        assert prop.direction == d and prop.bar_index == i and prop.entry_stop == prep.entry_stop[i]
    assert eng.evaluate(prep, int(np.flatnonzero(prep.direction == 0)[0])) is None
    with pytest.raises(Exception):
        prop.direction = 0                                               # proposals are immutable


def test_signals_have_no_look_ahead(synth_market):
    """Signals for bars <= cut are identical with and without the future."""
    cfg = FeatureConfig(with_regime=False, with_hurst=False, with_garch=False, with_evt=False, with_vpin=True)
    hb = synth_market["hourly"]
    flags = ModuleFlags(vpin_filter=True, spread_gate=True, event_blackout=True, trend_flip_exit=True)
    full = SignalEngine(build_feature_set(hb, cfg, n_jobs=1), CostModel(), synth_market["events"]).prepare(P, flags)
    n_cut = len(hb) - 7 * 60 - 3                                         # cut in the middle of a session
    part_fs = build_feature_set(hb.iloc[:n_cut], cfg, n_jobs=1)
    part = SignalEngine(part_fs, CostModel(), synth_market["events"]).prepare(P, flags)
    m = len(part.direction) - 1      # the event window of the very last bar depends on the next bar's calendar time
    assert (full.direction[:m] != 0).sum() > 10
    for name in ("direction", "entry_stop", "entry_limit", "stop_price", "d_stop", "m_liq", "sigma_1d",
                 "exit_long", "exit_short"):
        a, b = getattr(full, name)[:m], getattr(part, name)[:m]
        sel = full.direction[:m] != 0 if name not in ("direction", "exit_long", "exit_short") else slice(None)
        assert np.allclose(np.asarray(a, float)[sel], np.asarray(b, float)[sel], equal_nan=True), name


def _fake_features(synth_features, **daily_cols) -> FeatureSet:
    d = synth_features.daily.copy()
    for k, v in daily_cols.items():
        d[k] = v
    return FeatureSet(synth_features.hourly, d, synth_features.config, "test")


def test_daily_features_are_lagged_one_full_session(synth_features):
    marker = np.arange(len(synth_features.daily), dtype=float)
    eng = SignalEngine(_fake_features(synth_features, marker=marker))
    seen = eng.lagged_daily("marker")
    assert np.isnan(seen[:7]).all()
    assert (seen[7:] == eng.sess_idx[7:] - 1).all()                      # every bar of day d sees row d-1


def test_regime_gate_and_multiplier(synth_features):
    n = len(synth_features.daily)
    fs = _fake_features(synth_features, p_trend=0.80, p_side=0.15, p_stress=0.05, conf=0.6)
    base = SignalEngine(synth_features).prepare(P, ModuleFlags())
    on = SignalEngine(fs).prepare(P, ModuleFlags(regime_gate=True))
    assert (on.direction == base.direction)[14:].all() and (on.m_reg[14:] == 1.0).all()
    mid = SignalEngine(_fake_features(synth_features, p_trend=0.70, p_stress=0.05, conf=0.6)).prepare(
        P, ModuleFlags(regime_gate=True))
    assert (mid.m_reg[mid.direction != 0] == 0.75).all()
    for kw in (dict(p_trend=0.60, p_stress=0.05, conf=0.6), dict(p_trend=0.80, p_stress=0.25, conf=0.6),
               dict(p_trend=0.80, p_stress=0.05, conf=0.30)):
        off = SignalEngine(_fake_features(synth_features, **kw)).prepare(P, ModuleFlags(regime_gate=True))
        assert (off.direction == 0).all()
    stress = SignalEngine(_fake_features(synth_features, p_trend=0.5, p_stress=0.35, conf=0.6)).prepare(
        P, ModuleFlags(regime_gate=True))
    assert stress.exit_long[14:].all() and stress.exit_short[14:].all()  # hard stress lock exits open exposure
    assert n > 0


def test_hurst_gate_thresholds(synth_features):
    f = ModuleFlags(hurst_gate=True)
    strong = SignalEngine(_fake_features(synth_features, hurst=0.62, hurst_lcb=0.53)).prepare(P, f)
    assert (strong.m_hurst[strong.direction != 0] == 1.0).all() and (strong.direction != 0).any()
    moderate = SignalEngine(_fake_features(synth_features, hurst=0.57, hurst_lcb=0.51)).prepare(P, f)
    assert (moderate.m_hurst[moderate.direction != 0] == 0.8).all()
    for h, lcb in ((0.52, 0.51), (0.62, 0.50), (0.44, 0.40)):
        assert (SignalEngine(_fake_features(synth_features, hurst=h, hurst_lcb=lcb)).prepare(P, f).direction == 0).all()


def test_arma_z_and_net_expectancy_gate(synth_features):
    f = ModuleFlags(arma_z_filter=True)
    cols = lambda mu, sig: {k: v for h in (1, 2, 3, 5) for k, v in
                            ((f"mu_{h}", mu), (f"sigma_{h}", sig), (f"z_{h}", mu / sig))}
    base = SignalEngine(synth_features).prepare(P, ModuleFlags())
    strong_up = SignalEngine(_fake_features(synth_features, **cols(0.02, 0.03))).prepare(P, f)
    assert ((strong_up.direction == 1) == (base.direction == 1))[14:].all()      # longs pass, shorts blocked
    assert (strong_up.direction != -1).all()
    weak = SignalEngine(_fake_features(synth_features, **cols(0.003, 0.03))).prepare(P, f)   # Z = 0.1 < 0.35
    assert (weak.direction == 0).all()
    # Z clears the hurdle but the forecast cannot pay for costs + 0.25R
    thin = SignalEngine(_fake_features(synth_features, **cols(0.0004, 0.001))).prepare(P, f)
    assert thin.funnel["arma_z"] > 0 and thin.funnel["net_expectancy"] == 0


def test_score_and_missing_inputs(synth_features):
    fs = _fake_features(synth_features, p_trend=0.90, p_side=0.05, p_stress=0.05, conf=0.7, hurst=0.65, hurst_lcb=0.55)
    prep = SignalEngine(fs).prepare(P, ModuleFlags(regime_gate=True, hurst_gate=True, score_sizing=True))
    s = prep.score[prep.direction != 0]
    assert ((s >= 0) & (s <= 1)).all() and s == pytest.approx(1.0)
    with pytest.raises(ValueError, match="hurst_gate"):
        SignalEngine(synth_features).prepare(P, ModuleFlags(hurst_gate=True))    # no silent placeholder
    with pytest.raises(ValueError, match="event_blackout"):
        SignalEngine(synth_features).prepare(P, ModuleFlags(event_blackout=True))


def test_event_blackout_covers_intraday_and_overnight(synth_features):
    hb = synth_features.hourly
    day = hb["session"].iloc[70]
    fomc = (day + pd.Timedelta(hours=14)).tz_localize("America/New_York").tz_convert("UTC")
    nxt = hb["session"].iloc[77]
    cpi = (nxt + pd.Timedelta(hours=8, minutes=30)).tz_localize("America/New_York").tz_convert("UTC")
    ev = pd.DataFrame({"ts": [fomc, cpi], "name": ["FOMC", "CPI"], "tier": [1, 1]})
    eng = SignalEngine(synth_features, events=ev)
    blk = eng.event_block[70:77]                                         # bars start 09:30, 10, 11, 12, 13, 14, 15 ET
    # closes at 12:00 and 13:00 have 14:00 inside the next two hours; the 13:00-14:00 bar closes AT the event
    assert blk.tolist() == [False, False, True, True, False, False, True]
    # ... and the last bar of the day is blocked because the order would be live through next morning's CPI
    assert not eng.event_block[77:84].any()
