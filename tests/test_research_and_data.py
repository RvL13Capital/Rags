"""Walk-forward folds, optimisation plumbing, ablation, data store and ingestion."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ndxswing.config import MODULE_ORDER, BacktestConfig, ModuleFlags, StrategyParams
from ndxswing.data import ingest as ing
from ndxswing.data.futures import panama_adjust
from ndxswing.data.store import MarketDataStore
from ndxswing.data.synthetic import SYNTHETIC_SOURCE, SyntheticSpec, generate_synthetic_market
from ndxswing.research import walkforward as wfm
from ndxswing.research.ablation import AblationConfig, SequentialAblationEngine
from ndxswing.research.walkforward import (
    BacktestContext, PurgedWalkForwardRunner, WalkForwardConfig, active_space, apply_overrides, make_folds, neighbours,
)

# ----------------------------------------------------------------------- folds


def test_fold_geometry_purge_and_embargo():
    sessions = pd.bdate_range("2012-01-02", "2026-10-30")
    cfg = WalkForwardConfig()
    folds = make_folds(sessions, cfg)
    assert len(folds) > 35
    pos = {d: i for i, d in enumerate(sessions)}
    for f in folds:
        assert f.train_start < f.train_end < f.val_start < f.val_end < f.oos_start < f.oos_end
        # 5 purge + 5 embargo sessions lie strictly between consecutive windows
        assert pos[f.val_start] - pos[f.train_end] - 1 == 10
        assert pos[f.oos_start] - pos[f.val_end] - 1 == 10
        assert 740 <= pos[f.train_end] - pos[f.train_start] + 1 <= 790          # ~36 months
        assert 245 <= pos[f.val_end] - pos[f.val_start] + 1 <= 270              # ~12 months
        assert 120 <= pos[f.oos_end] - pos[f.oos_start] + 1 <= 135
    for a, b in zip(folds, folds[1:]):
        step = pos[b.train_start] - pos[a.train_start]
        assert 58 <= step <= 68                                                 # shifted by ~3 months
        assert b.oos_start > a.oos_start and b.val_end < b.oos_start
    assert folds[-1].oos_end <= sessions[-1]
    assert make_folds(sessions[:900], cfg) == []
    held = make_folds(sessions, cfg, end=pd.Timestamp("2022-12-31"))
    assert held and held[-1].oos_end <= pd.Timestamp("2022-12-31")


def test_search_space_matches_spec_ranges_and_modules():
    base = active_space(ModuleFlags())
    assert set(base) == {"donchian_window", "atr_stop_mult", "max_hold_days"}
    full = active_space(ModuleFlags.full())
    assert full["hurst_min"][1:3] == (0.53, 0.60) and full["p_trend_min"][1:3] == (0.60, 0.80)
    assert full["vpin_block_pct"][1:3] == (0.90, 0.99) and full["donchian_window"][1:3] == (16, 40)
    assert full["atr_stop_mult"][1:3] == (1.2, 2.2) and full["z_hurdle"][1:3] == (0.25, 0.75)
    assert full["max_hold_days"][1:3] == (1, 5) and full["kelly_fraction"][1:3] == (0.10, 0.25)
    theta = {"donchian_window": 16, "atr_stop_mult": 1.7, "max_hold_days": 5}
    nb = neighbours(theta, base)
    assert {"donchian_window": 18, "atr_stop_mult": 1.7, "max_hold_days": 5} in nb
    assert all(16 <= t["donchian_window"] <= 40 and 1 <= t["max_hold_days"] <= 5 for t in nb)
    assert len(nb) == 4                                                         # two dimensions sit on a bound
    p = apply_overrides(StrategyParams(), {"vpin_block_pct": 0.905})
    assert p.vpin_halve_pct <= p.vpin_block_pct


def test_default_tuning_policy_freezes_structure():
    """Default: geometry frozen, at most three module thresholds tuned; the
    challengers (Hurst, ARMA Z) sit at the end of the ablation order."""
    cfg = WalkForwardConfig()
    assert (cfg.train_months, cfg.val_months, cfg.oos_months, cfg.step_months, cfg.tune) == (36, 12, 6, 3, "thresholds")
    assert active_space(ModuleFlags(), cfg.tune) == {}                          # baseline: nothing to tune
    assert list(active_space(ModuleFlags(regime_gate=True), cfg.tune)) == ["p_trend_min"]
    assert active_space(ModuleFlags(regime_gate=True, regime_model="rv20"), cfg.tune) == {}
    full = active_space(ModuleFlags.full(), cfg.tune, cfg.max_tuned)
    assert len(full) == 3 and not set(full) & set(wfm.STRUCTURAL)
    assert active_space(ModuleFlags.full(), "none") == {}
    assert MODULE_ORDER[-2:] == ("hurst_gate", "arma_z_filter")
    assert MODULE_ORDER.index("garch_vol") < MODULE_ORDER.index("hurst_gate")
    assert not any(getattr(ModuleFlags.baseline(), m) for m in MODULE_ORDER)
    with pytest.raises(ValueError):
        WalkForwardConfig(tune="everything")


# -------------------------------------------------------- walk-forward (small)


@pytest.fixture(scope="module")
def small_ctx():
    from ndxswing.features.pipeline import FeatureConfig, build_feature_set

    m = generate_synthetic_market(SyntheticSpec(end="2016-12-30", seed=8))
    cfg = FeatureConfig(with_regime=False, with_hurst=False, with_garch=False, with_evt=False, with_vpin=True)
    fs = build_feature_set(m["hourly"], cfg, source=m["source"], n_jobs=1)
    return BacktestContext(fs, BacktestConfig(), m["fx"], m["rate"], m["events"])


SMALL_WF = WalkForwardConfig(train_months=18, val_months=6, oos_months=6, step_months=6, n_trials=6, top_k=2,
                             tune="all")


def test_walk_forward_end_to_end(small_ctx, tmp_path):
    runner = PurgedWalkForwardRunner(small_ctx, ModuleFlags(multi_stage_exits=True), SMALL_WF)
    wf = runner.run()
    n = len(wf.folds)
    assert n >= 4 and len(wf.fold_params) == n and len(wf.fold_table) == n and len(wf.oos_folds) == n
    # parameters come from the spec ranges
    for p in wf.fold_params:
        assert 16 <= p.donchian_window <= 40 and 1.2 <= p.atr_stop_mult <= 2.2 and 1 <= p.max_hold_days <= 5
    # stitched OOS is one continuous equity curve covering exactly the OOS span
    assert wf.oos.equity.index[0] == wf.folds[0].oos_start and wf.oos.equity.index[-1] == wf.folds[-1].oos_end
    assert wf.oos.equity.index.is_monotonic_increasing and wf.oos.equity.index.is_unique
    # every OOS trade was entered under the parameters of the fold that owns its date
    t = wf.oos.real_trades()
    starts = np.array([f.oos_start.to_datetime64() for f in wf.folds])
    owner = np.searchsorted(starts, pd.DatetimeIndex(t["entry_session"]).values, side="right") - 1
    assert (t["segment"].to_numpy() == owner).all()
    # no fold's OOS window overlaps the data its parameters were chosen on
    for f in wf.folds:
        assert f.val_end < f.oos_start
    # stressed costs cannot improve a fixed schedule's cost per trade
    assert wf.oos_stress.stats["cost_stress"] == 1.5
    assert len(wf.trial_sharpes_val) == SMALL_WF.n_trials * n and wf.n_evaluations > SMALL_WF.n_trials * n

    ts = runner.tearsheet(wf, n_configs=8, mc_config=__import__("ndxswing").analytics.montecarlo.MonteCarloConfig(n_paths=300))
    rep = ts.build()
    assert rep["synthetic"] and rep["verdict"].startswith("PIPELINE VERIFICATION ONLY")
    assert rep["dsr"] is not None and rep["pbo"] is not None and rep["folds"]["n_folds"] == n
    for seg in ("in_sample", "validation", "oos"):
        assert rep["segments"][seg] is not None
    paths = ts.save(tmp_path)
    assert "Walk-forward folds" in paths["markdown"].read_text()


def test_fixed_prior_walk_forward_has_no_selection(small_ctx):
    from dataclasses import replace
    from ndxswing.analytics.montecarlo import MonteCarloConfig

    runner = PurgedWalkForwardRunner(small_ctx, ModuleFlags(), replace(SMALL_WF, tune="none"))
    wf = runner.run()
    assert wf.n_evaluations == len(wf.folds)                                  # one run per fold, no search
    assert all(p == StrategyParams() for p in wf.fold_params)
    assert runner.config_trials(wf) == []
    rep = runner.tearsheet(wf, mc_config=MonteCarloConfig(n_paths=200)).build()
    gates = {g["gate"]: g["status"] for g in rep["gates"]}
    assert gates["PBO (CSCV)"] == "N/A (no selection)" and rep["pbo"] is None
    assert rep["dsr"]["no_selection"] and rep["dsr"]["sr0_per_period"] == 0.0
    assert rep["meta"]["tuned parameters"].startswith("none")
    # with other variants on the table the same run is judged against them
    other = small_ctx.backtester().run(StrategyParams(donchian_window=30), ModuleFlags(),
                                       start=wf.folds[0].oos_start, end=wf.folds[-1].oos_end)
    rep2 = runner.tearsheet(wf, extra_trials=[wf.oos, other], mc_config=MonteCarloConfig(n_paths=200)).build()
    assert rep2["dsr"]["n_trials"] == 2 and "no_selection" not in rep2["dsr"]


def test_validation_objective_never_sees_oos_data(small_ctx):
    """Corrupting every bar after a fold's validation window must not change
    the fold's objective value (selection is blind to the future)."""
    cfg = SMALL_WF
    folds = make_folds(small_ctx.sessions, cfg)
    f = folds[1]
    theta = {"donchian_window": 22, "atr_stop_mult": 1.6, "max_hold_days": 4}
    wfm._STATE.update(bt=small_ctx.backtester(), flags=ModuleFlags(), cfg=cfg, base=StrategyParams(), folds=folds)
    j_ref = wfm._evaluate(theta, f)

    from ndxswing.features.pipeline import FeatureSet
    fs = small_ctx.features
    hb = fs.hourly.copy()
    after = hb["session"] > f.val_end
    hb.loc[after, ["open", "high", "low", "close"]] *= 3.0
    daily = fs.daily.copy()
    daily.loc[daily.index > f.val_end, ["ema20", "ema50", "sigma_rv"]] *= 3.0
    ctx2 = BacktestContext(FeatureSet(hb, daily, fs.config, fs.source), small_ctx.config, small_ctx.fx,
                           small_ctx.rate, small_ctx.events)
    wfm._STATE.update(bt=ctx2.backtester())
    assert wfm._evaluate(theta, f) == j_ref


def test_trade_starved_configurations_are_penalised(small_ctx):
    cfg = WalkForwardConfig(min_val_trades=3)
    res = small_ctx.backtester().run(StrategyParams(), ModuleFlags(), start="2014-01-02", end="2014-01-10")
    comp = wfm.objective_components(res, cfg)
    assert comp["n_trades"] < 3 and comp["j"] <= -1.0


def test_sequential_ablation_small(small_ctx, tmp_path):
    cfg = AblationConfig(candidates=("multi_stage_exits", "regime_gate", "hurst_gate", "arma_z_filter", "vpin_filter"),
                         holdout_start="2016-07-01")
    res = SequentialAblationEngine(small_ctx, SMALL_WF, cfg).run()
    t = res.table.set_index("module")
    assert t.loc["baseline", "status"] == "BASELINE"
    assert t.loc["hurst_gate", "status"].startswith("SKIPPED")          # no Hurst features built: reported, not faked
    assert t.loc["arma_z_filter", "status"].startswith("BLOCKED") or t.loc["arma_z_filter", "status"].startswith("SKIPPED")
    regime_rows = res.table[res.table["module"] == "regime_gate"]
    assert set(regime_rows["variant"]) == {"rv20"}                      # only the deterministic benchmark has data
    for mod in ("multi_stage_exits", "vpin_filter"):
        row = t.loc[mod]
        assert row["status"] == "RETAINED" or row["status"].startswith("rejected")
        assert np.isfinite(row["d_sharpe"])
        retained = row["status"] == "RETAINED"
        assert getattr(res.final_flags, mod) == retained
        if retained:
            assert row["d_sharpe"] >= cfg.min_sharpe_lift and row["n_trades"] >= res.trade_floor
    assert res.n_variants == 4 == len(res.variant_oos) and res.holdout is not None and res.holdout["start"] == "2016-07-01"
    md = res.save(tmp_path).read_text()
    assert "Untouched holdout" in md and "| 0 | baseline |" in md
    assert set(cfg.candidates) <= set(MODULE_ORDER)


# ------------------------------------------------------------------------ store


def test_store_round_trip(tmp_path):
    m = generate_synthetic_market(SyntheticSpec(end="2012-03-30"))
    with MarketDataStore(tmp_path / "m.duckdb") as store:
        n = store.upsert_hourly("SYN", m["hourly"], SYNTHETIC_SOURCE)
        assert n == len(m["hourly"])
        store.upsert_hourly("SYN", m["hourly"].iloc[:50], SYNTHETIC_SOURCE)      # idempotent upsert
        back = store.load_hourly("SYN")
        assert len(back) == n and str(back.index.tz) == "UTC"
        assert np.allclose(back["close"].to_numpy(), m["hourly"]["close"].to_numpy())
        assert back.index.equals(m["hourly"].index)
        assert len(store.load_hourly("SYN", start="2012-02-01", end="2012-02-29")) == 20 * 7   # 21 weekdays less Presidents' Day
        store.upsert_series("EURUSD", m["fx"], "test")
        assert store.load_series("EURUSD").iloc[-1] == pytest.approx(m["fx"].iloc[-1])
        assert store.has_series("EURUSD") and not store.has_series("NOPE")
        store.upsert_events(m["events"], "test")
        ev = store.load_events()
        assert len(ev) == len(m["events"]) and str(ev["ts"].dt.tz) == "UTC"
        assert store.sources("SYN") == [SYNTHETIC_SOURCE]
        summary = store.summary()
        assert set(summary["tbl"]) == {"bars_1h", "daily_series", "events"}
        with pytest.raises(KeyError):
            store.load_hourly("MISSING")


def test_normalise_bars_timezones_and_quotes():
    raw = pd.DataFrame({"Date": ["2024-03-08 09:30", "2024-03-11 09:30"], "Open": [100, 101], "High": [101, 102],
                        "Low": [99, 100], "Close": [100.5, 101.5], "Volume": [10, 20],
                        "bid": [100.4, 101.4], "ask": [100.6, 101.7]})
    out = ing.normalise_bars(raw, tz="America/New_York", bid_col="bid", ask_col="ask")
    # 9 March 2024 is before the US DST switch, 11 March after it
    assert out.index[0] == pd.Timestamp("2024-03-08 14:30", tz="UTC")
    assert out.index[1] == pd.Timestamp("2024-03-11 13:30", tz="UTC")
    assert out["spread_rel"].iloc[0] == pytest.approx(0.2 / 100.5)
    with pytest.raises(ValueError):
        ing.normalise_bars(raw.drop(columns="Date").reset_index(drop=True), ts_col="when")
    rep = ing.data_quality_report(generate_synthetic_market(SyntheticSpec(end="2012-02-28"))["hourly"])
    assert rep["bars_per_session_mode"] == 7 and rep["bars_over_10pct_move"] == 0 and rep["has_spread"]


def test_fred_fetch_parses_and_refuses_without_key(monkeypatch):
    class Resp:
        def __init__(self, payload): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(self.payload).encode()

    seen = {}

    def opener(url):
        seen["url"] = url
        return Resp({"observations": [{"date": "2024-01-02", "value": "1.0956"}, {"date": "2024-01-03", "value": "."},
                                      {"date": "2024-01-04", "value": "1.0945"}]})

    s = ing.fetch_fred_series("DEXUSEU", api_key="k" * 32, opener=opener)
    assert list(s.round(4)) == [1.0956, 1.0945] and "series_id=DEXUSEU" in seen["url"]
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        ing.fetch_fred_series("DEXUSEU")
    with pytest.raises(RuntimeError, match="no observations"):
        ing.fetch_fred_series("X", api_key="k", opener=lambda url: Resp({"error_message": "bad"}))


def test_panama_adjustment_removes_roll_gap_and_keeps_raw_prices():
    days = pd.bdate_range("2024-01-02", "2024-06-28")
    ts = pd.DatetimeIndex([d + pd.Timedelta(hours=15) for d in days]).tz_localize("UTC")
    spot = 17000 + np.arange(len(days)) * 5.0                                    # a perfectly smooth underlying
    rows = []
    for name, expiry, basis in (("NQH4", "2024-03-15", 50.0), ("NQM4", "2024-06-21", 120.0), ("NQU4", "2024-09-20", 200.0)):
        for t, s in zip(ts, spot):
            if t.tz_localize(None) <= pd.Timestamp(expiry) + pd.Timedelta(days=1):
                rows.append({"contract": name, "expiry": expiry, "ts": t, "open": s + basis, "high": s + basis + 1,
                             "low": s + basis - 1, "close": s + basis, "volume": 100.0})
    cont = panama_adjust(pd.DataFrame(rows), roll_days_before_expiry=5)
    assert cont.index.is_unique and len(cont) == len(days)
    assert np.allclose(np.diff(cont["close"].to_numpy()), 5.0)                   # no artificial breakout at a roll
    assert list(cont["contract"].unique()) == ["NQH4", "NQM4", "NQU4"]
    last = cont[cont["contract"] == "NQU4"]
    assert (last["adjustment"] == 0).all() and np.allclose(last["close"], last["raw_close"])
    first = cont[cont["contract"] == "NQH4"]
    assert np.allclose(first["adjustment"], (120 - 50) + (200 - 120))
    assert np.allclose(first["raw_close"], first["close"] - 150.0)               # raw contract prices preserved
    # the unadjusted front-contract series does jump at the rolls
    assert np.abs(np.diff(cont["raw_close"].to_numpy())).max() > 50
    with pytest.raises(ValueError):
        panama_adjust(pd.DataFrame(rows).drop(columns="expiry"))


def test_synthetic_generator_is_labelled_and_well_formed():
    m = generate_synthetic_market(SyntheticSpec(end="2013-12-31", edge=0.0))
    hb = m["hourly"]
    assert m["source"] == SYNTHETIC_SOURCE == "SYNTHETIC-PIPELINE-VERIFICATION"
    assert (hb["high"] >= hb[["open", "close"]].max(axis=1)).all() and (hb["low"] <= hb[["open", "close"]].min(axis=1)).all()
    local = hb.index.tz_convert("America/New_York")
    assert set(zip(local.hour, local.minute)) == {(9, 30), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0)}
    r = np.log(hb["close"]).diff().dropna()
    assert abs(r.mean()) / (r.std() / np.sqrt(len(r))) < 3.0                     # null data: no drift
    same = generate_synthetic_market(SyntheticSpec(end="2013-12-31", edge=0.0))["hourly"]
    assert hb.equals(same)                                                       # reproducible from the seed
