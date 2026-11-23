"""Statistical layer: correctness against brute-force / Monte Carlo references
and strict causality (no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ndxswing.features import garch as g
from ndxswing.features import hurst as hu
from ndxswing.features.indicators import (
    atr, daily_from_hourly, donchian_prev, prepare_hourly_bars, regime_observations, trailing_pct_rank,
)
from ndxswing.features.pipeline import FeatureConfig, build_feature_set
from ndxswing.features.regime import CausalRegimeHMM, confidence, hamilton_filter
from ndxswing.features.vpin import compute_volume_time_vpin


def fgn(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Fractional Gaussian noise by spectral synthesis (test helper)."""
    f = np.fft.rfftfreq(n)[1:]
    spec = np.r_[0, f ** (-(2 * H - 1) / 2) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(f)))]
    x = np.fft.irfft(spec, n)
    return x / x.std()


# ------------------------------------------------------------------ indicators


def test_prepare_filters_to_regular_session_and_assigns_ny_session(synth_market):
    hb = synth_market["hourly"]
    extra = hb.iloc[:1].copy()
    extra.index = extra.index - pd.Timedelta(hours=3)          # 06:30 ET pre-market bar
    out = prepare_hourly_bars(pd.concat([extra, hb]))
    assert len(out) == len(hb)
    assert out.groupby("session").size().eq(7).all()
    with pytest.raises(ValueError, match="OHLC consistency"):
        bad = hb.copy()
        bad.iloc[10, bad.columns.get_loc("high")] = bad.iloc[10]["low"] - 1.0
        prepare_hourly_bars(bad)


def test_donchian_excludes_the_current_bar():
    high = pd.Series([1, 2, 3, 10, 4, 5], dtype=float)
    hi, _ = donchian_prev(high, high, 3)
    assert np.isnan(hi.iloc[2])
    assert hi.iloc[3] == 3.0            # the 10 printed on bar 3 is not part of bar 3's level
    assert hi.iloc[4] == 10.0


def test_atr_is_wilder_and_includes_gaps():
    h = pd.Series([10, 11, 20.0]); l = pd.Series([9, 10, 19.0]); c = pd.Series([9.5, 10.5, 19.5])
    a = atr(h, l, c, 2)
    # TR = [1, 1.5, 9.5]; Wilder alpha 1/2: 1 -> 1.25 -> 5.375
    assert a.iloc[2] == pytest.approx(5.375)


def test_trailing_pct_rank_uses_only_prior_values():
    x = pd.Series(np.arange(50, dtype=float))
    r = trailing_pct_rank(x, 20, min_periods=5)
    assert r.iloc[30] == 1.0
    y = x.copy(); y.iloc[40:] = -100.0                         # change the future
    assert np.allclose(trailing_pct_rank(y, 20, min_periods=5).iloc[:40], r.iloc[:40], equal_nan=True)


# ----------------------------------------------------------------------- Hurst


def test_dfa_matches_brute_force_on_integrated_profile():
    rng = np.random.default_rng(3)
    x = fgn(1024, 0.65, rng)
    sc = hu.dfa_scales(len(x))
    Y = np.cumsum(x - x.mean())
    F = []
    for s in sc:
        n = len(x) // s
        segs = list(Y[: n * s].reshape(n, s)) + list(Y[len(x) - n * s:].reshape(n, s))
        t = np.arange(s)
        F.append(np.sqrt(np.mean([np.mean((seg - np.polyval(np.polyfit(t, seg, 1), t)) ** 2) for seg in segs])))
    ref = np.polyfit(np.log(sc), np.log(F), 1)[0]
    assert hu.dfa_hurst(x)[0] == pytest.approx(ref, abs=1e-9)


def test_dfa_recovers_known_exponents():
    rng = np.random.default_rng(11)
    white = hu.dfa_hurst(rng.normal(size=(40, 1500)))
    assert abs(white.mean() - 0.5) < 0.03
    pers = np.array([hu.dfa_hurst(fgn(1500, 0.70, rng))[0] for _ in range(20)])
    assert abs(pers.mean() - 0.70) < 0.04
    anti = np.array([hu.dfa_hurst(fgn(1500, 0.35, rng))[0] for _ in range(20)])
    assert anti.mean() < 0.45


def test_dfa_on_raw_unintegrated_returns_would_be_wrong():
    """Guard for the audit fix: detrending the raw series (no integration)
    gives an exponent near 0, not H."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=1500)
    sc = hu.dfa_scales(1500)
    wrong = []
    for s in sc:
        n = 1500 // s
        seg = x[: n * s].reshape(n, s)
        t = np.arange(s)
        wrong.append(np.sqrt(np.mean([(row - np.polyval(np.polyfit(t, row, 1), t)) ** 2 for row in seg])))
    slope_wrong = np.polyfit(np.log(sc), np.log(wrong), 1)[0]
    assert slope_wrong < 0.15 and abs(hu.dfa_hurst(x)[0] - 0.5) < 0.12


def test_bootstrap_lcb_and_lo_statistic():
    rng = np.random.default_rng(7)
    idx = hu.stationary_bootstrap_indices(500, 50, 24.0, rng)
    assert idx.shape == (50, 500) and idx.min() >= 0 and idx.max() < 500
    runs = (np.diff(idx, axis=1) % 500 != 1).mean()
    assert 1 / 40 < runs < 1 / 15                              # mean block length about 24
    h, lcb = hu.hurst_with_lcb(fgn(1500, 0.75, rng), n_boot=200, rng=rng)
    assert lcb < h and lcb > 0.5
    h0, lcb0 = hu.hurst_with_lcb(rng.normal(size=1500), n_boot=200, rng=rng)
    assert lcb0 < 0.5                                          # noise must not pass the persistence gate
    v_noise, _ = hu.lo_modified_rs(rng.normal(size=2000))
    v_pers, _ = hu.lo_modified_rs(fgn(2000, 0.8, rng))
    assert 0.809 < v_noise < 1.862 and v_pers > hu.LO_RS_UPPER_95


# ----------------------------------------------------------------------- GARCH

PARAMS = g.ArmaGjrParams(c=0.0004, phi=0.15, theta=-0.05, omega=2e-6, alpha=0.04, gamma=0.10, beta=0.88, nu=6.0)


def test_exact_multi_horizon_forecast_matches_monte_carlo():
    r0, e0, s0 = -0.012, -0.011, 1.5e-4
    fc = g.forecast_arma_gjr_garch_exact(PARAMS, r0, e0, s0)
    rng = np.random.default_rng(1)
    n = 600_000
    rp, ep, sp, cum = np.full(n, r0), np.full(n, e0), np.full(n, s0), np.zeros(n)
    sc = np.sqrt((PARAMS.nu - 2) / PARAMS.nu)
    for h in range(1, 6):
        s2 = PARAMS.omega + (PARAMS.alpha + PARAMS.gamma * (ep < 0)) * ep**2 + PARAMS.beta * sp
        e = np.sqrt(s2) * rng.standard_t(PARAMS.nu, n) * sc
        r = PARAMS.c + PARAMS.phi * rp + PARAMS.theta * ep + e
        cum += r
        rp, ep, sp = r, e, s2
        if h in g.HORIZONS:
            assert fc[f"sigma_{h}"] == pytest.approx(cum.std(), rel=0.01)
            assert fc[f"mu_{h}"] == pytest.approx(cum.mean(), abs=4 * cum.std() / np.sqrt(n))
    assert fc["sigma_1d"] == fc["sigma_1"]


def test_wold_weights_for_ar1_and_naive_sum_is_wrong():
    p = g.ArmaGjrParams(c=0.0, phi=0.3, theta=0.0, omega=1e-6, alpha=0.05, gamma=0.0, beta=0.90, nu=8.0)
    fc = g.forecast_arma_gjr_garch_exact(p, 0.0, 0.0, 1e-4, (3,))
    s2 = [p.omega + p.beta * 1e-4]
    for _ in range(2):
        s2.append(p.omega + p.persistence * s2[-1])
    Psi = [(1 - p.phi ** (m + 1)) / (1 - p.phi) for m in range(3)]         # closed form for AR(1)
    exact = np.sqrt(sum(Psi[3 - h] ** 2 * s2[h - 1] for h in (1, 2, 3)))
    assert fc["sigma_3"] == pytest.approx(exact, rel=1e-12)
    assert fc["sigma_3"] > np.sqrt(sum(s2)) * 1.1                           # unweighted sum understates risk


def test_fit_recovers_asymmetry_and_filter_is_causal():
    rng = np.random.default_rng(4)
    n = 4000
    p = g.ArmaGjrParams(c=0.0003, phi=0.05, theta=0.0, omega=2e-6, alpha=0.03, gamma=0.12, beta=0.88, nu=7.0)
    r, e, s2 = np.zeros(n), np.zeros(n), np.full(n, 1e-4)
    sc = np.sqrt((p.nu - 2) / p.nu)
    for t in range(1, n):
        s2[t] = p.omega + (p.alpha + p.gamma * (e[t - 1] < 0)) * e[t - 1] ** 2 + p.beta * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * rng.standard_t(p.nu) * sc
        r[t] = p.c + p.phi * r[t - 1] + e[t]
    fit = g.fit_arma_gjr_garch_t(r)
    assert fit.gamma > 0.05 and 0.80 < fit.beta < 0.95 and fit.persistence < 1
    eps_a, s2_a = g.filter_arma_gjr(r[:3000], fit)
    r2 = r.copy(); r2[3000:] = 0.5
    eps_b, s2_b = g.filter_arma_gjr(r2[:3000], fit)
    assert np.array_equal(s2_a, s2_b)
    z = eps_a / np.sqrt(s2_a)
    assert g.ljung_box_pvalue(z[50:], 10, 1) > 0.01
    assert g.ljung_box_pvalue(r[1:] * np.r_[1, np.sign(r[1:-1])], 10) is not None


def test_ljung_box_matches_statsmodels_and_detects_autocorrelation():
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rng = np.random.default_rng(2)
    x = rng.normal(size=800)
    ref = float(acorr_ljungbox(x, lags=[10])["lb_pvalue"].iloc[0])
    assert g.ljung_box_pvalue(x, 10) == pytest.approx(ref, rel=1e-6)
    ar = np.zeros(800)
    for t in range(1, 800):
        ar[t] = 0.4 * ar[t - 1] + x[t]
    assert g.ljung_box_pvalue(ar, 10) < 1e-6


def test_pesaran_timmermann():
    rng = np.random.default_rng(9)
    a = rng.normal(size=600)
    _, p_noise = g.pesaran_timmermann(a, rng.normal(size=600))
    _, p_skill = g.pesaran_timmermann(a, a + rng.normal(size=600))
    assert p_noise > 0.01 and p_skill < 1e-6


# ---------------------------------------------------------------------- regime


def test_hamilton_filter_matches_brute_force_and_is_causal():
    rng = np.random.default_rng(0)
    T, K = 60, 3
    logf = rng.normal(size=(T, K))
    A = np.array([[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0.05, 0.15, 0.8]])
    xi0 = np.array([0.5, 0.3, 0.2])
    xi = hamilton_filter(logf, A, xi0)
    alpha = xi0.copy()
    for t in range(T):                                   # textbook forward algorithm
        alpha = (A.T @ alpha) * np.exp(logf[t])
        alpha /= alpha.sum()
        assert np.allclose(alpha, xi[t], atol=1e-12)
    logf2 = logf.copy(); logf2[40:] += rng.normal(size=(20, K)) * 5
    assert np.array_equal(hamilton_filter(logf2, A, xi0)[:40], xi[:40])
    assert confidence(np.array([[1.0, 0.0, 0.0]]))[0] == pytest.approx(1.0)
    assert confidence(np.full((1, 3), 1 / 3))[0] == pytest.approx(0.0, abs=1e-12)


def _regime_obs(n: int = 700, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < 0.97 else rng.integers(0, 3)
    vol = np.array([0.006, 0.008, 0.03])[state]
    drift = np.array([0.002, 0.0, -0.002])[state]
    close = 100 * np.exp(np.cumsum(drift + vol * rng.normal(size=n)))
    d = pd.DataFrame({"open": close * np.exp(rng.normal(0, 0.002, n)), "close": close,
                      "volume": 1e6 * np.exp(rng.normal(0, 0.2, n))},
                     index=pd.bdate_range("2015-01-01", periods=n))
    d["high"] = d[["open", "close"]].max(axis=1) * (1 + vol / 2)
    d["low"] = d[["open", "close"]].min(axis=1) * (1 - vol / 2)
    obs = regime_observations(d)
    obs["true_state"] = state
    return obs


def test_regime_alignment_and_rolling_filter_causality():
    obs = _regime_obs()
    hmm = CausalRegimeHMM(3, window=300, refit_every=50, n_init=2, n_iter=40)
    out = hmm.rolling_filter(obs)
    ok = out["p_trend"].notna()
    assert ok.sum() > 300
    probs = out.loc[ok, ["p_trend", "p_side", "p_stress"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0)
    stress_days = obs.loc[ok, "true_state"].to_numpy() == 2
    # stress = state with the highest RV20 (RV20 itself lags a regime switch by up to 20 days)
    assert probs[stress_days, 2].mean() > probs[~stress_days, 2].mean() + 0.3
    # causality: recomputing on a truncated history reproduces every earlier row
    cut = 520
    out_cut = hmm.rolling_filter(obs.iloc[:cut])
    a = out.iloc[:cut][["p_trend", "p_side", "p_stress", "conf"]].to_numpy()
    b = out_cut[["p_trend", "p_side", "p_stress", "conf"]].to_numpy()
    assert np.allclose(a, b, equal_nan=True, atol=1e-10)


def test_two_state_benchmark_orders_stress_last():
    obs = _regime_obs(seed=3)
    out = CausalRegimeHMM(2, window=300, refit_every=100, n_init=2, n_iter=40,
                          features=("r", "rv20", "vc", "gap")).rolling_filter(obs)
    ok = out["p_stress"].notna()
    stress_days = obs.loc[ok, "true_state"].to_numpy() == 2
    assert out.loc[ok, "p_stress"][stress_days].mean() > out.loc[ok, "p_stress"][~stress_days].mean() + 0.3


# ------------------------------------------------------------------------ VPIN


def test_vpin_is_bounded_and_strictly_causal(synth_features):
    hb = synth_features.hourly
    v = compute_volume_time_vpin(hb)
    assert v.notna().sum() > 1000 and v.dropna().between(0, 1).all()
    cut = len(hb) - 700
    v_cut = compute_volume_time_vpin(hb.iloc[:cut])
    assert np.allclose(v.iloc[:cut].to_numpy(), v_cut.to_numpy(), equal_nan=True)
    assert v.iloc[: 20 * 7].isna().all()                       # needs 20 prior sessions for the bucket size


# -------------------------------------------------------- whole feature table


def test_feature_table_has_no_look_ahead(synth_market):
    """Every daily feature dated <= cut is identical whether or not data after
    the cut exists (rolling refits included)."""
    cfg = FeatureConfig(hmm_window=150, hmm_refit_every=30, hmm_n_init=2, hurst_length=400, hurst_boot=20,
                        garch_window=250, garch_refit_every=20, evt_window=150, evt_refit_every=10)
    hb = synth_market["hourly"]
    hb = hb[hb.index < "2013-12-01"]
    full = build_feature_set(hb, cfg, n_jobs=4)
    cut_ts = pd.Timestamp("2013-09-16", tz="UTC")
    part = build_feature_set(hb[hb.index < cut_ts], cfg, n_jobs=4)
    idx = part.daily.index
    num = full.daily.select_dtypes("number").columns
    a, b = full.daily.loc[idx, num].to_numpy(float), part.daily[num].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True, rtol=1e-8, atol=1e-10)
    assert np.isfinite(b[-1]).mean() > 0.9                     # the comparison is not vacuous
    hcols = ["vpin", "vpin_pct", "spread_med"]
    assert np.allclose(full.hourly.loc[part.hourly.index, hcols].to_numpy(float),
                       part.hourly[hcols].to_numpy(float), equal_nan=True)


def test_daily_bars_from_hourly(synth_market):
    hb = prepare_hourly_bars(synth_market["hourly"])
    d = daily_from_hourly(hb)
    first = hb[hb["session"] == d.index[5]]
    assert d["open"].iloc[5] == first["open"].iloc[0] and d["close"].iloc[5] == first["close"].iloc[-1]
    assert d["high"].iloc[5] == first["high"].max() and d["volume"].iloc[5] == pytest.approx(first["volume"].sum())
