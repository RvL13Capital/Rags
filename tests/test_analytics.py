"""Metrics, Deflated Sharpe, CSCV-PBO, Monte Carlo and the tearsheet."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from ndxswing.analytics import montecarlo as mc
from ndxswing.analytics import stats as st
from ndxswing.analytics.tearsheet import SEGMENTS, TearsheetReporter
from ndxswing.backtest.engine import TRADE_COLUMNS, BacktestResult


def result_from_returns(r: np.ndarray, trades: pd.DataFrame | None = None, source: str = "unit-test") -> BacktestResult:
    idx = pd.bdate_range("2020-01-01", periods=len(r))
    eq = pd.Series(5000.0 * np.cumprod(1 + r), index=idx)
    if trades is None:
        trades = pd.DataFrame(columns=TRADE_COLUMNS + ["entry_session"])
    return BacktestResult(eq, trades, 5000.0, {"source": source})


def make_trades(pnl: list[float]) -> pd.DataFrame:
    n = len(pnl)
    d = {c: [0.0] * n for c in TRADE_COLUMNS}
    d.update(net_pnl=pnl, r_multiple=[p / 10.0 for p in pnl], shadow=[False] * n, turnover=[1000.0] * n,
             commission=[0.1] * n, spread_cost=[0.2] * n, slippage_cost=[0.3] * n, financing=[0.05] * n,
             entry_session=list(pd.bdate_range("2020-01-01", periods=n)))
    return pd.DataFrame(d)


# --------------------------------------------------------------------- metrics


def test_core_metrics_by_hand():
    r = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02] * 42)            # 252 days
    res = result_from_returns(r, make_trades([30.0, -10.0, 20.0, -20.0]))
    m = st.compute_metrics(res)
    growth = np.prod(1 + r)
    assert m["cagr"] == pytest.approx(growth - 1)
    assert m["ann_vol"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert m["sortino"] == pytest.approx(r.mean() / np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252))
    w = np.r_[1.0, np.cumprod(1 + r)]
    assert m["max_drawdown"] == pytest.approx(np.max(1 - w / np.maximum.accumulate(w)))
    assert m["calmar"] == pytest.approx(m["cagr"] / m["max_drawdown"])
    assert m["profit_factor"] == pytest.approx(50 / 30)
    assert m["win_rate"] == 0.5 and m["avg_win_over_avg_loss"] == pytest.approx(25 / 15)
    assert m["expectancy_eur"] == pytest.approx(5.0) and m["expectancy_r"] == pytest.approx(0.5)
    assert m["n_trades"] == 4
    assert m["transaction_costs"] == pytest.approx(4 * 0.6) and m["financing_costs"] == pytest.approx(0.2)
    assert m["turnover_ann"] == pytest.approx(4000 / res.equity.mean() / 1.0)
    assert m["es_99"] >= m["es_975"] > 0


def test_first_day_return_is_measured_from_initial_equity():
    res = result_from_returns(np.array([-0.05, 0.0, 0.0]))
    assert res.daily_returns().iloc[0] == pytest.approx(-0.05)
    assert st.max_drawdown(res.equity.to_numpy(), 5000.0) == pytest.approx(0.05)


# ------------------------------------------------------------------------- DSR


def test_expected_max_sharpe_includes_cross_trial_mean():
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 0.05, 200)
    base = st.expected_max_sharpe(s)
    z = (1 - st.EULER_GAMMA) * norm.ppf(1 - 1 / 200) + st.EULER_GAMMA * norm.ppf(1 - 1 / (200 * np.e))
    assert base == pytest.approx(s.mean() + s.std(ddof=1) * z)
    assert st.expected_max_sharpe(s + 0.03) == pytest.approx(base + 0.03)     # audit fix: mean is not dropped
    assert st.expected_max_sharpe(s, n_trials=2000) > base                     # more trials -> higher hurdle


def test_psr_uses_skew_and_raw_kurtosis():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 2000)
    sr = st.sharpe_per_period(r)
    assert st.probabilistic_sharpe_ratio(r, sr) == pytest.approx(0.5)
    fat = 0.001 + 0.01 * rng.standard_t(4, 2000) / np.sqrt(2)
    from scipy.stats import kurtosis, skew
    g3, g4 = skew(fat), kurtosis(fat, fisher=False)
    srf = st.sharpe_per_period(fat)
    expect = norm.cdf(srf * np.sqrt(1999) / np.sqrt(1 - g3 * srf + (g4 - 1) / 4 * srf**2))
    assert st.probabilistic_sharpe_ratio(fat, 0.0) == pytest.approx(expect)
    assert g4 > 3.5                                                            # raw (non-excess) kurtosis


def test_dsr_separates_luck_from_skill():
    rng = np.random.default_rng(2)
    T, N = 1500, 100
    noise = rng.normal(0, 0.01, (T, N))
    trial_sr = np.array([st.sharpe_per_period(noise[:, j]) for j in range(N)])
    lucky = noise[:, trial_sr.argmax()]
    assert st.deflated_sharpe_ratio(lucky, trial_sr)["dsr"] < 0.9              # best of 100 random walks
    skilled = rng.normal(0.0015, 0.01, T)
    assert st.deflated_sharpe_ratio(skilled, trial_sr)["dsr"] > 0.95


# ------------------------------------------------------------------------- PBO


def test_pbo_noise_vs_genuine_edge_and_ties():
    rng = np.random.default_rng(3)
    T, N = 1600, 20
    noise = rng.normal(0, 0.01, (T, N))
    assert st.cscv_pbo(noise, 16)["n_combinations"] == 12870
    # the splits of one data set are highly dependent, so a single PBO on noise is itself noisy;
    # across independent noise matrices it centres on 1/2 (the IS winner's OOS rank is uniform)
    many = [st.cscv_pbo(np.random.default_rng(100 + s).normal(0, 0.01, (800, N)), 8)["pbo"] for s in range(40)]
    assert 0.38 < np.mean(many) < 0.62
    edge = noise.copy(); edge[:, 7] += 0.002                                   # one configuration truly better
    assert st.cscv_pbo(edge, 16)["pbo"] < 0.05
    tied = np.zeros((T, 6)); tied[:, 0] = noise[:, 0]; tied[:, 1] = noise[:, 0]   # duplicates and no-trade configs
    out = st.cscv_pbo(tied, 8)
    assert 0.0 <= out["pbo"] <= 1.0 and np.isfinite(out["logit_median"])
    with pytest.raises(ValueError):
        st.cscv_pbo(noise, 7)


def test_pbo_overfit_selection_is_flagged():
    """Configurations whose in-sample winners mean-revert out of sample."""
    rng = np.random.default_rng(4)
    T, N, S = 1600, 30, 16
    b = T // S
    M = rng.normal(0, 0.01, (T, N))
    for j in range(N):                                                         # alternating block drift, zero overall
        sign = rng.choice([-1, 1], S // 2)
        drift = np.repeat(np.r_[sign, -sign][rng.permutation(S)], b) * 0.002
        M[:, j] += drift
    assert st.cscv_pbo(M, S)["pbo"] > 0.5


# ----------------------------------------------------------------- Monte Carlo


def test_wealth_never_flips_sign_and_recovery_is_longest_streak():
    sim = np.array([[0.10, -1.50, 0.20, 0.20],                                 # a -150% print must not flip wealth
                    [-0.10, 0.20, -0.05, -0.05]])
    s = mc.path_statistics(sim)
    assert s["median_terminal_wealth"] > 0
    w = np.exp(np.cumsum(np.log1p(np.maximum(sim[0], -0.99))))
    assert w.min() > 0 and w[1] == pytest.approx(1.1 * 0.01)
    # underwater: 3 days, recovered, 2 days -> longest contiguous streak is 3, not 5
    mask = np.array([[False, True, True, True, False, True, True, False]])
    assert mc.longest_true_run(mask)[0] == 3
    r = np.array([[-0.05, 0.01, 0.01, 0.10, -0.02, 0.01, 0.05]])
    wealth = np.r_[1, np.cumprod(1 + r[0])]
    underwater = wealth < np.maximum.accumulate(wealth)
    assert underwater.sum() == 5 and mc.path_statistics(r)["median_recovery_days"] == 3


def test_block_bootstrap_draws_contiguous_blocks_from_the_sample():
    r = np.arange(500, dtype=float)                                            # value = position
    cfg = mc.MonteCarloConfig(n_paths=300, horizon=120)
    sim = mc.simulate_block_bootstrap(r, cfg, np.random.default_rng(0))
    assert sim.shape == (300, 120) and np.isin(sim, r).all()
    steps = np.diff(sim, axis=1)
    cont = (steps == 1) | (steps == -499)
    breaks = (~cont).sum(axis=1)
    assert 120 / 10 - 1 <= breaks.mean() <= 120 / 5                            # blocks of 5..10 days
    run_len = []
    for row in cont[:50]:
        k = 1
        for c in row:
            if c:
                k += 1
            else:
                run_len.append(k); k = 1
    # (two blocks can abut by chance with probability 1/500, hence not a strict upper bound)
    assert min(run_len) >= 5 and np.mean(np.array(run_len) <= 10) > 0.98


def test_tempered_levy_has_fat_tails_but_finite_variance():
    rng = np.random.default_rng(1)
    x = mc.tempered_stable(1.7, 0.10, 400_000, rng)
    pure = mc.tempered_stable(1.7, 0.0, 400_000, rng)
    assert np.isfinite(x.std()) and abs(x).max() < abs(pure).max()
    k = np.mean((x / x.std()) ** 4)
    assert k > 4.0                                                             # heavier than Gaussian
    with pytest.raises(ValueError):
        mc.MonteCarloConfig(levy_alpha=1.2)


def test_run_monte_carlo_all_generators_and_tail_ordering():
    rng = np.random.default_rng(5)
    r = np.where(rng.random(1500) < 0.4, rng.standard_t(4, 1500) * 0.004 + 0.0004, 0.0)
    out = mc.run_monte_carlo(r, mc.MonteCarloConfig(n_paths=4000, horizon=252))
    assert set(out) == {"block", "student_t", "levy"}
    for v in out.values():
        assert 0 <= v["p_dd_30"] <= v["p_dd_20"] <= v["p_dd_10"] <= 1
        assert v["median_terminal_wealth"] > 0 and v["worst_99_dd"] >= 0 and v["n_paths"] == 4000
    assert out["student_t"]["fit_tsallis_q"] == pytest.approx(1 + 2 / (out["student_t"]["fit_nu"] + 1))
    assert out["block"]["median_terminal_wealth"] == pytest.approx(np.exp(252 * np.log1p(r).mean()), rel=0.05)
    short = mc.run_monte_carlo(r[:20])
    assert "error" in short
    idle = mc.run_monte_carlo(np.r_[np.zeros(300), 0.01, -0.01], mc.MonteCarloConfig(n_paths=200))
    assert "error" in idle["student_t"] and "p_dd_10" in idle["block"]         # refuses to fit 2 active days


# ------------------------------------------------------------------- tearsheet


def _ts(source: str, sharpe_boost: float = 0.0, **kw) -> TearsheetReporter:
    rng = np.random.default_rng(6)
    mk = lambda mu: result_from_returns(rng.normal(mu, 0.004, 900),
                                        make_trades(list(rng.normal(3, 20, 260))), source)
    seg = {"in_sample": mk(0.0006), "validation": mk(0.0005), "oos": mk(0.0 + sharpe_boost)}
    folds = pd.DataFrame({"fold": [0, 1, 2, 3], "oos_start": pd.bdate_range("2020-01-01", periods=4),
                          "oos_end": pd.bdate_range("2020-06-01", periods=4), "val_j": 1.0, "val_sharpe": 1.0,
                          "oos_sharpe": 0.5, "oos_net_pnl": [10.0, -5.0, 8.0, 3.0], "oos_trades": 10})
    return TearsheetReporter(seg, oos_stress=mk(0.0001 + sharpe_boost), trial_sharpes=rng.normal(0, 0.03, 64),
                             n_trials=640, pbo={"pbo": 0.2, "n_combinations": 12870, "n_configs": 64,
                                                "is_oos_slope": 0.1, "prob_oos_loss": 0.3},
                             fold_table=folds, mc_config=mc.MonteCarloConfig(n_paths=500), **kw)


def test_tearsheet_segments_are_separate_and_paper_live_are_empty(tmp_path):
    ts = _ts("unit-test")
    rep = ts.build()
    assert list(rep["segments"]) == list(SEGMENTS)
    assert rep["segments"]["paper"] is None and rep["segments"]["live"] is None
    assert rep["segments"]["in_sample"]["sharpe"] != rep["segments"]["oos"]["sharpe"]
    md = ts.to_markdown(rep)
    header = [l for l in md.splitlines() if l.startswith("| Metric")][0]
    assert header == "| Metric | In-Sample | Validation | Out-of-Sample | Paper | Live |"
    for n in range(1, 16):
        assert any(l.startswith(f"| {n}") for l in md.splitlines()), f"metric {n} missing"
    row = [l for l in md.splitlines() if l.startswith("| 3. Sharpe")][0].split("|")
    assert row[-3].strip() == "n/a" and row[-2].strip() == "n/a"
    paths = ts.save(tmp_path)
    saved = json.loads(paths["json"].read_text())
    assert saved["segments"]["oos"]["monte_carlo"]["block"]["n_paths"] == 500
    assert (tmp_path / "equity_oos.csv").exists() and (tmp_path / "metrics_by_segment.csv").exists()
    with pytest.raises(ValueError):
        TearsheetReporter({"backtest": None})


def test_gates_and_verdicts():
    weak = _ts("IBKR TRADES").build()
    gates = {g["gate"]: g for g in weak["gates"]}
    assert len(gates) == 9
    assert gates["OOS net Sharpe"]["status"] == "FAIL"
    assert gates["PBO (CSCV)"]["status"] == "PASS" and gates["PBO (CSCV)"]["value"] == 0.2
    assert gates["Profitable walk-forward OOS folds"]["value"] == 0.75
    assert weak["verdict"].startswith("NOT READY FOR LIVE DEPLOYMENT")
    strong = _ts("IBKR TRADES", sharpe_boost=0.0014).build()
    assert all(g["status"] == "PASS" for g in strong["gates"]), strong["gates"]
    assert "eligible for Phase 3 paper trading" in strong["verdict"] and "NOT READY FOR LIVE" in strong["verdict"]


def test_synthetic_data_can_never_produce_a_release_verdict():
    from ndxswing.data.synthetic import SYNTHETIC_SOURCE

    ts = _ts(SYNTHETIC_SOURCE, sharpe_boost=0.0014)
    rep = ts.build()
    assert all(g["status"] == "PASS" for g in rep["gates"])
    assert rep["verdict"].startswith("PIPELINE VERIFICATION ONLY")
    assert "PIPELINE VERIFICATION ON SYNTHETIC DATA" in ts.to_markdown(rep)


def test_missing_inputs_leave_gates_unevaluated():
    rng = np.random.default_rng(7)
    ts = TearsheetReporter({"oos": result_from_returns(rng.normal(0.001, 0.004, 400))},
                           mc_config=mc.MonteCarloConfig(n_paths=200))
    gates = {g["gate"]: g["status"] for g in ts.build()["gates"]}
    assert gates["Deflated Sharpe Ratio"] == "NOT EVALUATED" and gates["PBO (CSCV)"] == "NOT EVALUATED"
    assert ts.build()["verdict"].startswith("NOT READY")
