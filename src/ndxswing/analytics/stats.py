"""Performance metrics and overfitting corrections (sections 8.3 / 8.4)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, rankdata, skew

from ..backtest.engine import BacktestResult
from ..risk.tail import evt_var_es

EULER_GAMMA = 0.5772156649015329
ANN = 252.0


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #


def max_drawdown(equity: np.ndarray, initial: float) -> float:
    w = np.r_[initial, np.asarray(equity, dtype=float)]
    peak = np.maximum.accumulate(w)
    return float(np.max(1.0 - w / peak))


def sharpe_per_period(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 1e-12 else 0.0


def compute_metrics(result: BacktestResult) -> dict[str, float]:
    """The required performance metrics for one result (one segment).

    Sharpe/Sortino use a zero risk-free rate (idle cash earns nothing in the
    simulation).  Trade statistics use real (non-shadow) trades only.
    """
    r = result.daily_returns().to_numpy()
    n = len(r)
    years = n / ANN if n else np.nan
    eq = result.equity.to_numpy()
    final = float(eq[-1]) if n else result.initial_equity
    growth = final / result.initial_equity
    cagr = growth ** (1.0 / years) - 1.0 if n and growth > 0 else (-1.0 if n else np.nan)
    vol = float(r.std(ddof=1) * np.sqrt(ANN)) if n > 1 else np.nan
    sharpe = sharpe_per_period(r) * np.sqrt(ANN) if n > 1 else np.nan
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if n else np.nan
    sortino = float(r.mean() / downside * np.sqrt(ANN)) if n and downside > 1e-12 else (0.0 if n else np.nan)
    mdd = max_drawdown(eq, result.initial_equity) if n else np.nan
    calmar = float(cagr / mdd) if n and mdd > 1e-9 else (0.0 if n else np.nan)

    t = result.real_trades()
    nt = len(t)
    pnl = t["net_pnl"].to_numpy(dtype=float) if nt else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_w, gross_l = wins.sum(), -losses.sum()
    pf = float(gross_w / gross_l) if gross_l > 0 else (np.inf if gross_w > 0 else np.nan)
    avg_w = float(wins.mean()) if len(wins) else np.nan
    avg_l = float(-losses.mean()) if len(losses) else np.nan
    cost_cols = [c for c in ("commission", "spread_cost", "slippage_cost", "fx_fee") if c in t.columns]
    txn = float(t[cost_cols].to_numpy(dtype=float).sum()) if nt else 0.0
    fin = float(t["financing"].sum()) if nt else 0.0
    turnover = float(t["turnover"].sum()) if nt else 0.0
    mean_eq = float(np.mean(eq)) if n else np.nan

    loss = -r
    _, es975, m975 = evt_var_es(loss, 0.975) if n >= 50 else (np.nan, np.nan, "n/a")
    _, es99, m99 = evt_var_es(loss, 0.99) if n >= 50 else (np.nan, np.nan, "n/a")
    return {
        "cagr": float(cagr), "ann_vol": vol, "sharpe": float(sharpe), "sortino": sortino, "calmar": calmar,
        "max_drawdown": mdd, "profit_factor": pf, "win_rate": float(len(wins) / nt) if nt else np.nan,
        "avg_win_over_avg_loss": float(avg_w / avg_l) if nt and avg_l and avg_l > 0 else np.nan,
        "expectancy_eur": float(pnl.mean()) if nt else np.nan,
        "expectancy_r": float(t["r_multiple"].mean()) if nt else np.nan,
        "n_trades": nt, "turnover_ann": float(turnover / mean_eq / years) if n and mean_eq > 0 else np.nan,
        "transaction_costs": txn, "financing_costs": fin,
        "es_975": float(es975), "es_99": float(es99), "es_method": m99,
        "net_pnl": final - result.initial_equity, "n_days": n, "years": float(years),
        "final_equity": final,
    }


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #


def expected_max_sharpe(trial_sharpes: np.ndarray, n_trials: int | None = None) -> float:
    """Expected maximum Sharpe under the null across N trials.

    SR_0 = E[SR_k] + sqrt(V[SR_k]) * ((1-g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))

    The cross-trial mean is included (audit fix); all Sharpe ratios are per
    period (not annualised).
    """
    s = np.asarray(trial_sharpes, dtype=float)
    s = s[np.isfinite(s)]
    n = int(n_trials) if n_trials is not None else len(s)
    if len(s) < 2 or n < 2:
        return float(s.mean()) if len(s) else 0.0
    z = (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n) + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(s.mean() + s.std(ddof=1) * z)


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float) -> float:
    """PSR(SR*) with skewness and RAW kurtosis of the return series."""
    r = np.asarray(returns, dtype=float)
    T = len(r)
    if T < 3 or r.std(ddof=1) <= 1e-12:
        return np.nan
    sr = sharpe_per_period(r)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return np.nan
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1.0) / np.sqrt(denom)))


def average_pairwise_correlation(returns_matrix: np.ndarray) -> float:
    """Mean off-diagonal correlation of the columns of a (T, K) return matrix.
    A configuration that never traded has no defined correlation and is counted
    as uncorrelated (0), which errs towards MORE effective trials."""
    M = np.asarray(returns_matrix, dtype=float)
    k = M.shape[1]
    if k < 2:
        return 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(M, rowvar=False)
    off = c[~np.eye(k, dtype=bool)]
    return float(np.clip(np.nan_to_num(off, nan=0.0).mean(), 0.0, 1.0))


def effective_trials(k: int, rho_bar: float) -> float:
    """N_eff = max(1, rho_bar + (1 - rho_bar) K): K perfectly correlated
    variants are one trial, K independent ones are K trials."""
    return float(max(1.0, rho_bar + (1.0 - rho_bar) * k))


def expected_max_sharpe_zero_null(trial_sharpes: np.ndarray, n_eff: float) -> float:
    """Bailey & Lopez de Prado (2014): expected maximum Sharpe of N_eff trials
    whose true Sharpe ratio is zero, SR_0 = sqrt(V[SR_k]) * z(N_eff).  With a
    single effective trial nothing was selected and SR_0 = 0."""
    s = np.asarray(trial_sharpes, dtype=float)
    s = s[np.isfinite(s)]
    if n_eff <= 1.0 or len(s) < 2:
        return 0.0
    z = (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_eff) + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_eff * np.e))
    return float(max(0.0, s.std(ddof=1) * z))


def _expected_max_z(n: float) -> float:
    """E[max of n standard normals], Bailey & Lopez de Prado's approximation."""
    return float((1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n) + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * np.e)))


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes: np.ndarray | None, n_trials: int | None = None,
                          rho_bar: float = 0.0, prior_eff_trials: float = 0.0) -> dict:
    """Deflated Sharpe Ratio of ``returns`` given the K configurations compared.

    dsr                   release gate: zero-mean null, correlation-adjusted
                          effective trials N_eff = max(1, rho + (1 - rho) K)
    dsr_cross_trial_mean  informational: benchmark includes the cross-trial
                          mean and all K trials.  When the trials are siblings
                          sharing a real edge this asks whether the chosen
                          variant beats its siblings, not whether an edge exists
    psr_zero              informational: probabilistic Sharpe ratio against
                          zero, i.e. what the gate would be with N_eff = 1

    ``prior_eff_trials`` carries effective trials from EARLIER research on data
    that has since been looked at (N_eff_prior).  They are independent of the
    current family, so their Sharpe estimates vary with the sampling variance
    of a zero-edge Sharpe ratio, 1 / T per period, not with the (narrow)
    spread inside the current family:

        N_eff_total = N_eff_cur + N_eff_prior
        V_total     = 1 / T                                          (K_cur = 1)
        V_total     = (N_eff_cur V_cur + N_eff_prior / T) / N_eff_total   (K_cur > 1)
        SR_0        = sqrt(V_total) z(N_eff_total)

    With ``prior_eff_trials = 0`` nothing changes: SR_0 = sqrt(V_cur) z(N_eff_cur),
    and SR_0 = 0 when a single configuration was examined.
    """
    r = np.asarray(returns, dtype=float)
    s = np.asarray([] if trial_sharpes is None else trial_sharpes, dtype=float)
    s = s[np.isfinite(s)]
    k = int(n_trials) if n_trials is not None else max(int(len(s)), 1)
    prior = float(prior_eff_trials)
    if prior < 0:
        raise ValueError("prior_eff_trials must be >= 0")
    n_cur = effective_trials(k, rho_bar) if k > 1 else 1.0
    v_cur = float(s.var(ddof=1)) if (k > 1 and len(s) >= 2) else 0.0
    if prior > 0:
        n_tot = n_cur + prior
        t_days = max(len(r), 1)
        v_tot = 1.0 / t_days if k <= 1 else (n_cur * v_cur + prior / t_days) / n_tot
        sr0 = float(max(0.0, np.sqrt(v_tot) * _expected_max_z(n_tot)))
    else:
        n_tot, v_tot = n_cur, v_cur
        sr0 = expected_max_sharpe_zero_null(s, n_cur)
    sr0_mean = expected_max_sharpe(s, k) if len(s) >= 2 else float("nan")
    return {
        "dsr": probabilistic_sharpe_ratio(r, sr0),
        "dsr_cross_trial_mean": probabilistic_sharpe_ratio(r, sr0_mean) if np.isfinite(sr0_mean) else float("nan"),
        "psr_zero": probabilistic_sharpe_ratio(r, 0.0),
        "sr_per_period": sharpe_per_period(r),
        "sr0_per_period": sr0,
        "sr0_cross_trial_mean": sr0_mean,
        "n_trials": k, "n_eff": float(n_tot), "rho_bar": float(rho_bar) if k > 1 else 1.0,
        "trial_sr_mean": float(s.mean()) if len(s) else np.nan,
        "trial_sr_std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "n_eff_current": float(n_cur), "prior_eff_trials": prior, "sr_variance_total": float(v_tot),
    }


def slice_stability(result: BacktestResult, months: int = 6, min_fraction: float = 0.8) -> dict:
    """Net P&L of consecutive NON-overlapping calendar slices of one equity
    curve (informational: how evenly the result is spread over time).

    A slice is profitable when the equity at its last close exceeds the equity
    entering it; trades are attributed by entry date.  A final slice shorter
    than ``min_fraction`` of a full one is listed but not counted.  Slices in
    which nothing was traded and nothing was carried are reported separately
    as inactive rather than as losses.
    """
    eq = result.equity
    rows: list[dict] = []
    if len(eq) == 0:
        return {"months": months, "n_slices": 0, "profitable": 0, "losing": 0, "inactive": 0, "active": 0,
                "share_profitable": float("nan"), "share_profitable_active": float("nan"), "table": rows}
    idx = eq.index
    t = result.real_trades()
    ent = pd.DatetimeIndex(t["entry_session"]) if len(t) else pd.DatetimeIndex([])
    k = 0
    while True:
        a = idx[0] + pd.DateOffset(months=months * k)
        b = idx[0] + pd.DateOffset(months=months * (k + 1))
        if a > idx[-1]:
            break
        sub = eq[(idx >= a) & (idx < b)]
        if len(sub):
            before = eq[idx < a]
            init = float(before.iloc[-1]) if len(before) else float(result.initial_equity)
            n_tr = int(((ent >= a) & (ent < b)).sum())
            pnl = float(sub.iloc[-1]) - init
            rows.append({"slice": k, "start": sub.index[0], "end": sub.index[-1], "sessions": int(len(sub)),
                         "net_pnl": pnl, "trades": n_tr, "complete": bool(len(sub) >= min_fraction * 21 * months),
                         "inactive": bool(n_tr == 0 and abs(pnl) < 1e-9)})
        k += 1
    full = [x for x in rows if x["complete"]]
    inactive = sum(x["inactive"] for x in full)
    prof = sum(x["net_pnl"] > 0 for x in full)
    active = len(full) - inactive
    return {"months": months, "n_slices": len(full), "profitable": int(prof), "losing": int(active - prof),
            "inactive": int(inactive), "active": int(active),
            "share_profitable": prof / len(full) if full else float("nan"),
            "share_profitable_active": prof / active if active else float("nan"), "table": rows}


# --------------------------------------------------------------------------- #
# CSCV Probability of Backtest Overfitting
# --------------------------------------------------------------------------- #


def cscv_pbo(returns_matrix: np.ndarray, n_splits: int = 16) -> dict:
    """Combinatorially symmetric cross-validation (Bailey et al. 2017).

    ``returns_matrix`` is (T, N): one column of period returns per
    configuration on common dates.  Rows are cut into ``n_splits`` contiguous
    blocks; for each of the C(S, S/2) train/test splits the in-sample best
    configuration's out-of-sample rank is converted to a logit.  Ties are
    resolved by mid-ranks (audit fix).

    pbo            rank PBO = P(logit <= 0).  Among near-identical variants
                   the in-sample ranking is a coin flip, so this statistic is
                   only informative when the variants differ in sample:
                   ``is_sharpe_iqr`` is the median across splits of the
                   interquartile range of the annualised in-sample Sharpe.
    prob_oos_loss  P(SR_OOS(theta*) <= 0): how often the in-sample winner
                   loses money out of sample (Bailey et al. 2017, sec. 3.2).
    """
    M = np.asarray(returns_matrix, dtype=float)
    T, N = M.shape
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be a positive even number")
    if N < 2 or T < 2 * n_splits:
        raise ValueError("need at least 2 configurations and 2 rows per block")
    b = T // n_splits
    blocks = M[: b * n_splits].reshape(n_splits, b, N)
    s1 = blocks.sum(axis=1)                 # (S, N)
    s2 = (blocks**2).sum(axis=1)
    combos = np.array(list(combinations(range(n_splits), n_splits // 2)))
    ind = np.zeros((len(combos), n_splits))
    np.put_along_axis(ind, combos, 1.0, axis=1)

    def sharpe(mask: np.ndarray) -> np.ndarray:
        n = mask.sum(axis=1, keepdims=True) * b
        mean = (mask @ s1) / n
        var = (mask @ s2) / n - mean**2
        var = var * n / (n - 1.0)
        return np.where(var > 1e-18, mean / np.sqrt(np.clip(var, 1e-18, None)), 0.0)

    sr_is, sr_oos = sharpe(ind), sharpe(1.0 - ind)
    best = np.argmax(sr_is, axis=1)
    ranks = rankdata(sr_oos, method="average", axis=1)          # 1 = worst, N = best
    omega = ranks[np.arange(len(combos)), best] / (N + 1.0)
    logits = np.log(omega / (1.0 - omega))
    rows = np.arange(len(combos))
    slope = np.polyfit(sr_is[rows, best], sr_oos[rows, best], 1)[0] if len(combos) > 2 else np.nan
    return {
        "pbo": float(np.mean(logits <= 0.0)), "n_combinations": int(len(combos)), "n_configs": int(N),
        "block_length": int(b), "logit_median": float(np.median(logits)),
        "is_oos_slope": float(slope),
        "prob_oos_loss": float(np.mean(sr_oos[rows, best] <= 0.0)),
        "is_sharpe_iqr": float(np.median(np.subtract(*np.percentile(sr_is, [75, 25], axis=1))) * np.sqrt(ANN)),
    }


def returns_matrix(results: list[BacktestResult]) -> np.ndarray:
    """Daily returns of several runs on their COMMON dates as (T, N)."""
    df = pd.concat([r.daily_returns().rename(i) for i, r in enumerate(results)], axis=1, join="inner").fillna(0.0)
    return df.to_numpy()
