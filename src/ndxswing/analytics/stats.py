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
    txn = float(t[["commission", "spread_cost", "slippage_cost"]].to_numpy(dtype=float).sum()) if nt else 0.0
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


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes: np.ndarray, n_trials: int | None = None) -> dict:
    sr0 = expected_max_sharpe(trial_sharpes, n_trials)
    return {
        "dsr": probabilistic_sharpe_ratio(returns, sr0),
        "sr_per_period": sharpe_per_period(returns),
        "sr0_per_period": sr0,
        "n_trials": int(n_trials) if n_trials is not None else int(np.isfinite(trial_sharpes).sum()),
        "trial_sr_mean": float(np.nanmean(trial_sharpes)) if len(trial_sharpes) else np.nan,
        "trial_sr_std": float(np.nanstd(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else np.nan,
    }


# --------------------------------------------------------------------------- #
# CSCV Probability of Backtest Overfitting
# --------------------------------------------------------------------------- #


def cscv_pbo(returns_matrix: np.ndarray, n_splits: int = 16) -> dict:
    """Combinatorially symmetric cross-validation (Bailey et al. 2017).

    ``returns_matrix`` is (T, N): one column of period returns per
    configuration on common dates.  Rows are cut into ``n_splits`` contiguous
    blocks; for each of the C(S, S/2) train/test splits the in-sample best
    configuration's out-of-sample rank is converted to a logit.  Ties are
    resolved by mid-ranks (audit fix).  PBO = P(logit <= 0).
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
        "prob_oos_loss": float(np.mean(sr_oos[rows, best] < 0.0)),
    }


def returns_matrix(results: list[BacktestResult]) -> np.ndarray:
    """Align daily returns of several runs over the same window into (T, N)."""
    df = pd.concat([r.daily_returns().rename(i) for i, r in enumerate(results)], axis=1).fillna(0.0)
    return df.to_numpy()
