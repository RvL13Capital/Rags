"""Benchmark-relative inference for a strategy that holds a continuous weight.

A managed-exposure strategy has no round trips to count: its "trade" is a
weight that moves a little every day.  Profit factor and a trade floor say
nothing about it.  What can be asked is whether it adds to simply holding the
benchmark:

* spanning regression   R_s = alpha + beta R_b + e, Newey-West t(alpha) and
  the appraisal ratio alpha / sd(e);
* Ledoit & Wolf (2008)  studentised circular block bootstrap for the
  difference of two Sharpe ratios, extended to the difference of two
  mean-variance certainty equivalents (same delta-method machinery);
* a paired block bootstrap for any other statistic of two return series.

All inputs are per-period (daily) returns; annualisation is by ``periods``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ANN = 252.0


def newey_west_lags(n: int) -> int:
    """Newey & West (1994) plug-in rule of thumb, floor(4 (n / 100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def long_run_covariance(y: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-kernel (Newey-West) long-run covariance of the columns of a
    demeaned (T, k) array."""
    y = np.asarray(y, dtype=float)
    t = len(y)
    s = y.T @ y / t
    for lag in range(1, min(lags, t - 1) + 1):
        g = y[lag:].T @ y[:-lag] / t
        s += (1.0 - lag / (lags + 1.0)) * (g + g.T)
    return s


def spanning_regression(strategy: np.ndarray, benchmark: np.ndarray, lags: int | None = None,
                        periods: float = ANN) -> dict:
    """OLS of strategy on benchmark returns with HAC standard errors.

    ``alpha_ann`` is alpha times ``periods``; ``appraisal_ratio`` is
    alpha / sd(residual) annualised by sqrt(periods)."""
    y, x = np.asarray(strategy, dtype=float), np.asarray(benchmark, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 30:
        raise ValueError("need at least 30 paired observations")
    lags = newey_west_lags(n) if lags is None else int(lags)
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    u = y - X @ b
    s = long_run_covariance(X * u[:, None], lags) * n            # sum form of the HAC "meat"
    cov = xtx_inv @ s @ xtx_inv * n / (n - 2)
    se = np.sqrt(np.diag(cov))
    sd_e = float(u.std(ddof=2))
    return {
        "alpha": float(b[0]), "beta": float(b[1]), "se_alpha": float(se[0]), "se_beta": float(se[1]),
        "t_alpha": float(b[0] / se[0]) if se[0] > 0 else float("nan"),
        "t_beta": float(b[1] / se[1]) if se[1] > 0 else float("nan"),
        "alpha_ann": float(b[0] * periods), "resid_sd_ann": sd_e * float(np.sqrt(periods)),
        "appraisal_ratio": float(b[0] / sd_e * np.sqrt(periods)) if sd_e > 0 else float("nan"),
        "r2": float(1.0 - u.var() / y.var()) if y.var() > 0 else float("nan"), "n": int(n), "lags": lags,
    }


# --------------------------------------------------------------------------- #
# Differences of Sharpe ratios and certainty equivalents
# --------------------------------------------------------------------------- #


def _delta_and_gradient(v: np.ndarray, stat: str, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Statistic and gradient in v = (mu_a, mu_b, m2_a, m2_b), m2 = E[r^2].
    Works on a single vector or on a (B, 4) batch."""
    mu_a, mu_b, g_a, g_b = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    va, vb = g_a - mu_a**2, g_b - mu_b**2
    if stat == "sharpe":
        d = mu_a / np.sqrt(va) - mu_b / np.sqrt(vb)
        grad = np.stack([g_a / va**1.5, -g_b / vb**1.5, -0.5 * mu_a / va**1.5, 0.5 * mu_b / vb**1.5], axis=-1)
    elif stat == "cer":   # mean-variance certainty equivalent, mu - gamma / 2 * variance
        d = (mu_a - 0.5 * gamma * va) - (mu_b - 0.5 * gamma * vb)
        grad = np.stack([1.0 + gamma * mu_a, -(1.0 + gamma * mu_b), np.full_like(mu_a, -0.5 * gamma),
                         np.full_like(mu_a, 0.5 * gamma)], axis=-1)
    else:
        raise ValueError("stat must be 'sharpe' or 'cer'")
    return d, grad


def _moments(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack([a.mean(-1), b.mean(-1), (a * a).mean(-1), (b * b).mean(-1)], axis=-1)


def paired_difference_test(a: np.ndarray, b: np.ndarray, stat: str = "sharpe", gamma: float = 3.0, block: int = 10,
                           n_boot: int = 5000, seed: int = 0, lags: int | None = None, periods: float = ANN,
                           null_value: float = 0.0) -> dict:
    """Ledoit-Wolf (2008) studentised circular block bootstrap for
    delta = stat(a) - stat(b), with a and b observed on the same dates.

    The observed delta is studentised with a HAC standard error (delta
    method); each resample is studentised with the block-sum estimator of the
    paper, and the bootstrap distribution of the centred studentised statistic
    gives the p-values and the symmetric confidence interval.

    p_two_sided   H0: delta = null_value
    p_one_sided   H0: delta <= null_value against delta > null_value
    Values are per period; ``delta_ann`` scales a Sharpe difference by
    sqrt(periods) and a certainty-equivalent difference by periods.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    t = len(a)
    if t < 4 * block:
        raise ValueError("series too short for this block length")
    v = _moments(a, b)
    d_hat, grad = _delta_and_gradient(v, stat, gamma)
    y = np.column_stack([a - v[0], b - v[1], a * a - v[2], b * b - v[3]])
    psi = long_run_covariance(y, newey_west_lags(t) if lags is None else lags)
    se = float(np.sqrt(max(grad @ psi @ grad, 0.0) / t))
    rng = np.random.default_rng(seed)
    n_blocks = t // block
    offs = np.arange(block)
    z_one = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(400, n_boot - done)
        idx = (rng.integers(0, t, size=(m, n_blocks, 1)) + offs) % t           # (m, blocks, block)
        ra, rb = a[idx], b[idx]
        fa, fb = ra.reshape(m, -1), rb.reshape(m, -1)
        vs = _moments(fa, fb)
        ds, gs = _delta_and_gradient(vs, stat, gamma)
        ys = np.stack([ra - vs[:, None, None, 0], rb - vs[:, None, None, 1],
                       ra * ra - vs[:, None, None, 2], rb * rb - vs[:, None, None, 3]], axis=-1)   # (m, blocks, block, 4)
        zeta = ys.sum(axis=2) / np.sqrt(block)                                                    # (m, blocks, 4)
        psis = np.einsum("mji,mjk->mik", zeta, zeta) / n_blocks
        ses = np.sqrt(np.maximum(np.einsum("mi,mik,mk->m", gs, psis, gs), 0.0) / (n_blocks * block))
        with np.errstate(divide="ignore", invalid="ignore"):
            z_one[done:done + m] = (ds - d_hat) / ses
        done += m
    z_one = z_one[np.isfinite(z_one)]
    nb = len(z_one)
    scale = np.sqrt(periods) if stat == "sharpe" else periods
    if se <= 0 or nb == 0:      # the two series are the same (or differ by a constant multiple): nothing to test
        return {"stat": stat, "gamma": gamma if stat == "cer" else None, "delta": float(d_hat), "se": se,
                "delta_ann": float(d_hat * scale), "se_ann": float(se * scale), "z": float("nan"),
                "p_two_sided": 1.0, "p_one_sided": 1.0, "ci95_ann": (float(d_hat * scale), float(d_hat * scale)),
                "block": block, "n_boot": int(nb), "n": int(t)}
    z_obs = (d_hat - null_value) / se
    crit = float(np.quantile(np.abs(z_one), 0.95))
    return {
        "stat": stat, "gamma": gamma if stat == "cer" else None, "delta": float(d_hat), "se": se,
        "delta_ann": float(d_hat * scale), "se_ann": float(se * scale), "z": float(z_obs),
        "p_two_sided": float((np.sum(np.abs(z_one) >= abs(z_obs)) + 1) / (nb + 1)),
        "p_one_sided": float((np.sum(z_one >= z_obs) + 1) / (nb + 1)),
        "ci95_ann": (float((d_hat - crit * se) * scale), float((d_hat + crit * se) * scale)),
        "block": block, "n_boot": int(nb), "n": int(t),
    }


def paired_block_bootstrap(a: np.ndarray, b: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                           block: int = 10, n_boot: int = 5000, seed: int = 0) -> dict:
    """Percentile interval of ``fn(a*, b*)`` under a paired circular block
    bootstrap.  ``fn`` receives (m, T) arrays and returns m values."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    t = len(a)
    rng = np.random.default_rng(seed)
    n_blocks = t // block
    offs = np.arange(block)
    out = []
    done = 0
    while done < n_boot:
        m = min(400, n_boot - done)
        idx = ((rng.integers(0, t, size=(m, n_blocks, 1)) + offs) % t).reshape(m, -1)
        out.append(np.asarray(fn(a[idx], b[idx]), dtype=float))
        done += m
    x = np.concatenate(out)
    obs = float(np.asarray(fn(a[None, :], b[None, :]))[0])
    return {"observed": obs, "mean": float(x.mean()), "ci95": (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))),
            "share_positive": float((x > 0).mean()), "n_boot": int(len(x)), "block": block}


def sharpe_ann(r: np.ndarray, periods: float = ANN) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def cer_ann(r: np.ndarray, gamma: float, periods: float = ANN) -> float:
    """Mean-variance certainty equivalent, annualised: periods * (mean - gamma / 2 * variance)."""
    r = np.asarray(r, dtype=float)
    return float(periods * (r.mean() - 0.5 * gamma * r.var()))


def participation_ratio(corr: np.ndarray) -> float:
    """Effective number of independent components of a correlation matrix,
    (sum of eigenvalues)^2 / sum of squared eigenvalues: N for N uncorrelated
    series, 1 for N copies of one series."""
    lam = np.linalg.eigvalsh(np.asarray(corr, dtype=float))
    lam = np.clip(lam, 0.0, None)
    return float(lam.sum() ** 2 / (lam**2).sum())
