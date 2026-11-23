"""Fractal persistence (section 4.B): DFA Hurst, bootstrap lower bound, Lo's R/S.

DFA is applied to the *integrated* demeaned return profile
``Y(k) = sum_{i<=k} (r_i - mean r)``; forward and backward non-overlapping
segments are detrended with a local polynomial of order 1 or 2 and
``H = d log F(s) / d log s`` is the OLS slope over log-spaced scales in
``[16, L/4]``.  All routines accept a batch of series (rows) so that the
bootstrap runs vectorised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

LO_RS_UPPER_95 = 1.747  # one-sided 5% critical value of Lo's V statistic


def dfa_scales(L: int, s_min: int = 16, n_scales: int = 18) -> np.ndarray:
    s_max = L // 4
    if s_max <= s_min:
        raise ValueError(f"series of length {L} is too short for DFA scales >= {s_min}")
    return np.unique(np.round(np.geomspace(s_min, s_max, n_scales)).astype(int))


def dfa_fluctuation(returns: np.ndarray, scales: np.ndarray, order: int = 1) -> np.ndarray:
    """F(s) for each row of ``returns`` (shape (B, L)) and each scale."""
    r = np.atleast_2d(np.asarray(returns, dtype=float))
    B, L = r.shape
    Y = np.cumsum(r - r.mean(axis=1, keepdims=True), axis=1)
    # ||residual||^2 = ||segment||^2 - ||Q^T segment||^2 with Q an orthonormal
    # basis of the local polynomial; segment energies come from one cumsum.
    E = np.concatenate([np.zeros((B, 1)), np.cumsum(Y * Y, axis=1)], axis=1)
    F = np.empty((B, len(scales)))
    for j, s in enumerate(scales):
        n_seg = L // s
        m = n_seg * s
        Q, _ = np.linalg.qr(np.vander(np.arange(s, dtype=float), order + 1, increasing=True))
        pf = Y[:, :m].reshape(B, n_seg, s) @ Q
        pb = Y[:, L - m:].reshape(B, n_seg, s) @ Q
        ss = (E[:, m] - E[:, 0]) + (E[:, L] - E[:, L - m])
        ss -= np.einsum("bnk,bnk->b", pf, pf) + np.einsum("bnk,bnk->b", pb, pb)
        F[:, j] = np.sqrt(np.clip(ss, 0.0, None) / (2 * m))
    return F


def dfa_hurst(returns: np.ndarray, order: int = 1, scales: np.ndarray | None = None) -> np.ndarray:
    """DFA exponent for each row of ``returns``."""
    r = np.atleast_2d(np.asarray(returns, dtype=float))
    sc = dfa_scales(r.shape[1]) if scales is None else np.asarray(scales)
    F = dfa_fluctuation(r, sc, order)
    x = np.log(sc.astype(float))
    xc = x - x.mean()
    y = np.log(np.clip(F, 1e-300, None))
    return (y - y.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)


def stationary_bootstrap_indices(L: int, B: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (B, L), circular."""
    restart = rng.random((B, L)) < 1.0 / mean_block
    restart[:, 0] = True
    starts = rng.integers(0, L, size=(B, L))
    pos = np.arange(L)
    last = np.maximum.accumulate(np.where(restart, pos, 0), axis=1)
    block_start = np.take_along_axis(starts, last, axis=1)
    return (block_start + (pos - last)) % L


def hurst_with_lcb(
    returns: np.ndarray,
    order: int = 1,
    n_boot: int = 200,
    mean_block: float = 24.0,
    level: float = 0.95,
    method: str = "normal",
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Point estimate and one-sided lower confidence bound.

    ``method="normal"`` (default) uses the block-bootstrap standard error:
    ``H - z * sd(H*)``.  ``"percentile"`` returns the (1-level) quantile of the
    bootstrap distribution; because blocks of ~24 bars cannot carry dependence
    beyond the block length it is biased towards 0.5 for long-memory series and
    is therefore the more conservative choice for H > 0.5.
    """
    r = np.asarray(returns, dtype=float)
    L = len(r)
    sc = dfa_scales(L)
    h = float(dfa_hurst(r, order, sc)[0])
    rng = np.random.default_rng(0) if rng is None else rng
    idx = stationary_bootstrap_indices(L, n_boot, mean_block, rng)
    hb = dfa_hurst(r[idx], order, sc)
    if method == "normal":
        lcb = h - norm.ppf(level) * float(hb.std(ddof=1))
    elif method == "percentile":
        lcb = float(np.quantile(hb, 1.0 - level))
    elif method == "basic":
        lcb = 2.0 * h - float(np.quantile(hb, level))
    else:
        raise ValueError(f"unknown method {method!r}")
    return h, lcb


def lo_modified_rs(returns: np.ndarray, q: int | None = None) -> tuple[float, int]:
    """Lo (1991) HAC-modified rescaled range statistic V_n(q).

    ``q`` defaults to Andrews' data-dependent truncation lag.  Under the null
    of short memory V lies in [0.809, 1.862] with 95% probability; V > 1.747
    rejects at 5% one-sided in favour of long-range persistence.
    """
    x = np.asarray(returns, dtype=float)
    n = len(x)
    d = x - x.mean()
    if q is None:
        rho = float(np.clip((d[1:] @ d[:-1]) / (d @ d), -0.99, 0.99))
        q = int(np.floor((1.5 * n) ** (1 / 3) * (2 * abs(rho) / (1 - rho**2)) ** (2 / 3)))
        q = min(q, n // 4)
    var = d @ d / n
    for j in range(1, q + 1):
        var += 2.0 * (1.0 - j / (q + 1.0)) * (d[j:] @ d[:-j]) / n
    cs = np.cumsum(d)
    v = (cs.max() - cs.min()) / np.sqrt(max(var, 1e-300)) / np.sqrt(n)
    return float(v), int(q)


def rolling_hurst(
    hourly: pd.DataFrame,
    length: int = 1500,
    order: int = 1,
    n_boot: int = 200,
    mean_block: float = 24.0,
    method: str = "normal",
    seed: int = 11,
) -> pd.DataFrame:
    """Daily-updated Hurst estimates from the trailing ``length`` hourly log
    returns that end at each session's close.  Row d uses bars of sessions
    <= d only."""
    close = hourly["close"].to_numpy(dtype=float)
    lr = np.diff(np.log(close), prepend=np.nan)
    sess = hourly["session"].to_numpy()
    last_bar = np.flatnonzero(np.r_[sess[1:] != sess[:-1], True])
    sessions = pd.DatetimeIndex(sess[last_bar])
    out = np.full((len(last_bar), 3), np.nan)
    sc = None
    for j, e in enumerate(last_bar):
        if e - length + 1 < 1:  # element 0 of lr is NaN
            continue
        w = lr[e - length + 1:e + 1]
        if sc is None:
            sc = dfa_scales(length)
        rng = np.random.default_rng([seed, j])
        if n_boot > 0:
            h, lcb = hurst_with_lcb(w, order, n_boot, mean_block, 0.95, method, rng)
        else:
            h, lcb = float(dfa_hurst(w, order, sc)[0]), np.nan
        out[j] = h, lcb, lo_modified_rs(w)[0]
    return pd.DataFrame(out, index=sessions.rename("session"), columns=["hurst", "hurst_lcb", "lo_rs_v"])
