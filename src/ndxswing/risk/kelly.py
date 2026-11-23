"""Exact concave lower-confidence-bound fractional Kelly (section 6.1).

No Taylor expansion and no Gaussian mu/sigma^2 shortcut: the empirical
log-wealth objective is maximised directly on the trade sample after shifting
its mean down to a 20th-percentile lower confidence bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

Z_20PCT = 0.8416  # one-sided 80% bound (20th percentile of the mean estimate)


@dataclass(frozen=True)
class KellyResult:
    f_kelly: float      # c_kelly * f_star, a fraction of equity as exposure
    f_star: float       # full-Kelly maximiser on the shifted sample
    mu_hat: float
    mu_lcb: float
    n: int
    active: bool        # False while the sample is too small (Kelly bypassed)


def exact_concave_fractional_kelly(
    trade_returns: np.ndarray,
    c_kelly: float = 0.20,
    f_max: float = 10.0,
    min_trades: int = 30,
) -> KellyResult:
    """``trade_returns`` are UNLEVERAGED simple underlying returns per trade,
    net of costs, each > -1.

    * fewer than ``min_trades`` observations -> ``active=False`` (caller binds
      exposure with the ES / vol / leverage caps instead)
    * mu_LCB = mean - 0.8416 * sd / sqrt(M) <= 0 -> f_kelly = 0
    * otherwise maximise (1/M) sum ln(1 + f * R~_i) on
      f in [0, min(f_max, 0.95 / |min R~|)], R~_i = max(R_i - mean + mu_LCB, -0.95)
    """
    if not 0.10 <= c_kelly <= 0.25:
        raise ValueError("c_kelly must lie in [0.10, 0.25]")
    r = np.asarray(trade_returns, dtype=float)
    r = r[np.isfinite(r)]
    m = len(r)
    if m and r.min() <= -1.0:
        raise ValueError("trade returns must be simple returns > -1")
    if m < min_trades:
        return KellyResult(np.nan, np.nan, float(r.mean()) if m else np.nan, np.nan, m, False)
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    mu_lcb = mu - Z_20PCT * sd / np.sqrt(m)
    if mu_lcb <= 0.0:
        return KellyResult(0.0, 0.0, mu, mu_lcb, m, True)
    rt = np.maximum(r - mu + mu_lcb, -0.95)
    worst = float(rt.min())
    hi = f_max if worst >= 0 else min(f_max, 0.95 / abs(worst))

    def dgrowth(f: float) -> float:  # derivative of the concave objective
        return float(np.mean(rt / (1.0 + f * rt)))

    if dgrowth(0.0) <= 0.0:
        f_star = 0.0
    elif dgrowth(hi) >= 0.0:
        f_star = hi
    else:
        f_star = float(brentq(dgrowth, 0.0, hi, xtol=1e-10))
    return KellyResult(c_kelly * f_star, f_star, mu, mu_lcb, m, True)
