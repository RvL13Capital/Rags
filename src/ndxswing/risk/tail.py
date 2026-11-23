"""Fat-tail risk (section 4.E): POT-EVT VaR/ES, Tsallis q monitor, VaR/ES backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, genpareto, norm
from scipy.stats import t as student_t

Q_KURTOSIS_BOUNDARY = 1.40  # q >= 1.40 <=> nu_eff <= 4: fourth moment infinite


@dataclass(frozen=True)
class GPDFit:
    u: float        # threshold
    xi: float       # shape
    beta: float     # scale
    n: int          # sample size
    n_u: int        # exceedances

    def var(self, alpha: float) -> float:
        ratio = (self.n / self.n_u) * (1.0 - alpha)
        if abs(self.xi) < 1e-8:
            return self.u - self.beta * np.log(ratio)
        return self.u + (self.beta / self.xi) * (ratio ** (-self.xi) - 1.0)

    def es(self, alpha: float) -> float:
        if self.xi >= 1.0:
            return np.inf
        return (self.var(alpha) + self.beta - self.xi * self.u) / (1.0 - self.xi)


def fit_gpd_pot(losses: np.ndarray, threshold_q: float = 0.95, min_exceed: int = 15) -> GPDFit | None:
    """Peaks-over-threshold GPD fit to ``losses`` (positive = loss).

    Returns ``None`` when there are too few exceedances for a meaningful fit.
    """
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return None
    u = float(np.quantile(x, threshold_q))
    exc = x[x > u] - u
    if len(exc) < min_exceed or exc.std() <= 0:
        return None
    xi, _, beta = genpareto.fit(exc, floc=0.0)
    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0:
        return None
    return GPDFit(u=u, xi=float(xi), beta=float(beta), n=len(x), n_u=len(exc))


def empirical_es(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical (VaR, ES) of the loss sample at level alpha."""
    x = np.sort(np.asarray(losses, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    var = float(np.quantile(x, alpha))
    tail = x[x >= var]
    return var, float(tail.mean()) if len(tail) else var


def evt_var_es(losses: np.ndarray, alpha: float, threshold_q: float = 0.95) -> tuple[float, float, str]:
    """(VaR, ES, method).  EVT when a valid fit with xi < 1 exists, else empirical."""
    fit = fit_gpd_pot(losses, threshold_q)
    if fit is not None and fit.xi < 1.0 and alpha > threshold_q:
        return float(fit.var(alpha)), float(fit.es(alpha)), "evt_gpd"
    v, e = empirical_es(losses, alpha)
    return v, e, "empirical"


def rolling_evt(returns: pd.Series, window: int = 504, refit_every: int = 5) -> pd.DataFrame:
    """Causal rolling daily VaR/ES for long (loss = -r) and short (loss = +r)
    exposure, as positive return fractions.  Row t uses returns up to t."""
    r = returns.to_numpy(dtype=float)
    cols = ["var99_long", "es99_long", "es975_long", "var99_short", "es99_short", "es975_short", "gpd_xi_long"]
    out = np.full((len(r), len(cols)), np.nan)
    last = None
    first = int(np.argmax(np.isfinite(r)))
    for t in range(first + window - 1, len(r)):
        if last is None or (t - first - window + 1) % refit_every == 0:
            w = r[t - window + 1:t + 1]
            row, xi_long = [], np.nan
            for sign in (-1.0, 1.0):
                loss = sign * w
                fit = fit_gpd_pot(loss)
                if fit is not None and fit.xi < 1.0:
                    row += [fit.var(0.99), fit.es(0.99), fit.es(0.975)]
                else:
                    v99, e99 = empirical_es(loss, 0.99)
                    row += [v99, e99, empirical_es(loss, 0.975)[1]]
                if sign < 0 and fit is not None:
                    xi_long = fit.xi
            last = row + [xi_long]
        out[t] = last
    return pd.DataFrame(out, index=returns.index, columns=cols)


def tsallis_q_from_returns(returns: np.ndarray) -> tuple[float, float]:
    """Tsallis q and effective Student-t degrees of freedom.

    A q-Gaussian with q > 1 is a rescaled Student-t with
    nu = 2 / (q - 1) - 1, so q is obtained from the t MLE: q = 1 + 2 / (nu + 1).
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 60 or x.std() <= 0:
        return np.nan, np.nan
    nu, _, _ = student_t.fit((x - x.mean()) / x.std())
    nu = float(np.clip(nu, 1.01, 1e6))
    return 1.0 + 2.0 / (nu + 1.0), nu


def rolling_tsallis_q(returns: pd.Series, window: int = 504, refit_every: int = 5) -> pd.DataFrame:
    r = returns.to_numpy(dtype=float)
    out = np.full((len(r), 2), np.nan)
    first = int(np.argmax(np.isfinite(r)))
    last = (np.nan, np.nan)
    for t in range(first + window - 1, len(r)):
        if (t - first - window + 1) % refit_every == 0:
            last = tsallis_q_from_returns(r[t - window + 1:t + 1])
        out[t] = last
    df = pd.DataFrame(out, index=returns.index, columns=["tsallis_q", "nu_eff"])
    df["tail_alert"] = df["tsallis_q"] >= Q_KURTOSIS_BOUNDARY
    return df


# --------------------------------------------------------------------------- #
# Out-of-sample VaR / ES forecast validation
# --------------------------------------------------------------------------- #


def kupiec_pof(violations: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR test.  ``violations`` is boolean
    (loss > VaR_alpha).  Returns (LR, p-value)."""
    v = np.asarray(violations, dtype=bool)
    n, x = len(v), int(v.sum())
    p = 1.0 - alpha
    if n == 0:
        return np.nan, np.nan
    phat = x / n

    def ll(q: float) -> float:
        q = min(max(q, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - q) + x * np.log(q)

    lr = -2.0 * (ll(p) - ll(phat))
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen_independence(violations: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) first-order Markov independence LR test."""
    v = np.asarray(violations, dtype=int)
    if len(v) < 2:
        return np.nan, np.nan
    a, b = v[:-1], v[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    if n01 + n11 == 0:
        return 0.0, 1.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)

    def xlogy(k: int, q: float) -> float:
        return 0.0 if k == 0 else k * np.log(max(q, 1e-300))

    l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    l1 = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    lr = max(-2.0 * (l0 - l1), 0.0)
    return float(lr), float(chi2.sf(lr, 1))


def acerbi_szekely_z2(losses: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> tuple[float, float]:
    """Acerbi-Szekely (2014) unconditional ES test statistic Z2.

    Z2 = sum_t L_t 1[L_t > VaR_t] / (T (1 - alpha) ES_t) - 1 ; E[Z2] = 0 under
    a correct forecast and Z2 > 0 signals under-estimated tail risk.  The
    p-value uses the normal approximation with the sample dispersion of the
    summands (the distribution-free thresholds of the paper are ~0.7 at 5%).
    """
    loss, var, es = (np.asarray(a, dtype=float) for a in (losses, var, es))
    m = np.isfinite(loss) & np.isfinite(var) & np.isfinite(es) & (es > 0)
    loss, var, es = loss[m], var[m], es[m]
    T = len(loss)
    if T < 50:
        return np.nan, np.nan
    terms = loss * (loss > var) / ((1.0 - alpha) * es)
    z2 = float(terms.mean() - 1.0)
    se = float(terms.std(ddof=1) / np.sqrt(T))
    return z2, float(norm.sf(z2 / se)) if se > 0 else np.nan


def validate_var_es(returns: pd.Series, var: pd.Series, es: pd.Series, alpha: float) -> dict[str, float]:
    """Backtest one-day-ahead forecasts made at t against the loss at t+1."""
    loss_next = -returns.shift(-1)
    df = pd.DataFrame({"loss": loss_next, "var": var, "es": es}).dropna()
    viol = (df["loss"] > df["var"]).to_numpy()
    lr_pof, p_pof = kupiec_pof(viol, alpha)
    lr_ind, p_ind = christoffersen_independence(viol)
    z2, p_z2 = acerbi_szekely_z2(df["loss"].to_numpy(), df["var"].to_numpy(), df["es"].to_numpy(), alpha)
    return {
        "n": len(df), "violations": int(viol.sum()), "expected": len(df) * (1 - alpha),
        "kupiec_p": p_pof, "christoffersen_p": p_ind, "acerbi_szekely_z2": z2, "acerbi_szekely_p": p_z2,
    }
