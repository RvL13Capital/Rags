"""ARMA/AR - GJR-GARCH - Student-t forecasts with exact multi-horizon variance
propagation (section 4.C).

Model (daily log returns r_t):

    r_t      = c + phi r_{t-1} + theta eps_{t-1} + eps_t,   eps_t = sigma_t z_t
    sigma_t^2 = omega + (alpha + gamma 1[eps_{t-1} < 0]) eps_{t-1}^2 + beta sigma_{t-1}^2

Cumulative H-day forecast error is  sum_{h=1..H} Psi_{H-h} eps_{t+h}  with Wold
weights psi_0 = 1, psi_k = phi^{k-1} (phi + theta) and Psi_m = sum_{k<=m} psi_k,
hence

    sigma_hat_{t,H}^2 = sum_{h=1..H} Psi_{H-h}^2 E_t[sigma_{t+h}^2]

The one-day volatility ``sigma_1d = sqrt(E_t[sigma_{t+1}^2])`` is exported
separately and is the only quantity that may enter the daily vol target.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import chi2

HORIZONS = (1, 2, 3, 5)


@dataclass(frozen=True)
class ArmaGjrParams:
    c: float
    phi: float
    theta: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float

    @property
    def persistence(self) -> float:
        # symmetric innovations: P(z < 0) = 1/2
        return self.alpha + 0.5 * self.gamma + self.beta


def fit_arma_gjr_garch_t(returns: np.ndarray, mean: str = "ar1", vol: str = "gjr") -> ArmaGjrParams:
    """Maximum likelihood fit.  ``returns`` are decimal log returns.

    mean="ar1"     joint MLE of AR(1) + (GJR-)GARCH(1,1) + Student-t
    mean="arma11"  two-step: Gaussian ARMA(1,1) for the mean, then zero-mean
                   (GJR-)GARCH-t on its residuals (consistent, not efficient)
    vol="gjr"|"garch"
    """
    y = 100.0 * np.asarray(returns, dtype=float)
    o = 1 if vol == "gjr" else 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mean == "ar1":
            res = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, o=o, q=1, dist="t", rescale=False).fit(
                disp="off", show_warning=False
            )
            p = res.params
            c, phi, theta = float(p["Const"]) / 100.0, float(p.iloc[1]), 0.0
        elif mean == "arma11":
            from statsmodels.tsa.arima.model import ARIMA

            arma = ARIMA(y, order=(1, 0, 1), trend="c").fit()
            phi, theta = float(arma.arparams[0]), float(arma.maparams[0])
            # statsmodels reports the unconditional mean; convert to intercept
            c = float(arma.params[0]) * (1.0 - phi) / 100.0
            res = arch_model(arma.resid, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist="t", rescale=False).fit(
                disp="off", show_warning=False
            )
            p = res.params
        else:
            raise ValueError(f"unknown mean specification {mean!r}")
    return ArmaGjrParams(
        c=c, phi=phi, theta=theta,
        omega=float(p["omega"]) / 1e4, alpha=float(p["alpha[1]"]),
        gamma=float(p["gamma[1]"]) if o else 0.0, beta=float(p["beta[1]"]), nu=float(p["nu"]),
    )


def filter_arma_gjr(returns: np.ndarray, p: ArmaGjrParams) -> tuple[np.ndarray, np.ndarray]:
    """Run the recursion with fixed parameters; returns (eps, sigma2).

    Initialised at the sample variance; element t uses returns up to t only.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    eps, s2 = np.empty(n), np.empty(n)
    uncond_mean = p.c / (1.0 - p.phi) if abs(p.phi) < 1 else float(r.mean())
    eps[0] = r[0] - uncond_mean
    s2[0] = max(float(np.var(r)), 1e-12)
    for t in range(1, n):
        e1 = eps[t - 1]
        s2[t] = p.omega + (p.alpha + (p.gamma if e1 < 0 else 0.0)) * e1 * e1 + p.beta * s2[t - 1]
        eps[t] = r[t] - p.c - p.phi * r[t - 1] - p.theta * e1
    return eps, s2


def forecast_arma_gjr_garch_exact(
    p: ArmaGjrParams, r_last: float, eps_last: float, sigma2_last: float, horizons: tuple[int, ...] = HORIZONS
) -> dict[str, float]:
    """Exact conditional mean / volatility of 1..H day cumulative returns.

    Returns ``sigma_1d`` plus ``mu_H``, ``sigma_H`` and ``z_H`` per horizon.
    """
    H = max(horizons)
    s2 = np.empty(H + 1)
    m = np.empty(H + 1)
    s2[1] = p.omega + (p.alpha + (p.gamma if eps_last < 0 else 0.0)) * eps_last**2 + p.beta * sigma2_last
    m[1] = p.c + p.phi * r_last + p.theta * eps_last
    for h in range(2, H + 1):
        s2[h] = p.omega + p.persistence * s2[h - 1]
        m[h] = p.c + p.phi * m[h - 1]
    psi = np.empty(H)
    psi[0] = 1.0
    for k in range(1, H):
        psi[k] = p.phi ** (k - 1) * (p.phi + p.theta)
    Psi = np.cumsum(psi)
    out = {"sigma_1d": float(np.sqrt(s2[1]))}
    for hz in horizons:
        mu = float(m[1:hz + 1].sum())
        var = float(sum(Psi[hz - h] ** 2 * s2[h] for h in range(1, hz + 1)))
        sig = float(np.sqrt(var))
        out[f"mu_{hz}"], out[f"sigma_{hz}"], out[f"z_{hz}"] = mu, sig, mu / sig if sig > 0 else 0.0
    return out


def ljung_box_pvalue(x: np.ndarray, lags: int, model_df: int = 0) -> float:
    """Ljung-Box Q(lags) p-value with ``lags - model_df`` degrees of freedom."""
    z = np.asarray(x, dtype=float)
    z = z - z.mean()
    n = len(z)
    denom = z @ z
    if n <= lags + 1 or denom <= 0:
        return np.nan
    q = 0.0
    for k in range(1, lags + 1):
        rho = (z[k:] @ z[:-k]) / denom
        q += rho * rho / (n - k)
    q *= n * (n + 2.0)
    return float(chi2.sf(q, max(lags - model_df, 1)))


def rolling_garch_forecasts(
    returns: pd.Series,
    window: int = 756,
    refit_every: int = 5,
    horizons: tuple[int, ...] = HORIZONS,
    mean: str = "ar1",
    vol: str = "gjr",
) -> pd.DataFrame:
    """Causal rolling forecasts.  Parameters are re-estimated every
    ``refit_every`` days on the trailing ``window`` returns ending at the refit
    day; in between the recursion is rolled forward with frozen parameters.
    Row t holds forecasts made at the close of day t for days t+1..t+H.

    ``lb_p10`` / ``lb_p20`` are Ljung-Box p-values of the standardised
    residuals of the trailing window; ``garch_ok`` requires both > 0.05, a
    stationary variance recursion and a successful fit.
    """
    r = returns.to_numpy(dtype=float)
    n = len(r)
    cols = ["sigma_1d"] + [f"{k}_{h}" for h in horizons for k in ("mu", "sigma", "z")]
    cols += ["lb_p10", "lb_p20", "nu", "persistence", "garch_ok"]
    out = pd.DataFrame(np.nan, index=returns.index, columns=cols)
    first = int(np.argmax(np.isfinite(r)))
    params: ArmaGjrParams | None = None
    since = refit_every
    fit_ok = False
    model_df = 2 if mean == "arma11" else 1
    rows: list[tuple] = []
    for t in range(first + window - 1, n):
        w = r[t - window + 1:t + 1]
        if since >= refit_every or params is None:
            try:
                params = fit_arma_gjr_garch_t(w, mean, vol)
                fit_ok = bool(
                    np.isfinite([params.c, params.phi, params.omega, params.alpha, params.beta, params.nu]).all()
                    and params.persistence < 1.0 and abs(params.phi) < 1.0 and params.omega > 0
                )
            except Exception:
                fit_ok = False
            since = 0
        since += 1
        if params is None:
            continue
        eps, s2 = filter_arma_gjr(w, params)
        z = eps / np.sqrt(s2)
        fc = forecast_arma_gjr_garch_exact(params, w[-1], eps[-1], s2[-1], horizons)
        p10 = ljung_box_pvalue(z[1:], 10, model_df)
        p20 = ljung_box_pvalue(z[1:], 20, model_df)
        ok = fit_ok and p10 > 0.05 and p20 > 0.05
        rows.append((t, fc, p10, p20, params.nu, params.persistence, ok))
    for t, fc, p10, p20, nu, pers, ok in rows:
        vals = [fc["sigma_1d"]] + [fc[f"{k}_{h}"] for h in horizons for k in ("mu", "sigma", "z")]
        out.iloc[t] = vals + [p10, p20, nu, pers, float(ok)]
    out["garch_ok"] = out["garch_ok"].fillna(0.0).astype(bool)
    return out


def pesaran_timmermann(actual: np.ndarray, forecast: np.ndarray) -> tuple[float, float]:
    """Pesaran-Timmermann (1992) test of directional accuracy.

    Returns (statistic, one-sided p-value) for H0: forecast and realised signs
    are independent.  Used to decide whether the ARMA directional filter may
    be enabled at all (section 8.1).
    """
    from scipy.stats import norm

    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    m = np.isfinite(a) & np.isfinite(f) & (f != 0)
    a, f = a[m] > 0, f[m] > 0
    n = len(a)
    if n < 30:
        return np.nan, np.nan
    p_hit = np.mean(a == f)
    pa, pf = a.mean(), f.mean()
    p_star = pa * pf + (1 - pa) * (1 - pf)
    v_hit = p_star * (1 - p_star) / n
    v_star = ((2 * pa - 1) ** 2 * pf * (1 - pf) + (2 * pf - 1) ** 2 * pa * (1 - pa)) / n + 4 * pa * pf * (1 - pa) * (
        1 - pf
    ) / n**2
    if v_hit - v_star <= 0:
        return np.nan, np.nan
    stat = (p_hit - p_star) / np.sqrt(v_hit - v_star)
    return float(stat), float(norm.sf(stat))
