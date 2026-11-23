"""Causal Markov regime classifier (section 4.A).

The model is re-estimated on a rolling window by EM and *applied* strictly via
Hamilton's forward recursion:

    xi_{t|t-1} = A^T xi_{t-1|t-1}
    xi_{t|t,k} = xi_{t|t-1,k} f_k(X_t) / sum_j xi_{t|t-1,j} f_j(X_t)

Smoothed probabilities are used only inside a training window to align state
labels after a refit; they never appear in the output for any date.  Inputs
are standardised with training-window moments and winsorised at +/-4 sigma so
that crash observations do not capture a Gaussian emission mean.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

FEATURES = ("r", "rv20", "slope20", "vc", "gap", "vtilde20")


@dataclass(frozen=True)
class RegimeParams:
    """Fitted, label-aligned parameters.  State order is (trend, sideways, stress)
    for three states and (calm, stress) for two."""

    mean_: np.ndarray       # training-window feature means used for standardisation
    std_: np.ndarray
    start: np.ndarray       # (K,)
    trans: np.ndarray       # (K, K) row-stochastic
    mu: np.ndarray          # (K, d)
    cov: np.ndarray         # (K, d, d)
    winsor: float
    loglik: float


def _log_emission(z: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """log N(z_t; mu_k, cov_k) for all t, k."""
    T, d = z.shape
    out = np.empty((T, mu.shape[0]))
    for k in range(mu.shape[0]):
        L = np.linalg.cholesky(cov[k])
        y = np.linalg.solve(L, (z - mu[k]).T)
        out[:, k] = -0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + (y**2).sum(axis=0))
    return out


def hamilton_filter(logf: np.ndarray, trans: np.ndarray, xi0: np.ndarray) -> np.ndarray:
    """Forward (filtered) probabilities.  ``xi0`` is xi_{0|0} of the bar before
    the first row of ``logf``; row t depends on rows <= t only."""
    T, K = logf.shape
    xi = np.empty((T, K))
    prev = xi0
    AT = trans.T
    for t in range(T):
        pred = AT @ prev
        w = pred * np.exp(logf[t] - logf[t].max())
        s = w.sum()
        prev = w / s if s > 0 and np.isfinite(s) else pred
        xi[t] = prev
    return xi


def confidence(xi: np.ndarray) -> np.ndarray:
    """Conf_t = 1 + sum_k xi_k ln xi_k / ln K  (1 = certain, 0 = uniform)."""
    K = xi.shape[1]
    p = np.clip(xi, 1e-300, 1.0)
    return 1.0 + (p * np.log(p)).sum(axis=1) / np.log(K)


class CausalRegimeHMM:
    def __init__(
        self,
        n_states: int = 3,
        window: int = 756,
        refit_every: int = 21,
        n_init: int = 4,
        n_iter: int = 100,
        winsor: float = 4.0,
        seed: int = 7,
        features: tuple[str, ...] = FEATURES,
        min_covar: float = 1e-3,
    ) -> None:
        if n_states not in (2, 3):
            raise ValueError("n_states must be 2 or 3")
        self.n_states, self.window, self.refit_every = n_states, window, refit_every
        self.n_init, self.n_iter, self.winsor, self.seed = n_init, n_iter, winsor, seed
        self.features, self.min_covar = tuple(features), min_covar

    # ------------------------------------------------------------------ fit
    def _standardise(self, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return np.clip((X - mean) / std, -self.winsor, self.winsor)

    def fit(self, X: np.ndarray) -> RegimeParams:
        """EM on one training window (rows are consecutive days, no NaN)."""
        mean, std = X.mean(axis=0), X.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        Z = self._standardise(X, mean, std)
        best, best_ll = None, -np.inf
        for i in range(self.n_init):
            m = GaussianHMM(
                n_components=self.n_states, covariance_type="full", n_iter=self.n_iter,
                tol=1e-4, min_covar=self.min_covar, random_state=self.seed + 1000 * i,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(Z)
                    ll = float(m.score(Z))
            except Exception:  # degenerate initialisation; try the next seed
                continue
            if np.isfinite(ll) and ll > best_ll and np.all(np.isfinite(m.transmat_)):
                best, best_ll = m, ll
        if best is None:
            raise RuntimeError("regime HMM failed to converge on every initialisation")
        order = self._align(best, Z)
        trans = best.transmat_[np.ix_(order, order)]
        trans = trans / trans.sum(axis=1, keepdims=True)
        start = np.clip(best.startprob_[order], 1e-6, None)
        return RegimeParams(
            mean_=mean, std_=std, start=start / start.sum(), trans=trans,
            mu=best.means_[order].copy(), cov=best.covars_[order].copy(),
            winsor=self.winsor, loglik=best_ll,
        )

    def _align(self, model: GaussianHMM, Z: np.ndarray) -> list[int]:
        """Deterministic state alignment (anti label switching).

        stress = argmax_k E[RV20 | S=k]; trend = argmax_{k != stress}
        E[|Slope20| | S=k]; the remaining state is sideways.  Expectations are
        posterior weighted over the training window.
        """
        gamma = model.predict_proba(Z)  # training-window only, used for labels
        w = gamma / np.clip(gamma.sum(axis=0, keepdims=True), 1e-12, None)
        i_rv = self.features.index("rv20")
        e_rv = w.T @ Z[:, i_rv]
        stress = int(np.argmax(e_rv))
        rest = [k for k in range(self.n_states) if k != stress]
        if self.n_states == 2:
            return [rest[0], stress]
        i_sl = self.features.index("slope20")
        e_abs_slope = w.T @ np.abs(Z[:, i_sl] * 1.0)
        trend = max(rest, key=lambda k: e_abs_slope[k])
        side = [k for k in rest if k != trend][0]
        return [trend, side, stress]

    # --------------------------------------------------------------- filter
    def filter(self, X: np.ndarray, params: RegimeParams, xi0: np.ndarray | None = None) -> np.ndarray:
        Z = np.clip((X - params.mean_) / params.std_, -params.winsor, params.winsor)
        logf = _log_emission(Z, params.mu, params.cov)
        return hamilton_filter(logf, params.trans, params.start if xi0 is None else xi0)

    def rolling_filter(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Filtered probabilities for every day, refitting every ``refit_every``
        days on the trailing ``window`` days ending at (and including) the
        refit day.  Row t is a function of observations up to day t only."""
        X_all = obs[list(self.features)].to_numpy(dtype=float)
        valid = np.isfinite(X_all).all(axis=1)
        first = int(np.argmax(valid)) if valid.any() else len(obs)
        if not valid[first:].all():
            raise ValueError("regime observations contain NaN after the warm-up period")
        names = ["p_trend", "p_side", "p_stress"] if self.n_states == 3 else ["p_calm", "p_stress"]
        out = np.full((len(obs), self.n_states), np.nan)
        refit_id = np.full(len(obs), -1, dtype=int)
        t = first + self.window - 1
        k = 0
        while t < len(obs):
            lo = t - self.window + 1
            try:
                params = self.fit(X_all[lo:t + 1])
            except RuntimeError:
                if k == 0:
                    raise
            # Re-run the filter through the training window so xi at the refit
            # day is conditioned on that window, then continue forward.
            end = min(t + self.refit_every, len(obs))
            xi = self.filter(X_all[lo:end], params)
            out[t:end] = xi[t - lo:]
            refit_id[t:end] = k
            t, k = end, k + 1
        df = pd.DataFrame(out, index=obs.index, columns=names)
        ok = np.isfinite(out).all(axis=1)
        conf = np.full(len(obs), np.nan)
        if ok.any():
            conf[ok] = confidence(out[ok])
        df["conf"] = conf
        df["refit_id"] = refit_id
        return df
