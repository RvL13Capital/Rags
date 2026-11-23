"""Non-Gaussian Monte Carlo of the strategy's daily net returns (section 8.3).

Three generators, each producing ``n_paths`` paths of ``horizon`` days:

A. ``block``     circular block bootstrap with random block lengths in
                 [block_min, block_max] days (keeps short-range dependence and
                 volatility clustering of the realised return stream);
B. ``student_t`` i.i.d. location-scale Student-t (equivalently a Tsallis
                 q-Gaussian with q = 1 + 2 / (nu + 1)) fitted to the active
                 (non-zero) days, mixed with the empirical share of flat days;
C. ``levy``      exponentially tempered alpha-stable innovations
                 (alpha in [1.5, 2)), variance-matched to the sample.  Pure
                 stable laws have infinite variance and are used here for
                 tail stress only - never inside GARCH or Kelly.

Audit fixes: wealth is compounded in log space with simple returns clamped
above -0.99 (no sign flip of wealth), and recovery time is the LONGEST
CONTIGUOUS underwater streak per path, not the count of underwater days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

RETURN_FLOOR = -0.99


@dataclass(frozen=True)
class MonteCarloConfig:
    n_paths: int = 10_000
    horizon: int = 252
    block_min: int = 5
    block_max: int = 10
    levy_alpha: float = 1.7
    levy_tempering: float = 0.10     # exp(-lambda |x|) tilt in stable scale units
    ruin_level: float = 0.50         # wealth fraction counted as ruin / forced stop
    seed: int = 2024

    def __post_init__(self) -> None:
        if not 1.5 <= self.levy_alpha < 2.0:
            raise ValueError("levy_alpha must lie in [1.5, 2.0)")
        if self.n_paths < 1 or self.horizon < 2:
            raise ValueError("n_paths and horizon must be positive")


# ------------------------------------------------------------- generators


def simulate_block_bootstrap(r: np.ndarray, cfg: MonteCarloConfig, rng: np.random.Generator) -> np.ndarray:
    T, H = len(r), cfg.horizon
    nb = H // cfg.block_min + 1
    out = np.empty((cfg.n_paths, H))
    t = np.arange(H)
    for lo in range(0, cfg.n_paths, 2000):
        n = min(2000, cfg.n_paths - lo)
        lens = rng.integers(cfg.block_min, cfg.block_max + 1, size=(n, nb))
        ends = np.cumsum(lens, axis=1)
        starts = rng.integers(0, T, size=(n, nb))
        blk = (ends[:, :, None] <= t[None, None, :]).sum(axis=1)          # block id per day
        blk_start = np.concatenate([np.zeros((n, 1), dtype=int), ends[:, :-1]], axis=1)
        offset = t[None, :] - np.take_along_axis(blk_start, blk, axis=1)
        idx = (np.take_along_axis(starts, blk, axis=1) + offset) % T
        out[lo:lo + n] = r[idx]
    return out


def _active_split(r: np.ndarray) -> tuple[np.ndarray, float]:
    active = r[np.abs(r) > 1e-12]
    return active, len(active) / len(r)


def simulate_student_t(r: np.ndarray, cfg: MonteCarloConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    active, p_act = _active_split(r)
    if len(active) < 30:
        raise ValueError("fewer than 30 active days: parametric Monte Carlo is not meaningful")
    nu, loc, scale = student_t.fit(active)
    nu = float(np.clip(nu, 2.1, 200.0))        # keep the variance finite
    x = loc + scale * rng.standard_t(nu, size=(cfg.n_paths, cfg.horizon))
    mask = rng.random((cfg.n_paths, cfg.horizon)) < p_act
    return np.where(mask, x, 0.0), {"nu": nu, "tsallis_q": 1.0 + 2.0 / (nu + 1.0), "p_active": p_act}


def tempered_stable(alpha: float, lam: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Symmetric alpha-stable draws (Chambers-Mallows-Stuck) exponentially
    tempered by rejection with acceptance probability exp(-lam |x|)."""
    out = np.empty(size)
    filled = 0
    while filled < size:
        m = int((size - filled) * 1.6) + 1024
        u = rng.uniform(-np.pi / 2, np.pi / 2, m)
        w = rng.exponential(1.0, m)
        x = np.sin(alpha * u) / np.cos(u) ** (1 / alpha) * (np.cos(u - alpha * u) / w) ** ((1 - alpha) / alpha)
        x = x[rng.random(m) < np.exp(-lam * np.abs(x))]
        k = min(len(x), size - filled)
        out[filled:filled + k] = x[:k]
        filled += k
    return out


def simulate_tempered_levy(r: np.ndarray, cfg: MonteCarloConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    active, p_act = _active_split(r)
    if len(active) < 30:
        raise ValueError("fewer than 30 active days: parametric Monte Carlo is not meaningful")
    ref = tempered_stable(cfg.levy_alpha, cfg.levy_tempering, 400_000, rng)
    unit_sd = float(ref.std())
    x = tempered_stable(cfg.levy_alpha, cfg.levy_tempering, cfg.n_paths * cfg.horizon, rng)
    x = active.mean() + active.std(ddof=1) * x.reshape(cfg.n_paths, cfg.horizon) / unit_sd
    mask = rng.random((cfg.n_paths, cfg.horizon)) < p_act
    kurt = float(np.mean((ref / unit_sd) ** 4))
    return np.where(mask, x, 0.0), {"alpha": cfg.levy_alpha, "tempering": cfg.levy_tempering,
                                    "innovation_kurtosis": kurt, "p_active": p_act}


# -------------------------------------------------------------- statistics


def longest_true_run(mask: np.ndarray) -> np.ndarray:
    """Length of the longest contiguous run of True in each row."""
    c = np.cumsum(mask, axis=1)
    reset = np.maximum.accumulate(np.where(~mask, c, 0), axis=1)
    return (c - reset).max(axis=1)


def path_statistics(sim_returns: np.ndarray, ruin_level: float = 0.5) -> dict[str, float]:
    r = np.maximum(np.asarray(sim_returns, dtype=float), RETURN_FLOOR)      # wealth sign-flip guard
    wealth = np.exp(np.cumsum(np.log1p(r), axis=1))
    w = np.concatenate([np.ones((len(wealth), 1)), wealth], axis=1)
    peak = np.maximum.accumulate(w, axis=1)
    dd = 1.0 - w / peak
    mdd = dd.max(axis=1)
    recovery = longest_true_run(dd > 1e-12)
    return {
        "p_dd_10": float(np.mean(mdd >= 0.10)),
        "p_dd_20": float(np.mean(mdd >= 0.20)),
        "p_dd_30": float(np.mean(mdd >= 0.30)),
        "p_ruin": float(np.mean(w.min(axis=1) <= ruin_level)),
        "median_terminal_wealth": float(np.median(wealth[:, -1])),
        "p05_terminal_wealth": float(np.quantile(wealth[:, -1], 0.05)),
        "worst_99_dd": float(np.quantile(mdd, 0.99)),
        "median_recovery_days": float(np.median(recovery)),
        "p_loss": float(np.mean(wealth[:, -1] < 1.0)),
    }


def run_monte_carlo(daily_returns: np.ndarray, cfg: MonteCarloConfig = MonteCarloConfig()) -> dict[str, dict]:
    """All three generators.  A generator that cannot be fitted (too few
    active days) reports ``{"error": ...}`` instead of numbers."""
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 60:
        return {"error": {"error": f"only {len(r)} daily returns - Monte Carlo skipped"}}
    out: dict[str, dict] = {}
    rng = np.random.default_rng(cfg.seed)
    out["block"] = path_statistics(simulate_block_bootstrap(r, cfg, rng), cfg.ruin_level)
    for name, fn in (("student_t", simulate_student_t), ("levy", simulate_tempered_levy)):
        try:
            sim, info = fn(r, cfg, rng)
            out[name] = path_statistics(sim, cfg.ruin_level) | {f"fit_{k}": v for k, v in info.items()}
        except ValueError as exc:
            out[name] = {"error": str(exc)}
    for v in out.values():
        v.setdefault("n_paths", cfg.n_paths)
        v.setdefault("horizon_days", cfg.horizon)
    return out
