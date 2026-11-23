"""Purged rolling walk-forward with neighbourhood-penalised Optuna search (8.2).

Cycle: 36 months train -> 12 months validation -> 6 months untouched OOS,
shifted by 3 months (a 6-month validation window holds only a handful of
trades, which makes parameter selection a fit to small-sample luck), with ``purge_days + embargo_days`` sessions removed
between consecutive windows so that a 5-day holding period cannot straddle a
boundary.

* The statistical layer (HMM, Hurst, GARCH, EVT) is re-estimated on rolling
  windows inside the causal feature table, so it never sees validation or OOS
  data of the date it is used on.
* By default only the non-structural thresholds of the enabled modules are
  tuned (at most three); Donchian window, stop multiple and holding period
  stay at their canonical values.  With nothing to tune the fold is a pure
  fixed-prior out-of-sample test and Optuna is not run at all.
* Hyper-parameters are chosen per fold on the validation window only
  (objective J below); the trial is simulated from the start of the training
  window so that circuit-breaker and Kelly state are warm, but only the
  validation slice is scored.
* The chosen parameters are then frozen and applied to the fold's OOS window.

    J = 0.40 Sharpe_val + 0.30 Calmar_val - 0.20 MDD_val / 0.10
        - 0.10 Turnover_val / turnover_ref - lambda_stab * NeighbourhoodStd(Sharpe_val)

``NeighbourhoodStd`` is the standard deviation of the validation Sharpe over
the candidate and its +/- one-step neighbours in every tuned dimension; it is
evaluated explicitly for the ``top_k`` candidates of the TPE search, so an
isolated spike (good at H=0.57, bad at 0.56 and 0.58) is rejected.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd

from ..analytics.stats import cscv_pbo, max_drawdown, returns_matrix, sharpe_per_period
from ..analytics.tearsheet import TearsheetReporter
from ..analytics.montecarlo import MonteCarloConfig
from ..backtest.engine import BacktestResult, EventDrivenBacktester
from ..config import BacktestConfig, ModuleFlags, StrategyParams
from ..features.pipeline import FeatureSet
from ..signal.engine import SignalEngine

optuna.logging.set_verbosity(optuna.logging.WARNING)

#: name -> (kind, low, high, neighbour step, module that must be enabled or None)
SEARCH_SPACE: dict[str, tuple[str, float, float, float, str | None]] = {
    "donchian_window": ("int", 16, 40, 2, None),
    "atr_stop_mult": ("float", 1.2, 2.2, 0.1, None),
    "max_hold_days": ("int", 1, 5, 1, None),
    "p_trend_min": ("float", 0.60, 0.80, 0.02, "regime_gate"),
    "vpin_block_pct": ("float", 0.90, 0.99, 0.01, "vpin_filter"),
    "kelly_fraction": ("float", 0.10, 0.25, 0.02, "kelly_cap"),
    "hurst_min": ("float", 0.53, 0.60, 0.01, "hurst_gate"),
    "z_hurdle": ("float", 0.25, 0.75, 0.05, "arma_z_filter"),
}
#: entry / exit geometry: frozen at canonical values unless tune="all"
STRUCTURAL = ("donchian_window", "atr_stop_mult", "max_hold_days")


@dataclass
class BacktestContext:
    """Everything needed to build backtesters for a data set."""

    features: FeatureSet
    config: BacktestConfig = field(default_factory=BacktestConfig)
    fx: pd.Series | None = None
    rate: pd.Series | None = None
    events: pd.DataFrame | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def backtester(self, cost_stress: float = 1.0) -> EventDrivenBacktester:
        if cost_stress not in self._cache:
            cfg = self.config
            if cost_stress != 1.0:
                from dataclasses import replace

                cfg = replace(cfg, costs=cfg.costs.stressed(cost_stress))
            eng = SignalEngine(self.features, cfg.costs, self.events, self.rate)
            self._cache[cost_stress] = EventDrivenBacktester(eng, cfg, self.fx)
        return self._cache[cost_stress]

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.features.sessions


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int = 36
    val_months: int = 12
    oos_months: int = 6
    step_months: int = 3
    purge_days: int = 5
    embargo_days: int = 5
    n_trials: int = 100
    top_k: int = 8
    lambda_stab: float = 0.25
    min_val_trades: int = 3
    turnover_ref: float = 100.0
    calmar_cap: float = 3.0   # 6-month Calmar is explosive; keep it on the scale of the Sharpe term
    warm_start: bool = True
    seed: int = 13
    n_jobs: int = 1
    cost_stress: float = 1.5
    #: "thresholds": tune only module thresholds, at most ``max_tuned`` of them
    #: "none":       no tuning at all (fixed canonical priors)
    #: "all":        also tune Donchian window, stop multiple and holding period
    tune: str = "thresholds"
    max_tuned: int = 3

    def __post_init__(self) -> None:
        if self.tune not in ("thresholds", "none", "all"):
            raise ValueError("tune must be 'thresholds', 'none' or 'all'")


@dataclass(frozen=True)
class Fold:
    k: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


def make_folds(sessions: pd.DatetimeIndex, cfg: WalkForwardConfig, start: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None) -> list[Fold]:
    """Rolling folds on session dates.  Consecutive windows are separated by
    ``purge_days + embargo_days`` sessions that belong to neither."""
    s = sessions if start is None else sessions[sessions >= pd.Timestamp(start)]
    s = s if end is None else s[s <= pd.Timestamp(end)]
    gap = cfg.purge_days + cfg.embargo_days
    folds: list[Fold] = []
    k = 0
    while len(s):
        t0 = s[0] + pd.DateOffset(months=k * cfg.step_months)
        i_tr0 = int(s.searchsorted(t0))
        i_tr1 = int(s.searchsorted(t0 + pd.DateOffset(months=cfg.train_months))) - 1
        i_v0 = i_tr1 + 1 + gap
        if i_v0 >= len(s):
            break
        i_v1 = int(s.searchsorted(s[i_v0] + pd.DateOffset(months=cfg.val_months))) - 1
        i_o0 = i_v1 + 1 + gap
        if i_o0 >= len(s):
            break
        i_o1 = int(s.searchsorted(s[i_o0] + pd.DateOffset(months=cfg.oos_months))) - 1
        if i_o1 >= len(s) or i_o1 <= i_o0:
            break
        if i_o1 - i_o0 < 0.8 * 21 * cfg.oos_months:   # incomplete final window
            break
        folds.append(Fold(k, s[i_tr0], s[i_tr1], s[i_v0], s[i_v1], s[i_o0], s[i_o1]))
        k += 1
    return folds


# --------------------------------------------------------------------------- #
# Search space helpers
# --------------------------------------------------------------------------- #


def active_space(flags: ModuleFlags, tune: str = "all", max_tuned: int = 3) -> dict[str, tuple]:
    """Parameters Optuna may move for this module set.  ``p_trend_min`` only
    exists for the 3-state model; the parsimony benchmarks run on fixed priors."""
    if tune == "none":
        return {}
    space = {k: v for k, v in SEARCH_SPACE.items() if v[4] is None or getattr(flags, v[4])}
    if flags.regime_model != "hmm3":
        space.pop("p_trend_min", None)
    if tune == "thresholds":
        keep = [k for k in space if k not in STRUCTURAL][:max_tuned]
        space = {k: space[k] for k in keep}
    return space


def apply_overrides(base: StrategyParams, theta: dict) -> StrategyParams:
    kw = dict(theta)
    if "vpin_block_pct" in kw:
        kw["vpin_halve_pct"] = min(base.vpin_halve_pct, kw["vpin_block_pct"])
    return base.with_(**kw)


def suggest(trial: optuna.Trial, space: dict) -> dict:
    theta = {}
    for name, (kind, lo, hi, _, _) in space.items():
        theta[name] = trial.suggest_int(name, int(lo), int(hi)) if kind == "int" else trial.suggest_float(name, lo, hi)
    return theta


def neighbours(theta: dict, space: dict) -> list[dict]:
    """One-step perturbations in each tuned dimension, clipped to the range."""
    out = []
    for name, (kind, lo, hi, step, _) in space.items():
        for sgn in (-1, 1):
            v = theta[name] + sgn * step
            v = int(min(max(v, lo), hi)) if kind == "int" else float(min(max(v, lo), hi))
            if v != theta[name]:
                out.append({**theta, name: v})
    return out


def random_configs(space: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({name: (int(rng.integers(int(lo), int(hi) + 1)) if kind == "int" else float(rng.uniform(lo, hi)))
                    for name, (kind, lo, hi, _, _) in space.items()})
    return out


def objective_components(val: BacktestResult, cfg: WalkForwardConfig) -> dict[str, float]:
    r = val.daily_returns().to_numpy()
    n = len(r)
    years = n / 252.0
    eq = val.equity.to_numpy()
    sr_pp = sharpe_per_period(r)
    sharpe = sr_pp * np.sqrt(252.0)
    mdd = max_drawdown(eq, val.initial_equity) if n else 0.0
    growth = eq[-1] / val.initial_equity if n else 1.0
    cagr = growth ** (1.0 / years) - 1.0 if n and growth > 0 else -1.0
    calmar = float(np.clip(cagr / mdd, -cfg.calmar_cap, cfg.calmar_cap)) if mdd > 1e-9 else 0.0
    t = val.real_trades()
    turnover = float(t["turnover"].sum()) / float(np.mean(eq)) / years if n and len(t) else 0.0
    j = 0.40 * sharpe + 0.30 * calmar - 0.20 * (mdd / 0.10) - 0.10 * (turnover / cfg.turnover_ref)
    if len(t) < cfg.min_val_trades:
        j = -1.0 - (cfg.min_val_trades - len(t)) / cfg.min_val_trades   # trade starvation is not a result
    return {"j": float(j), "sharpe": float(sharpe), "sharpe_pp": float(sr_pp), "calmar": calmar, "mdd": float(mdd),
            "turnover": turnover, "n_trades": int(len(t)), "net_pnl": float(eq[-1] - val.initial_equity) if n else 0.0}


# --------------------------------------------------------------------------- #
# Per-fold optimisation (module-level so it can run in forked workers)
# --------------------------------------------------------------------------- #

_STATE: dict = {}


def _evaluate(theta: dict, fold: Fold) -> dict[str, float]:
    bt, flags, cfg, base = _STATE["bt"], _STATE["flags"], _STATE["cfg"], _STATE["base"]
    params = apply_overrides(base, theta)
    run_start = fold.train_start if cfg.warm_start else fold.val_start
    res = bt.run(params, flags, start=run_start, end=fold.val_end)
    return objective_components(res.slice(fold.val_start, fold.val_end), cfg)


def _optimise_fold(k: int) -> dict:
    cfg, flags = _STATE["cfg"], _STATE["flags"]
    fold: Fold = _STATE["folds"][k]
    space = active_space(flags, cfg.tune, cfg.max_tuned)
    if not space:   # fixed priors: nothing is selected, the fold is a plain out-of-sample test
        comp = _evaluate({}, fold)
        return {"theta": {}, "j_raw": comp["j"], "j_pen": comp["j"], "nb_std": 0.0, "val": comp,
                "trial_sharpes_pp": [comp["sharpe_pp"]], "n_evaluations": 1, "fold": k}
    evaluated: list[tuple[dict, dict]] = []

    def objective(trial: optuna.Trial) -> float:
        theta = suggest(trial, space)
        comp = _evaluate(theta, fold)
        evaluated.append((theta, comp))
        return comp["j"]

    sampler = optuna.samplers.TPESampler(seed=cfg.seed + 1000 * k,
                                         n_startup_trials=min(cfg.n_trials, max(10, cfg.n_trials // 4)))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

    order = sorted(range(len(evaluated)), key=lambda i: -evaluated[i][1]["j"])
    best = None
    for i in order[: cfg.top_k]:
        theta, comp = evaluated[i]
        nb = [_evaluate(t, fold)["sharpe"] for t in neighbours(theta, space)]
        nb_std = float(np.std([comp["sharpe"]] + nb)) if nb else 0.0
        j_pen = comp["j"] - cfg.lambda_stab * nb_std
        if best is None or j_pen > best["j_pen"]:
            best = {"theta": theta, "j_raw": comp["j"], "j_pen": j_pen, "nb_std": nb_std, "val": comp}
    best["trial_sharpes_pp"] = [c["sharpe_pp"] for _, c in evaluated]
    best["n_evaluations"] = len(evaluated) + sum(len(neighbours(evaluated[i][0], space)) for i in order[: cfg.top_k])
    best["fold"] = k
    return best


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


@dataclass
class WalkForwardResult:
    flags: ModuleFlags
    config: WalkForwardConfig
    folds: list[Fold]
    fold_params: list[StrategyParams]
    fold_table: pd.DataFrame
    in_sample: BacktestResult
    validation: BacktestResult
    oos: BacktestResult
    oos_stress: BacktestResult
    oos_folds: list[BacktestResult]
    trial_sharpes_val: np.ndarray       # per-period validation Sharpe of every Optuna trial
    n_evaluations: int

    def oos_schedule(self) -> list[tuple[pd.Timestamp, StrategyParams]]:
        return [(f.oos_start, p) for f, p in zip(self.folds, self.fold_params)]


class PurgedWalkForwardRunner:
    def __init__(self, ctx: BacktestContext, flags: ModuleFlags, config: WalkForwardConfig = WalkForwardConfig(),
                 base_params: StrategyParams = StrategyParams(), start: pd.Timestamp | None = None,
                 end: pd.Timestamp | None = None) -> None:
        self.ctx, self.flags, self.cfg, self.base = ctx, flags, config, base_params
        self.folds = make_folds(ctx.sessions, config, start, end)
        if not self.folds:
            raise ValueError("not enough history for a single train/validation/OOS fold")

    def run(self, verbose: bool = False) -> WalkForwardResult:
        cfg, folds = self.cfg, self.folds
        bt = self.ctx.backtester()
        _STATE.update(bt=bt, flags=self.flags, cfg=cfg, base=self.base, folds=folds)
        ks = list(range(len(folds)))
        if cfg.n_jobs > 1 and len(ks) > 1:
            with mp.get_context("fork").Pool(min(cfg.n_jobs, len(ks))) as pool:
                best = pool.map(_optimise_fold, ks, chunksize=1)
        else:
            best = []
            for k in ks:
                best.append(_optimise_fold(k))
                if verbose:
                    b = best[-1]
                    print(f"[wf] fold {k + 1}/{len(ks)} J={b['j_pen']:.3f} val_sharpe={b['val']['sharpe']:.2f} "
                          f"theta={b['theta']}", flush=True)
        params = [apply_overrides(self.base, b["theta"]) for b in best]

        rows, oos_folds = [], []
        for f, b, p in zip(folds, best, params):
            r = bt.run(p, self.flags, start=f.oos_start, end=f.oos_end)
            oc = objective_components(r, cfg)
            oos_folds.append(r)
            rows.append({
                "fold": f.k, "train_start": f.train_start, "train_end": f.train_end, "val_start": f.val_start,
                "val_end": f.val_end, "oos_start": f.oos_start, "oos_end": f.oos_end, "val_j": b["j_pen"],
                "val_j_raw": b["j_raw"], "val_nb_std": b["nb_std"], "val_sharpe": b["val"]["sharpe"],
                "val_trades": b["val"]["n_trades"], "oos_sharpe": oc["sharpe"], "oos_net_pnl": oc["net_pnl"],
                "oos_mdd": oc["mdd"], "oos_trades": oc["n_trades"], **{f"p_{k}": v for k, v in b["theta"].items()},
            })
        table = pd.DataFrame(rows)

        nxt = self.ctx.sessions
        is_sched = [(folds[0].train_start, params[0])] + [
            (nxt[nxt.searchsorted(folds[k - 1].train_end, side="right")], params[k]) for k in range(1, len(folds))
        ]
        is_res = bt.run(schedule=is_sched, flags=self.flags, start=folds[0].train_start, end=folds[-1].train_end)
        val_sched = [(f.val_start, p) for f, p in zip(folds, params)]
        val_res = bt.run(schedule=val_sched, flags=self.flags, start=folds[0].val_start, end=folds[-1].val_end)
        oos_sched = [(f.oos_start, p) for f, p in zip(folds, params)]
        oos_res = bt.run(schedule=oos_sched, flags=self.flags, start=folds[0].oos_start, end=folds[-1].oos_end)
        stress = self.ctx.backtester(cfg.cost_stress).run(
            schedule=oos_sched, flags=self.flags, start=folds[0].oos_start, end=folds[-1].oos_end
        )
        return WalkForwardResult(
            flags=self.flags, config=cfg, folds=folds, fold_params=params, fold_table=table, in_sample=is_res,
            validation=val_res, oos=oos_res, oos_stress=stress, oos_folds=oos_folds,
            trial_sharpes_val=np.array([s for b in best for s in b["trial_sharpes_pp"]]),
            n_evaluations=int(sum(b["n_evaluations"] for b in best)),
        )

    # ------------------------------------------------------------ overfitting
    @property
    def space(self) -> dict:
        return active_space(self.flags, self.cfg.tune, self.cfg.max_tuned)

    def config_trials(self, wf: WalkForwardResult, n_configs: int = 64, seed: int = 99) -> list[BacktestResult]:
        """Fixed-parameter runs of ``n_configs`` random draws from the search
        space over the stitched OOS span.  Their daily returns form the (T, N)
        matrix for CSCV-PBO and their Sharpe ratios the trial distribution of
        the Deflated Sharpe Ratio (same sample length as the tested series).
        Empty when nothing is tuned."""
        if not self.space:
            return []
        bt = self.ctx.backtester()
        lo, hi = wf.folds[0].oos_start, wf.folds[-1].oos_end
        return [bt.run(apply_overrides(self.base, th), self.flags, start=lo, end=hi)
                for th in random_configs(self.space, n_configs, seed)]

    def tearsheet(self, wf: WalkForwardResult, n_configs: int = 64, extra_trials: list[BacktestResult] | None = None,
                  mc_config: MonteCarloConfig = MonteCarloConfig(), meta: dict | None = None,
                  title: str = "NASDAQ-100 swing system") -> TearsheetReporter:
        """``extra_trials`` are OOS results of other configurations that were
        looked at before settling on this one (the ablation variants).  They
        join the random parameter draws as trials for DSR and PBO.  With no
        tuning and no other variants there was no selection at all: DSR then
        reduces to the probabilistic Sharpe ratio against zero and PBO is not
        applicable."""
        trials = self.config_trials(wf, n_configs) + list(extra_trials or [])
        pbo, trial_sr = None, None
        if len(trials) >= 2:
            M = returns_matrix(trials)
            active = np.abs(M).sum(axis=0) > 0
            pbo = cscv_pbo(M, 16) if active.sum() >= 2 and len(M) >= 64 else None
            trial_sr = np.array([sharpe_per_period(M[:, j]) for j in range(M.shape[1])])
        tuned = list(self.space)
        info = {
            "modules": ", ".join(wf.flags.enabled()) or "baseline only",
            "regime model": wf.flags.regime_model if wf.flags.regime_gate else "n/a",
            "walk-forward": f"{wf.config.train_months}m train / {wf.config.val_months}m val / "
                            f"{wf.config.oos_months}m OOS, step {wf.config.step_months}m, "
                            f"purge {wf.config.purge_days}d + embargo {wf.config.embargo_days}d",
            "folds": len(wf.folds),
            "tuned parameters": ", ".join(tuned) if tuned else "none (fixed canonical priors)",
            "Optuna trials per fold": wf.config.n_trials if tuned else 0,
            "backtests evaluated in search": wf.n_evaluations,
            "configurations compared (DSR / PBO trials)": len(trials) if len(trials) >= 2 else "1 (no selection)",
            "instrument": self.ctx.config.instrument.name,
            "initial equity": f"{self.ctx.config.initial_equity:,.0f} {self.ctx.config.account_ccy}",
        } | (meta or {})
        return TearsheetReporter(
            {"in_sample": wf.in_sample, "validation": wf.validation, "oos": wf.oos},
            oos_stress=wf.oos_stress, trial_sharpes=trial_sr, n_trials=len(trials) if len(trials) >= 2 else 1,
            pbo=pbo, fold_table=wf.fold_table, mc_config=mc_config, meta=info, title=title,
        )
