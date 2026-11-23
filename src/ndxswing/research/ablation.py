"""Sequential Phase 2 -> Phase 3 ablation (section 8.1).

Start from the simplest baseline (daily EMA20/50 direction, 20-hour Donchian
stop-limit entry, 1.5 ATR stop, 5-day time exit, realised-vol targeted
sizing) and add one module at a time in ``MODULE_ORDER``.  Each candidate is
judged by a complete purged walk-forward; it is retained only if the stitched
out-of-sample result improves:

    Sharpe lift >= min_sharpe_lift  AND  Calmar lift >= 0  AND  trades >= floor

Profit factor and ES_0.99 lifts are reported alongside.  Special cases:

* regime gate - the 3-state HMM is benchmarked against a 2-state
  volatility-switching model and a deterministic RV20 percentile filter; the
  SIMPLEST variant within ``parsimony_margin`` Sharpe of the best is the one
  carried forward;
* Hurst gate - a challenger evaluated late: first the specification gate
  (point estimate and bootstrap lower bound), then a point-estimate-only
  version; neither sits in the baseline;
* ARMA directional filter - a challenger evaluated only if the Pesaran-Timmermann test of
  the forecast sign is significant (p < 0.05) on non-overlapping horizons; when the
  forecast never changes sign the statistic is 0/0 and the test is reported as
  undefined (still blocked: fail closed);
* modules whose input data is missing are reported as skipped, never faked.

Because retention decisions look at OOS results, the OOS curve of every
variant is one of the K configurations compared: their average pairwise
correlation sets the effective number of trials in the Deflated Sharpe Ratio
and their returns form the CSCV matrix.  This holds even when no parameter is
tuned - choosing among module variants is still selection.  ``holdout_start`` reserves a final period
that no ablation decision ever sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..analytics.stats import compute_metrics
from ..config import MODULE_ORDER, ModuleFlags, StrategyParams
from ..features.garch import pesaran_timmermann
from .walkforward import BacktestContext, PurgedWalkForwardRunner, WalkForwardConfig, WalkForwardResult

REGIME_VARIANTS = ("rv20", "hmm2", "hmm3")   # simplest first
REGIME_COLUMNS = {"rv20": "rv20_pct", "hmm2": "p2_stress", "hmm3": "p_trend"}
KEYS = ("sharpe", "calmar", "profit_factor", "es_99", "max_drawdown", "n_trades", "net_pnl")
#: the Pesaran-Timmermann statistic is 0/0 when forecast (or realised) signs never vary
PT_UNDEFINED = "undefined (degenerate sign variance)"
PT_TOO_FEW = "undefined (fewer than 30 observations)"


@dataclass(frozen=True)
class AblationConfig:
    min_sharpe_lift: float = 0.05
    min_trades: int = 200
    parsimony_margin: float = 0.05
    pt_alpha: float = 0.05
    candidates: tuple[str, ...] = MODULE_ORDER
    holdout_start: str | None = None


@dataclass
class AblationResult:
    table: pd.DataFrame
    final_flags: ModuleFlags
    baseline: WalkForwardResult
    final: WalkForwardResult
    n_variants: int
    pt_test: dict
    holdout: dict | None
    trade_floor: int
    variant_oos: list   # stitched OOS result of every variant examined (trials for DSR / PBO)
    max_tuned: int = 0  # largest number of parameters the optimiser moved in any variant

    def to_markdown(self) -> str:
        t = self.table
        L = ["# Sequential ablation (stitched walk-forward OOS)", "",
             f"Retention floor on trade count: {self.trade_floor}. Variants evaluated: {self.n_variants}.", "",
             "| Step | Module | Variant | Status | OOS Sharpe | d Sharpe | Calmar | d Calmar | PF | d PF | ES99 | d ES99 | "
             "Max DD | Trades |",
             "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

        def f(v: object, pct: bool = False) -> str:
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "n/a"
            return f"{100 * v:.2f}%" if pct else f"{v:.3f}"

        for r in t.to_dict(orient="records"):
            L.append(
                f"| {r['step']} | {r['module']} | {r['variant']} | {r['status']} | {f(r['sharpe'])} | {f(r['d_sharpe'])} | "
                f"{f(r['calmar'])} | {f(r['d_calmar'])} | {f(r['profit_factor'])} | {f(r['d_profit_factor'])} | "
                f"{f(r['es_99'], True)} | {f(r['d_es_99'], True)} | {f(r['max_drawdown'], True)} | "
                f"{'n/a' if pd.isna(r['n_trades']) else int(r['n_trades'])} |"
            )
        L += ["", f"**Retained modules:** {', '.join(self.final_flags.enabled()) or 'none (baseline only)'}"
              + (f" (regime model: {self.final_flags.regime_model})" if self.final_flags.regime_gate else ""), ""]
        if self.pt_test:
            p = self.pt_test
            outcome = (f"statistic {p['stat']:.3f}, p-value {p['pvalue']:.4f}" if np.isfinite(p["pvalue"])
                       else f"{p.get('note', PT_UNDEFINED)}; forecast sign positive on "
                            f"{100 * p.get('share_forecast_up', float('nan')):.1f}% of the points")
            L += [f"Pesaran-Timmermann test of the ARMA forecast sign (H={p['horizon']}d, n={p['n']}): {outcome}.", ""]
        if self.holdout:
            h = self.holdout
            L += [f"**Untouched holdout from {h['start']}** (never used for a retention decision): "
                  f"Sharpe {f(h['sharpe'])}, profit factor {f(h['profit_factor'])}, "
                  f"max drawdown {f(h['max_drawdown'], True)}, trades {h['n_trades']}, net P&L {h['net_pnl']:.2f}.", ""]
        return "\n".join(L)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(out / "ablation.csv", index=False)
        (out / "ablation.md").write_text(self.to_markdown())
        return out / "ablation.md"


class SequentialAblationEngine:
    def __init__(self, ctx: BacktestContext, wf_config: WalkForwardConfig = WalkForwardConfig(),
                 config: AblationConfig = AblationConfig(), base_params: StrategyParams = StrategyParams()) -> None:
        self.ctx, self.wf_cfg, self.cfg, self.base = ctx, wf_config, config, base_params
        self._end = None if config.holdout_start is None else (
            pd.Timestamp(config.holdout_start) - pd.Timedelta(days=1)
        )

    def _walk_forward(self, flags: ModuleFlags) -> tuple[WalkForwardResult, dict]:
        runner = PurgedWalkForwardRunner(self.ctx, flags, self.wf_cfg, self.base, end=self._end)
        self._max_tuned = max(getattr(self, "_max_tuned", 0), len(runner.space))
        wf = runner.run()
        return wf, compute_metrics(wf.oos)

    def _pt_test(self, wf: WalkForwardResult) -> dict:
        H = self.base.horizon_days
        d = self.ctx.features.daily
        if f"mu_{H}" not in d:
            return {}
        fwd = d["r"].rolling(H).sum().shift(-H)          # realised return over t+1..t+H
        df = pd.DataFrame({"f": d[f"mu_{H}"], "a": fwd}).dropna()
        df = df[(df.index >= wf.folds[0].oos_start) & (df.index <= wf.folds[-1].oos_end)].iloc[::H]
        stat, p = pesaran_timmermann(df["a"].to_numpy(), df["f"].to_numpy())
        out = {"horizon": H, "n": int(len(df)), "stat": stat, "pvalue": p,
               "share_forecast_up": float((df["f"] > 0).mean()) if len(df) else float("nan")}
        if not np.isfinite(p):
            out["note"] = PT_TOO_FEW if len(df) < 30 else PT_UNDEFINED
        return out

    def run(self, verbose: bool = False) -> AblationResult:
        cfg = self.cfg
        avail = self.ctx.backtester().engine.available_modules()
        rows: list[dict] = []

        def log(msg: str) -> None:
            if verbose:
                print(f"[ablation] {msg}", flush=True)

        def row(step: int, module: str, variant: str, status: str, m: dict | None, ref: dict | None) -> None:
            r = {"step": step, "module": module, "variant": variant, "status": status}
            for k in KEYS:
                r[k] = np.nan if m is None else m[k]
                if k in ("sharpe", "calmar", "profit_factor", "es_99"):
                    r[f"d_{k}"] = np.nan if m is None or ref is None else m[k] - ref[k]
            rows.append(r)

        flags = ModuleFlags.baseline()
        log("baseline")
        base_wf, cur = self._walk_forward(flags)
        cur_wf = base_wf
        n_variants = 1
        variant_oos = [base_wf.oos]
        row(0, "baseline", "-", "BASELINE", cur, None)
        floor = int(min(cfg.min_trades, max(1, 0.5 * cur["n_trades"])))
        pt: dict = {}

        def accept(m: dict, ref: dict) -> bool:
            return (m["sharpe"] - ref["sharpe"] >= cfg.min_sharpe_lift and m["calmar"] - ref["calmar"] >= 0.0
                    and m["n_trades"] >= floor)

        for step, module in enumerate(cfg.candidates, start=1):
            if module in avail and not avail[module]:
                row(step, module, "-", "SKIPPED (input data not available)", None, None)
                continue
            if module == "arma_z_filter":
                pt = self._pt_test(cur_wf)
                if not pt or not (pt["pvalue"] < cfg.pt_alpha):   # NaN compares False: fail closed
                    why = pt.get("note", "not significant") if pt else "not available"
                    row(step, module, "-", f"BLOCKED (Pesaran-Timmermann {why})", None, None)
                    continue
            if module == "regime_gate":
                tried = []
                for variant in REGIME_VARIANTS:
                    if REGIME_COLUMNS[variant] not in self.ctx.features.daily:
                        continue
                    log(f"{module} [{variant}]")
                    wf, m = self._walk_forward(flags.with_(regime_gate=True, regime_model=variant))
                    n_variants += 1
                    variant_oos.append(wf.oos)
                    tried.append((variant, wf, m))
                best = max(m["sharpe"] for _, _, m in tried)
                pick = next(v for v, _, m in tried if m["sharpe"] >= best - cfg.parsimony_margin)
                for variant, wf, m in tried:
                    ok = variant == pick and accept(m, cur)
                    status = "RETAINED" if ok else ("rejected" if variant == pick else "rejected (parsimony benchmark)")
                    row(step, module, variant, status, m, cur)
                    if ok:
                        new = (flags.with_(regime_gate=True, regime_model=variant), wf, m)
                if any(r["status"] == "RETAINED" and r["step"] == step for r in rows):
                    flags, cur_wf, cur = new
                continue
            if module == "hurst_gate":
                # spec gate first (H >= min and bootstrap LCB > 0.50); if that
                # is not retained, the relaxed point-estimate-only version
                for mode in ("lcb", "point"):
                    log(f"{module} [{mode}]")
                    cand = flags.with_(hurst_gate=True, hurst_mode=mode)
                    wf, m = self._walk_forward(cand)
                    n_variants += 1
                    variant_oos.append(wf.oos)
                    if accept(m, cur):
                        row(step, module, mode, "RETAINED", m, cur)
                        flags, cur_wf, cur = cand, wf, m
                        break
                    why = "trade starvation" if m["n_trades"] < floor else "no OOS net gain"
                    row(step, module, mode, f"rejected ({why})", m, cur)
                continue
            log(module)
            cand = flags.with_(**{module: True})
            wf, m = self._walk_forward(cand)
            n_variants += 1
            variant_oos.append(wf.oos)
            if accept(m, cur):
                row(step, module, "-", "RETAINED", m, cur)
                flags, cur_wf, cur = cand, wf, m
            else:
                why = "trade starvation" if m["n_trades"] < floor else "no OOS net gain"
                row(step, module, "-", f"rejected ({why})", m, cur)

        holdout = None
        final_wf = cur_wf
        if cfg.holdout_start is not None:
            log("final configuration through the holdout")
            final_wf = PurgedWalkForwardRunner(self.ctx, flags, self.wf_cfg, self.base).run()
            hs = pd.Timestamp(cfg.holdout_start)
            if final_wf.oos.equity.index[-1] >= hs:
                hm = compute_metrics(final_wf.oos.slice(hs, None))
                holdout = {"start": str(hs.date()), **{k: hm[k] for k in KEYS}}
        return AblationResult(
            table=pd.DataFrame(rows), final_flags=flags, baseline=base_wf, final=final_wf, n_variants=n_variants,
            pt_test=pt, holdout=holdout, trade_floor=floor, variant_oos=variant_oos,
            max_tuned=self._max_tuned,
        )
