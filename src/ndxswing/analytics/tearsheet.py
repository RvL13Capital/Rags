"""Tearsheet: 15 required metrics, strictly segregated by data segment, plus
DSR, PBO, Monte Carlo tail probabilities and the OOS release gates (8.4).

Segments are never pooled.  A segment without data is reported as ``n/a`` -
in particular Paper and Live stay empty until real ledgers exist.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from ..data.synthetic import SYNTHETIC_SOURCE
from .montecarlo import MonteCarloConfig, run_monte_carlo
from .stats import compute_metrics, deflated_sharpe_ratio, probabilistic_sharpe_ratio, sharpe_per_period

SEGMENTS = ("in_sample", "validation", "oos", "paper", "live")
SEGMENT_TITLES = {"in_sample": "In-Sample", "validation": "Validation", "oos": "Out-of-Sample",
                  "paper": "Paper", "live": "Live"}

#: (row label, metric key, format)
ROWS = [
    ("1. CAGR (net)", "cagr", "pct"),
    ("2. Annualised volatility", "ann_vol", "pct"),
    ("3. Sharpe ratio (net, rf=0)", "sharpe", "f2"),
    ("4. Sortino ratio", "sortino", "f2"),
    ("5. Calmar ratio", "calmar", "f2"),
    ("6. Maximum drawdown", "max_drawdown", "pct"),
    ("7. Profit factor (net)", "profit_factor", "f2"),
    ("8. Win rate", "win_rate", "pct"),
    ("9. Avg win / avg loss", "avg_win_over_avg_loss", "f2"),
    ("10a. Net expectancy / trade (EUR)", "expectancy_eur", "eur"),
    ("10b. Net expectancy / trade (R)", "expectancy_r", "f3"),
    ("11. Independent trades", "n_trades", "int"),
    ("12. Annualised turnover (x equity)", "turnover_ann", "f1"),
    ("13a. Transaction costs (EUR)", "transaction_costs", "eur"),
    ("13b. Financing costs (EUR)", "financing_costs", "eur"),
    ("14a. ES 97.5% daily", "es_975", "pct"),
    ("14b. ES 99% daily", "es_99", "pct"),
    ("15a. MC P(DD >= 10%)", "mc_p_dd_10", "pct"),
    ("15b. MC P(DD >= 20%)", "mc_p_dd_20", "pct"),
    ("15c. MC P(DD >= 30%)", "mc_p_dd_30", "pct"),
    ("15d. MC P(ruin / forced stop)", "mc_p_ruin", "pct"),
    ("15e. MC median terminal wealth (x)", "mc_median_terminal_wealth", "f3"),
    ("15f. MC worst 99% drawdown", "mc_worst_99_dd", "pct"),
    ("15g. MC median contiguous recovery (days)", "mc_median_recovery_days", "f0"),
    ("Net P&L (EUR)", "net_pnl", "eur"),
    ("Sample length (years)", "years", "f2"),
]


def _fmt(v: object, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    if kind == "pct":
        return f"{100 * v:.2f}%"
    if kind == "eur":
        return f"{v:,.2f}"
    if kind == "int":
        return f"{int(v)}"
    return f"{v:.{int(kind[1])}f}"


def _json_safe(o: object) -> object:
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


class TearsheetReporter:
    def __init__(
        self,
        segments: dict[str, BacktestResult | None],
        *,
        oos_stress: BacktestResult | None = None,
        trial_sharpes: np.ndarray | None = None,
        n_trials: int | None = None,
        pbo: dict | None = None,
        fold_table: pd.DataFrame | None = None,
        mc_config: MonteCarloConfig = MonteCarloConfig(),
        meta: dict | None = None,
        title: str = "NASDAQ-100 swing system",
    ) -> None:
        unknown = set(segments) - set(SEGMENTS)
        if unknown:
            raise ValueError(f"unknown segments {sorted(unknown)}; allowed: {SEGMENTS}")
        self.segments = {s: segments.get(s) for s in SEGMENTS}
        self.oos_stress = oos_stress
        self.trial_sharpes = None if trial_sharpes is None else np.asarray(trial_sharpes, dtype=float)
        self.n_trials = n_trials
        self.pbo = pbo
        self.fold_table = fold_table
        self.mc_config = mc_config
        self.meta = dict(meta or {})
        self.title = title

    # ----------------------------------------------------------------- build
    def _segment_metrics(self, res: BacktestResult) -> dict:
        m = compute_metrics(res)
        mc = run_monte_carlo(res.daily_returns().to_numpy(), self.mc_config)
        m["monte_carlo"] = mc
        # headline MC row = the most adverse generator for each statistic
        usable = [v for v in mc.values() if "error" not in v]
        for key, worst in (("p_dd_10", max), ("p_dd_20", max), ("p_dd_30", max), ("p_ruin", max),
                           ("median_terminal_wealth", min), ("worst_99_dd", max), ("median_recovery_days", max)):
            m[f"mc_{key}"] = float(worst(v[key] for v in usable)) if usable else np.nan
        return m

    def build(self) -> dict:
        seg = {name: (self._segment_metrics(r) if r is not None and len(r.equity) else None)
               for name, r in self.segments.items()}
        sources = {r.stats.get("source", "unknown") for r in self.segments.values() if r is not None}
        synthetic = SYNTHETIC_SOURCE in sources
        report: dict = {"title": self.title, "segments": seg, "data_sources": sorted(sources),
                        "synthetic": synthetic, "meta": self.meta, "mc_config": asdict(self.mc_config)}

        oos = self.segments["oos"]
        dsr = None
        if oos is not None and self.trial_sharpes is not None and len(self.trial_sharpes) >= 2:
            dsr = deflated_sharpe_ratio(oos.daily_returns().to_numpy(), self.trial_sharpes, self.n_trials)
        elif oos is not None and self.n_trials == 1:
            # a single configuration was run and nothing was selected: the
            # benchmark Sharpe under the null is zero (DSR == PSR(0))
            r = oos.daily_returns().to_numpy()
            psr0 = probabilistic_sharpe_ratio(r, 0.0)
            dsr = {"dsr": psr0, "dsr_zero_mean_null": psr0, "sr_per_period": sharpe_per_period(r),
                   "sr0_per_period": 0.0, "sr0_zero_mean_null": 0.0, "n_trials": 1, "trial_sr_mean": float("nan"),
                   "trial_sr_std": float("nan"), "no_selection": True}
        report["dsr"] = dsr
        report["breakers"] = None if oos is None else {
            "pauses": oos.stats.get("n_pauses"), "deactivated_at": oos.stats.get("deactivated_at"),
            "true_peak": oos.stats.get("true_peak"), "max_dd_intrabar": oos.stats.get("max_dd_bar"),
        }
        report["pbo"] = self.pbo
        stress = None
        if self.oos_stress is not None:
            stress = compute_metrics(self.oos_stress)
            stress["cost_stress"] = self.oos_stress.stats.get("cost_stress")
        report["oos_cost_stress"] = stress
        folds = None
        if self.fold_table is not None and len(self.fold_table):
            ft = self.fold_table
            folds = {"n_folds": int(len(ft)), "profitable_share": float((ft["oos_net_pnl"] > 0).mean()),
                     "table": ft.to_dict(orient="records")}
        report["folds"] = folds
        report["gates"] = self._gates(seg.get("oos"), dsr, stress, folds)
        report["verdict"] = self._verdict(report["gates"], synthetic)
        return report

    @staticmethod
    def _gates(oos: dict | None, dsr: dict | None, stress: dict | None, folds: dict | None) -> list[dict]:
        def g(name: str, value: object, ok: bool | None, rule: str) -> dict:
            return {"gate": name, "value": value, "rule": rule,
                    "status": "NOT EVALUATED" if ok is None else ("PASS" if ok else "FAIL")}

        def val(d: dict | None, k: str) -> float | None:
            v = None if d is None else d.get(k)
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

        sh, pf, mdd, nt = (val(oos, k) for k in ("sharpe", "profit_factor", "max_drawdown", "n_trades"))
        p20 = val(oos, "mc_p_dd_20")
        d = val(dsr, "dsr")
        pbo_v = None
        return [
            g("OOS net Sharpe", sh, None if sh is None else sh > 0.80, "> 0.80"),
            g("Deflated Sharpe Ratio", d, None if d is None else d > 0.95, "> 0.95"),
            g("PBO (CSCV)", pbo_v, None, "< 0.25"),
            g("OOS net profit factor", pf, None if pf is None else pf > 1.20, "> 1.20"),
            g("OOS net P&L under 1.5x costs", val(stress, "net_pnl"),
              None if stress is None else stress["net_pnl"] > 0, "> 0"),
            g("OOS maximum drawdown", mdd, None if mdd is None else mdd < 0.10, "< 10%"),
            g("Monte Carlo P(DD >= 20%)", p20, None if p20 is None else p20 < 0.02, "< 2%"),
            g("Profitable walk-forward OOS folds", None if folds is None else folds["profitable_share"],
              None if folds is None else folds["profitable_share"] >= 0.70, ">= 70%"),
            g("Independent OOS trades", nt, None if nt is None else nt > 200, "> 200"),
        ]

    def _verdict(self, gates: list[dict], synthetic: bool) -> str:
        # PBO is filled in here because it lives outside the per-segment metrics
        for gate in gates:
            if gate["gate"].startswith("PBO"):
                if self.pbo is not None:
                    gate["value"] = self.pbo["pbo"]
                    gate["status"] = "PASS" if self.pbo["pbo"] < 0.25 else "FAIL"
                elif self.n_trials == 1:
                    gate["status"] = "N/A (no selection)"
        statuses = ["PASS" if g["status"].startswith("N/A") else g["status"] for g in gates]
        if synthetic:
            return ("PIPELINE VERIFICATION ONLY - synthetic data. No empirical conclusion about the "
                    "NASDAQ-100 can be drawn. NOT READY FOR LIVE DEPLOYMENT.")
        if all(s == "PASS" for s in statuses):
            return ("OOS release gates passed: eligible for Phase 3 paper trading (8-12 weeks). "
                    "NOT READY FOR LIVE DEPLOYMENT until paper trading confirms.")
        failed = [g["gate"] for g in gates if g["status"] != "PASS" and not g["status"].startswith("N/A")]
        return "NOT READY FOR LIVE DEPLOYMENT - gates not passed: " + "; ".join(failed)

    # -------------------------------------------------------------- rendering
    def to_markdown(self, report: dict | None = None) -> str:
        rep = self.build() if report is None else report
        seg = rep["segments"]
        L = [f"# {rep['title']} - tearsheet", ""]
        if rep["synthetic"]:
            L += ["> **PIPELINE VERIFICATION ON SYNTHETIC DATA.** Every number below comes from a simulated",
                  "> price process and is not empirical market performance.", ""]
        L += [f"**Verdict:** {rep['verdict']}", "", f"Data sources: {', '.join(rep['data_sources']) or 'n/a'}", ""]
        if rep["meta"]:
            L += ["| Run setting | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rep["meta"].items()] + [""]
        L += ["## Performance by segment (never pooled)", "",
              "| Metric | " + " | ".join(SEGMENT_TITLES[s] for s in SEGMENTS) + " |",
              "|---|" + "---:|" * len(SEGMENTS)]
        for label, key, kind in ROWS:
            cells = [_fmt(None if seg[s] is None else seg[s].get(key), kind) for s in SEGMENTS]
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L += ["", "Monte Carlo rows show the most adverse of the three generators (block bootstrap, "
              "Student-t / q-Gaussian, tempered Levy); per-generator detail is in the JSON report.", "",
              "Validation is the span the hyper-parameters were selected on (optimistic by construction); "
              "In-Sample is the model-estimation span with the parameters later chosen for it; only "
              "Out-of-Sample is untouched by selection.", ""]

        L += ["## Release gates (untouched OOS)", "", "| Gate | Value | Rule | Status |", "|---|---:|---|---|"]
        for g in rep["gates"]:
            v = g["value"]
            vs = "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
            L.append(f"| {g['gate']} | {vs} | {g['rule']} | {g['status']} |")
        L.append("")
        if rep.get("breakers"):
            b = rep["breakers"]
            off = b["deactivated_at"]
            L += [f"Circuit breakers in the OOS run: {b['pauses']} pause(s) at -8%; hard deactivation at -10% of the "
                  f"all-time peak: {'none' if off is None else 'YES, on ' + str(off)[:10] + ' (no trading afterwards)'}. "
                  f"Maximum drawdown is always measured from the un-rebased high-water mark.", ""]
        if rep["dsr"]:
            d = rep["dsr"]
            L += ["## Overfitting diagnostics", ""]
            if d.get("no_selection"):
                L.append(f"- One fixed configuration, nothing selected: DSR equals the probabilistic Sharpe ratio "
                         f"against zero, **{_fmt(d['dsr'], 'f4')}** (per-period SR {d['sr_per_period']:.4f}). "
                         f"PBO is not applicable.")
            else:
                L.append(f"- Deflated Sharpe Ratio: **{_fmt(d['dsr'], 'f4')}** (per-period SR {d['sr_per_period']:.4f} "
                         f"vs expected maximum under the null SR0 {d['sr0_per_period']:.4f}, N = {d['n_trials']} trials, "
                         f"cross-trial mean {d['trial_sr_mean']:.4f}, std {d['trial_sr_std']:.4f}). "
                         f"With the null centred at zero instead (original Bailey / Lopez de Prado form, "
                         f"SR0 {d['sr0_zero_mean_null']:.4f}) it is {_fmt(d['dsr_zero_mean_null'], 'f4')}; the gate "
                         f"uses the first figure.")
            if rep["pbo"]:
                p = rep["pbo"]
                L.append(f"- PBO (CSCV, {p['n_combinations']} splits of {p['n_configs']} configurations): "
                         f"**{p['pbo']:.3f}**; IS->OOS Sharpe slope {p['is_oos_slope']:.3f}; "
                         f"P(OOS loss of IS-best) {p['prob_oos_loss']:.3f}")
            L.append("")
        if rep["oos_cost_stress"]:
            s = rep["oos_cost_stress"]
            L += [f"## OOS under {s.get('cost_stress', 1.5)}x spread / slippage / commission / financing markup", "",
                  f"- Net P&L {_fmt(s['net_pnl'], 'eur')} EUR, Sharpe {_fmt(s['sharpe'], 'f2')}, "
                  f"profit factor {_fmt(s['profit_factor'], 'f2')}, max drawdown {_fmt(s['max_drawdown'], 'pct')}", ""]
        if rep["folds"]:
            f = rep["folds"]
            L += [f"## Walk-forward folds ({f['n_folds']} folds, {100 * f['profitable_share']:.1f}% profitable OOS)", "",
                  "| Fold | OOS window | Val J | Val Sharpe | OOS Sharpe | OOS net P&L | OOS trades |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
            for row in f["table"]:
                L.append(f"| {row['fold']} | {str(row['oos_start'])[:10]} .. {str(row['oos_end'])[:10]} | "
                         f"{row['val_j']:.3f} | {row['val_sharpe']:.2f} | {row['oos_sharpe']:.2f} | "
                         f"{row['oos_net_pnl']:.2f} | {int(row['oos_trades'])} |")
            L.append("")
        return "\n".join(L)

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rep = self.build()
        paths = {"markdown": out / "tearsheet.md", "json": out / "tearsheet.json", "csv": out / "metrics_by_segment.csv"}
        paths["markdown"].write_text(self.to_markdown(rep))
        paths["json"].write_text(json.dumps(_json_safe(rep), indent=2, default=str))
        table = pd.DataFrame(
            {SEGMENT_TITLES[s]: {label: (None if rep["segments"][s] is None else rep["segments"][s].get(key))
                                 for label, key, _ in ROWS} for s in SEGMENTS}
        )
        table.to_csv(paths["csv"])
        for name, res in self.segments.items():
            if res is not None:
                res.equity.to_csv(out / f"equity_{name}.csv")
                res.trades.to_csv(out / f"trades_{name}.csv", index=False)
        return paths
