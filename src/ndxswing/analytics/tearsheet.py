"""Tearsheet: 15 required metrics, strictly segregated by data segment, plus
DSR, PBO, Monte Carlo tail probabilities and the OOS release gates (8.4).

Segments are never pooled.  A segment without data is reported as ``n/a`` -
in particular Paper and Live stay empty until real ledgers exist.  Holdout is
the tail of the stitched OOS series that no retention decision ever saw (only
present when an ablation reserved one).

Release battery (locked, revised once after the 2015-2025 audit):

* significance  - Sharpe > 0.80, DSR > 0.95, CSCV P(OOS loss) < 15%, rank PBO
                  (conditional), profit factor > 1.20, positive under 1.5x costs;
* survival      - max drawdown < 10% and Monte Carlo P(DD >= 20%) < 2%, at the
                  risk budget of the run AND replayed at the 0.75% ceiling;
* evidence      - at least 100 independent OOS trades.

The share of profitable walk-forward folds is informational only: it is
reported on non-overlapping six-month slices of the stitched OOS curve.
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
from .stats import compute_metrics, deflated_sharpe_ratio, slice_stability

SEGMENTS = ("in_sample", "validation", "oos", "holdout", "paper", "live")
SEGMENT_TITLES = {"in_sample": "In-Sample", "validation": "Validation", "oos": "Out-of-Sample",
                  "holdout": "Holdout", "paper": "Paper", "live": "Live"}

#: (row label, metric key, format)
ROWS = [
    ("1. CAGR (net)", "cagr", "pct"),
    ("2. Annualised volatility", "ann_vol", "pct"),
    ("3. Sharpe ratio (net, rf=0)", "sharpe", "f2"),
    ("4. Sortino ratio", "sortino", "f2"),
    ("5. Calmar ratio", "calmar", "f2"),
    ("6. Maximum drawdown", "max_drawdown", "pct"),
    ("6b. Max drawdown / annualised volatility (info)", "dd_over_vol", "f2"),
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
        n_tuned: int | None = None,
        rho_bar: float = 0.0,
        pbo: dict | None = None,
        fold_table: pd.DataFrame | None = None,
        mc_config: MonteCarloConfig = MonteCarloConfig(),
        meta: dict | None = None,
        title: str = "NASDAQ-100 swing system",
        oos_ceiling: BacktestResult | None = None,
        ceiling_risk: float | None = None,
        prior_eff_trials: float = 0.0,
        selection_on_oos: bool = False,
        holdout_start: str | None = None,
    ) -> None:
        unknown = set(segments) - set(SEGMENTS)
        if unknown:
            raise ValueError(f"unknown segments {sorted(unknown)}; allowed: {SEGMENTS}")
        self.segments = {s: segments.get(s) for s in SEGMENTS}
        self.oos_stress = oos_stress
        self.trial_sharpes = None if trial_sharpes is None else np.asarray(trial_sharpes, dtype=float)
        self.n_trials = n_trials          # K: configurations compared (1 = nothing was selected)
        self.n_tuned = n_tuned            # parameters the optimiser moves for the reported configuration
        self.rho_bar = rho_bar            # average pairwise correlation of the K return series
        self.pbo = pbo
        self.fold_table = fold_table
        self.mc_config = mc_config
        self.meta = dict(meta or {})
        self.title = title
        self.oos_ceiling = oos_ceiling            # stitched OOS replayed at the risk ceiling (same fold parameters)
        self.ceiling_risk = ceiling_risk
        self.prior_eff_trials = float(prior_eff_trials)   # effective trials carried from earlier research
        self.selection_on_oos = selection_on_oos  # module variants were compared on the stitched OOS series
        self.holdout_start = holdout_start

    # ----------------------------------------------------------------- build
    def _segment_metrics(self, res: BacktestResult) -> dict:
        m = compute_metrics(res)
        mc = run_monte_carlo(res.daily_returns().to_numpy(), self.mc_config)
        m["monte_carlo"] = mc
        m["dd_over_vol"] = m["max_drawdown"] / m["ann_vol"] if m.get("ann_vol", 0) > 1e-12 else np.nan
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
            dsr = deflated_sharpe_ratio(oos.daily_returns().to_numpy(), self.trial_sharpes, self.n_trials, self.rho_bar,
                                        self.prior_eff_trials)
        elif oos is not None and self.n_trials == 1:
            # one fixed configuration, nothing selected in THIS run: N_eff_cur = 1; SR_0 = 0 unless
            # effective trials are carried forward from earlier research
            dsr = deflated_sharpe_ratio(oos.daily_returns().to_numpy(), None, 1, 1.0, self.prior_eff_trials)
            dsr["no_selection"] = True
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
                     "zero_trade_folds": int((ft["oos_trades"] == 0).sum()) if "oos_trades" in ft else None,
                     "table": ft.to_dict(orient="records")}
        report["folds"] = folds
        report["stability"] = None if oos is None else slice_stability(oos, 6)
        ceiling = None
        if self.oos_ceiling is not None and len(self.oos_ceiling.equity):
            cm = self._segment_metrics(self.oos_ceiling)
            ceiling = {k: cm[k] for k in ("sharpe", "ann_vol", "max_drawdown", "dd_over_vol", "mc_p_dd_20", "mc_p_dd_10",
                                          "mc_worst_99_dd", "net_pnl", "n_trades")}
            ceiling |= {"b_risk": self.ceiling_risk, "pauses": self.oos_ceiling.stats.get("n_pauses"),
                        "deactivated_at": self.oos_ceiling.stats.get("deactivated_at")}
        report["risk_ceiling"] = ceiling
        report["selection_on_oos"] = self.selection_on_oos
        report["holdout_start"] = self.holdout_start
        report["gates"] = self._gates(seg.get("oos"), dsr, stress, ceiling)
        report["verdict"] = self._verdict(report["gates"], synthetic)
        return report

    #: thresholds of the release battery (section 8.4; revised once after the 2015-2025 audit:
    #: the profitable-folds share became informational, the trade floor went from 150 to 100
    #: and the two survival gates are also applied at the risk ceiling)
    MIN_SHARPE, MIN_DSR, MAX_OOS_LOSS, MAX_RANK_PBO, RANK_PBO_MIN_IQR = 0.80, 0.95, 0.15, 0.25, 0.25
    MIN_PF, MAX_DD, MAX_MC_DD20, MIN_TRADES = 1.20, 0.10, 0.02, 100

    def _gates(self, oos: dict | None, dsr: dict | None, stress: dict | None, ceiling: dict | None) -> list[dict]:
        def g(name: str, value: object, ok: bool | None, rule: str, status: str | None = None) -> dict:
            st = status or ("NOT EVALUATED" if ok is None else ("PASS" if ok else "FAIL"))
            return {"gate": name, "value": value, "rule": rule, "status": st}

        def val(d: dict | None, k: str) -> float | None:
            v = None if d is None else d.get(k)
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

        sh, pf, mdd, nt = (val(oos, k) for k in ("sharpe", "profit_factor", "max_drawdown", "n_trades"))
        p20 = val(oos, "mc_p_dd_20")
        d = val(dsr, "dsr")
        no_selection = self.n_trials == 1
        pbo = self.pbo
        # probability that the in-sample winner loses out of sample
        if no_selection:
            loss = g("CSCV P(OOS loss of IS-best)", None, None, "< 15%", "N/A (no selection)")
        else:
            lv = val(pbo, "prob_oos_loss")
            loss = g("CSCV P(OOS loss of IS-best)", lv, None if lv is None else lv < self.MAX_OOS_LOSS, "< 15%")
        # rank PBO is only informative when something was tuned and the variants differ in sample
        rv, iqr = val(pbo, "pbo"), val(pbo, "is_sharpe_iqr")
        rule = "< 0.25 when tuned and IS Sharpe IQR > 0.25"
        if no_selection:
            rank = g("CSCV rank PBO", None, None, rule, "N/A (no selection)")
        elif not self.n_tuned:
            rank = g("CSCV rank PBO", rv, None, rule, "N/A (nothing tuned)")
        elif rv is None or iqr is None:
            rank = g("CSCV rank PBO", rv, None, rule)
        elif iqr <= self.RANK_PBO_MIN_IQR:
            rank = g("CSCV rank PBO", rv, None, rule, f"N/A (IS Sharpe IQR {iqr:.2f}: homogeneous variants)")
        else:
            rank = g("CSCV rank PBO", rv, rv < self.MAX_RANK_PBO, rule)
        cdd, cp20 = val(ceiling, "max_drawdown"), val(ceiling, "mc_p_dd_20")
        tier = "risk ceiling" if not self.ceiling_risk else f"{100 * self.ceiling_risk:.2f}% risk ceiling"
        return [
            g("OOS net Sharpe", sh, None if sh is None else sh > self.MIN_SHARPE, "> 0.80"),
            g("Deflated Sharpe Ratio (zero-mean null, N_eff)", d, None if d is None else d > self.MIN_DSR, "> 0.95"),
            loss,
            rank,
            g("OOS net profit factor", pf, None if pf is None else pf > self.MIN_PF, "> 1.20"),
            g("OOS net P&L under 1.5x costs", val(stress, "net_pnl"),
              None if stress is None else stress["net_pnl"] > 0, "> 0"),
            g("OOS maximum drawdown (true high-water mark)", mdd, None if mdd is None else mdd < self.MAX_DD, "< 10%"),
            g("Monte Carlo P(DD >= 20%)", p20, None if p20 is None else p20 < self.MAX_MC_DD20, "< 2%"),
            g(f"OOS maximum drawdown at the {tier}", cdd, None if cdd is None else cdd < self.MAX_DD, "< 10%"),
            g(f"Monte Carlo P(DD >= 20%) at the {tier}", cp20, None if cp20 is None else cp20 < self.MAX_MC_DD20, "< 2%"),
            g("Independent OOS trades", nt, None if nt is None else nt >= self.MIN_TRADES, f">= {self.MIN_TRADES}"),
        ]

    def _verdict(self, gates: list[dict], synthetic: bool) -> str:
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
              "In-Sample is the model-estimation span with the parameters later chosen for it.", ""]
        hs = rep.get("holdout_start")
        if rep.get("selection_on_oos"):
            L += ["Out-of-Sample is the stitched walk-forward series. Its hyper-parameters were chosen on validation data, "
                  "but the module retention decisions of the ablation were made ON this series"
                  + (f" up to the day before {hs}" if hs else "") + ", so it is not selection-free; the DSR and "
                  "P(OOS loss) gates exist to account for that. "
                  + (f"Holdout is the part of the same series from {hs} on, which no decision ever saw: the only "
                     "selection-free column, and too short to carry a gate on its own." if hs else
                     "No holdout was reserved, so no column of this table is selection-free."), ""]
        else:
            L += ["Out-of-Sample is the stitched walk-forward series; it was not used to choose anything in this run "
                  "(earlier runs on the same data are a separate matter: see prior effective trials).", ""]

        L += ["## Release gates (stitched walk-forward OOS)", "", "| Gate | Value | Rule | Status |", "|---|---:|---|---|"]
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
            prior = d.get("prior_eff_trials", 0.0) or 0.0
            if d.get("no_selection") and not prior:
                L.append(f"- One fixed configuration, nothing selected: N_eff = 1, SR0 = 0, so the DSR is the "
                         f"probabilistic Sharpe ratio against zero, **{_fmt(d['dsr'], 'f4')}** "
                         f"(per-period SR {d['sr_per_period']:.4f}). The CSCV statistics are not applicable.")
            elif d.get("no_selection"):
                L.append(f"- **Gate.** One fixed configuration in this run, {prior:.1f} effective trials carried from "
                         f"earlier research: N_eff = {d['n_eff']:.1f}, SR0 = {d['sr0_per_period']:.4f} per period "
                         f"(variance 1/T), DSR **{_fmt(d['dsr'], 'f4')}** (per-period SR {d['sr_per_period']:.4f}; "
                         f"against zero: {_fmt(d['psr_zero'], 'f4')}). The CSCV statistics are not applicable.")
            else:
                L.append(f"- **Gate.** DSR, zero-mean null: **{_fmt(d['dsr'], 'f4')}** (per-period SR "
                         f"{d['sr_per_period']:.4f} vs SR0 {d['sr0_per_period']:.4f}; K = {d['n_trials']} configurations "
                         f"compared, average pairwise correlation {d['rho_bar']:.2f}, N_eff = "
                         f"{d.get('n_eff_current', d['n_eff']):.1f}; cross-trial Sharpe std {d['trial_sr_std']:.4f})")
                if prior:
                    L.append(f"- Prior effective trials carried from earlier research: {prior:.1f}, so N_eff = "
                             f"{d['n_eff']:.1f} in total and the benchmark variance blends the cross-trial variance "
                             f"with 1/T for the prior trials.")
                L.append(f"- Informational. DSR with the cross-trial mean in the benchmark and all K trials: "
                         f"{_fmt(d['dsr_cross_trial_mean'], 'f4')} (SR0 {d['sr0_cross_trial_mean']:.4f}, cross-trial mean "
                         f"{d['trial_sr_mean']:.4f}). Probabilistic Sharpe ratio against zero (N_eff = 1): "
                         f"{_fmt(d['psr_zero'], 'f4')}.")
            if rep["pbo"]:
                p = rep["pbo"]
                L.append(f"- CSCV over {p['n_combinations']} splits of the {p['n_configs']} configurations compared: "
                         f"P(OOS loss of the in-sample winner) **{p['prob_oos_loss']:.3f}**; IS->OOS Sharpe slope "
                         f"{p['is_oos_slope']:.3f}")
                if p.get("pbo") is not None:
                    L.append(f"- Rank PBO over {p.get('n_rank_configs', p['n_configs'])} parameter draws of this configuration: "
                             f"{p['pbo']:.3f}; in-sample Sharpe IQR across the draws {p['is_sharpe_iqr']:.2f}")
            L.append("")
        if rep["oos_cost_stress"]:
            s = rep["oos_cost_stress"]
            L += [f"## OOS under {s.get('cost_stress', 1.5)}x spread / slippage / commission / financing markup", "",
                  f"- Net P&L {_fmt(s['net_pnl'], 'eur')} EUR, Sharpe {_fmt(s['sharpe'], 'f2')}, "
                  f"profit factor {_fmt(s['profit_factor'], 'f2')}, max drawdown {_fmt(s['max_drawdown'], 'pct')}", ""]
        if rep.get("risk_ceiling"):
            c = rep["risk_ceiling"]
            off = c["deactivated_at"]
            L += [f"## Stitched OOS replayed at the {_fmt(c['b_risk'], 'pct')} risk ceiling", "",
                  "Same signals and fold parameters, nothing re-estimated; only the risk budget per trade changes.", "",
                  f"- Max drawdown {_fmt(c['max_drawdown'], 'pct')}, annualised volatility {_fmt(c['ann_vol'], 'pct')}, "
                  f"max drawdown / volatility {_fmt(c['dd_over_vol'], 'f2')}, Sharpe {_fmt(c['sharpe'], 'f2')}, "
                  f"net P&L {_fmt(c['net_pnl'], 'eur')} EUR",
                  f"- Monte Carlo P(DD >= 10%) {_fmt(c['mc_p_dd_10'], 'pct')}, P(DD >= 20%) {_fmt(c['mc_p_dd_20'], 'pct')}, "
                  f"worst 99% drawdown {_fmt(c['mc_worst_99_dd'], 'pct')}; breaker pauses {c['pauses']}, hard deactivation "
                  f"{'none' if off is None else 'YES, on ' + str(off)[:10]}", ""]
        if rep.get("stability") and rep["stability"]["n_slices"]:
            st = rep["stability"]
            L += [f"## Stability over time (informational, not a gate)", "",
                  f"Non-overlapping {st['months']}-month slices of the stitched OOS curve: {st['profitable']} of "
                  f"{st['n_slices']} profitable ({100 * st['share_profitable']:.1f}%), {st['losing']} losing, "
                  f"{st['inactive']} with no trade (active slices: {st['active']}).", "",
                  "| Slice | Window | Net P&L | Trades | Note |", "|---:|---|---:|---:|---|"]
            for row in st["table"]:
                note = "" if row["complete"] else "incomplete, not counted"
                note = "no trade" if row["inactive"] and row["complete"] else note
                L.append(f"| {row['slice']} | {str(row['start'])[:10]} .. {str(row['end'])[:10]} | {row['net_pnl']:.2f} | "
                         f"{row['trades']} | {note} |")
            L.append("")
        if rep["folds"]:
            f = rep["folds"]
            L += [f"## Walk-forward folds ({f['n_folds']} overlapping OOS windows, each run on its own; informational)", "",
                  f"{100 * f['profitable_share']:.1f}% of the windows were profitable. Adjacent windows share half "
                  f"their span, so trades are counted twice across the table.", "",
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
