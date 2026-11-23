"""K_prior and N_eff_prior from the return series written by ``prior_trials.py``.

Raw K counts every variant the four ablations evaluated.  Before the average
pairwise correlation is taken, series that never traded (zero variance, no
defined correlation) and exact duplicates of an earlier series are dropped.

    N_eff_prior = max(1, rho_bar + (1 - rho_bar) K_used)

Usage: prior_trials_summary.py <dir written by prior_trials.py> [<csv.gz to write the used series to>]
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

from ndxswing.analytics.stats import average_pairwise_correlation, effective_trials

src = sys.argv[1]


def load(kind: str) -> pd.DataFrame:
    parts = [pd.read_csv(f, index_col=0, parse_dates=True) for f in sorted(glob.glob(f"{src}/*_{kind}.csv"))]
    return pd.concat(parts, axis=1, join="inner")


def reduce(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop zero-variance series and exact duplicates (first occurrence kept)."""
    live = df.loc[:, df.std(ddof=1) > 0]
    keep: list[str] = []
    for c in live.columns:
        if not any(np.allclose(live[c].to_numpy(), live[k].to_numpy(), rtol=0.0, atol=1e-12) for k in keep):
            keep.append(c)
    return live[keep], df.shape[1] - live.shape[1], live.shape[1] - len(keep)


def summary(df: pd.DataFrame) -> dict:
    used, n_zero, n_dup = reduce(df)
    M = used.to_numpy()
    assert np.isfinite(np.corrcoef(M, rowvar=False)).all()
    rho = average_pairwise_correlation(M)
    return {"k_raw": int(df.shape[1]), "dropped_zero_trade": int(n_zero), "dropped_duplicates": int(n_dup),
            "k_used": int(used.shape[1]), "rho_bar": round(rho, 4), "n_eff": round(effective_trials(used.shape[1], rho), 2),
            "days": int(len(used)), "from": str(used.index[0].date()), "to": str(used.index[-1].date())}


variants = load("variants")
out = {"all four runs, ablation variants": summary(variants)}
for run in sorted({c.split("|")[0] for c in variants.columns}):
    out[f"{run} alone"] = summary(variants[[c for c in variants.columns if c.startswith(run + "|")]])
draws = load("draws")
both = pd.concat([variants, draws], axis=1, join="inner")
out["sensitivity: variants plus the 64 parameter draws of each final configuration"] = summary(both)
print(json.dumps(out, indent=2))
if len(sys.argv) > 2:
    reduce(variants)[0].round(10).to_csv(sys.argv[2], compression="gzip")
