"""Size of the spanning t(alpha) and the paired Sharpe test on simulated GJR-GARCH paths.

premium = variance    mu = lambda sigma^2: timing has no value (the null)
premium = volatility  mu = lambda sigma:   constant conditional Sharpe ratio, timing HAS value;
                      the check there is coverage of the true difference

Usage: calibration.py <out dir> [paths] [n_boot] [jobs]
"""
import json
import multiprocessing as mp
import sys

import numpy as np

from ndxswing.analytics import relative as rel
from ndxswing.research import volmanaged as vm

out = sys.argv[1]
PATHS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
JOBS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
T = 2653            # sessions in the NASDAQ-100 archive


def truth(premium: str, cap: float | None) -> dict:
    """True alpha and Sharpe difference of the managed series from a long simulation."""
    r, _ = vm.simulate_gjr_returns(60, 60_000, premium=premium, seed=999)
    ms, bs = [], []
    for i in range(len(r)):
        m, b = vm.managed_from_returns(r[i], w_max=cap)
        ms.append(m)
        bs.append(b)
    m, b = np.concatenate(ms), np.concatenate(bs)
    sp = rel.spanning_regression(m, b, lags=0)
    return {"alpha_ann": sp["alpha_ann"], "d_sharpe_ann": rel.sharpe_ann(m) - rel.sharpe_ann(b),
            "d_sharpe": (rel.sharpe_ann(m) - rel.sharpe_ann(b)) / np.sqrt(252), "alpha": sp["alpha"],
            "sharpe_bh": rel.sharpe_ann(b), "sharpe_managed": rel.sharpe_ann(m)}


def one(args: tuple) -> dict:
    premium, cap, i, d_true, a_true = args
    r, _ = vm.simulate_gjr_returns(1, T, premium=premium, seed=10_000 + i)
    m, b = vm.managed_from_returns(r[0], w_max=cap)
    sp = rel.spanning_regression(m, b)
    s0 = rel.paired_difference_test(m, b, "sharpe", n_boot=NB, seed=i)
    st = rel.paired_difference_test(m, b, "sharpe", n_boot=NB, seed=i, null_value=d_true)
    c3 = rel.paired_difference_test(m, b, "cer", gamma=3.0, n_boot=NB, seed=i)
    return {"t_alpha": sp["t_alpha"], "t_alpha_true": (sp["alpha"] - a_true) / sp["se_alpha"],
            "p1_sharpe": s0["p_one_sided"], "p2_sharpe_true": st["p_two_sided"], "p1_cer3": c3["p_one_sided"],
            "d_sharpe": s0["delta_ann"]}


res = {}
for premium in ("variance", "volatility"):
    for cap in (None, 1.0):
        tr = truth(premium, cap)
        with mp.get_context("fork").Pool(JOBS) as pool:
            rows = pool.map(one, [(premium, cap, i, tr["d_sharpe"], tr["alpha"]) for i in range(PATHS)], chunksize=10)
        t = np.array([x["t_alpha"] for x in rows])
        key = f"premium proportional to {premium}, {'no cap' if cap is None else 'cap 1.0'}"
        res[key] = {
            "paths": PATHS, "n_boot": NB, "days": T, "true": tr,
            "reject alpha <= 0 (one-sided 5%)": float((t > 1.645).mean()),
            "reject alpha = 0 (two-sided 5%)": float((np.abs(t) > 1.96).mean()),
            "reject alpha = true value (two-sided 5%)": float((np.abs([x["t_alpha_true"] for x in rows]) > 1.96).mean()),
            "reject Sharpe difference <= 0 (one-sided 5%)": float(np.mean([x["p1_sharpe"] < 0.05 for x in rows])),
            "reject Sharpe difference = true value (two-sided 5%)": float(np.mean([x["p2_sharpe_true"] < 0.05 for x in rows])),
            "reject CER(3) difference <= 0 (one-sided 5%)": float(np.mean([x["p1_cer3"] < 0.05 for x in rows])),
            "mean estimated Sharpe difference": float(np.mean([x["d_sharpe"] for x in rows])),
        }
        print(key, json.dumps(res[key], indent=1), flush=True)
json.dump(res, open(f"{out}/calibration.json", "w"), indent=1)
