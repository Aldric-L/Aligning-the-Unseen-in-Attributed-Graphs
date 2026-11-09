"""
Analysis of the tax experiment (Fig. 2, Supplementary Section S5).

Per seed s and kappa k: graphs warp(k) (k in {0, 1.0, 1.5, 2.0}) and twin(k) (k in {1.0, 1.5, 2.0}).
  tax(g)               d = 2 primary (mean of model seeds), d = 4 secondary          [global_tax_run.py]
  excess_sigma(g)      SD(log d2 - log d1) on g minus the same on g's bootstrap graph [run_global_twist.py arrays]
  competitors(g)       homophily_X, direct, sigma_sp, sigma_link
H1: per-seed mean over k of tax(warp) - tax(twin) > 0 (t over seeds).
H2: per-seed mean over k of excess(warp) - excess(twin) > 0 (t); same contrast for competitors.
H3: per-seed Spearman(excess, tax) over the seed's 7 graphs; mean > 0 (t over seeds); ranking by |rho|.
Holm over H2, H3.

usage: python experiments/analyze_global_tax.py [--base results/synthetic/globaltax] [--dim 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ROOT, WORK, jsonable

KAPPAS = (1.0, 1.5, 2.0)


def sigma_delta(f):
    z = np.load(f)
    return float((np.log(z["d2"].astype(float)) - np.log(z["d1"].astype(float))).std())


def iso_sd(f):
    """SD of the isotonic-regression residual of log d2 on log d1 (non-monotone distortion)."""
    from sklearn.isotonic import IsotonicRegression
    z = np.load(f)
    x, y = np.log(z["d1"].astype(float)), np.log(z["d2"].astype(float))
    return float((y - IsotonicRegression(out_of_bounds="clip").fit_transform(x, y)).std())


def competitors(f, X):
    z = np.load(f)
    A = z["A"].astype(float)
    d1 = squareform(z["d1"].astype(float))
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    i, j = np.where(np.triu(A > 0, 1))
    sp = shortest_path(np.where(A > 0, d1, 0.0), directed=False)
    fin = np.isfinite(sp[iu])
    import networkx as nx
    deg = (A > 0).sum(1)
    clus = float(np.mean(list(nx.clustering(nx.from_numpy_array((A > 0).astype(int))).values())))
    return dict(homophily_X=float(np.mean(np.log(np.linalg.norm(X[i] - X[j], axis=1)))),
                sd_log_degree=float(np.log(np.maximum(deg, 1)).std()), clustering=clus,
                direct=float(np.mean(d1[i, j] / np.median(d1[iu]))),
                sigma_sp=float((np.log(sp[iu][fin]) - np.log(d1[iu][fin])).std()),
                sigma_link=float(np.log(d1[i, j]).std()))


def one_sided_t(x):
    x = np.asarray(x, float)
    t = stats.ttest_1samp(x, 0.0, alternative="greater")
    return dict(mean=float(x.mean()), p=float(t.pvalue), positive=int((x > 0).sum()), n=len(x), per_seed=x.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(WORK, "results", "synthetic", "globaltax"))
    ap.add_argument("--dim", type=int, default=2)
    args = ap.parse_args()
    seeds = sorted(int(os.path.basename(d)[4:]) for d in glob.glob(os.path.join(args.base, "seed*")))
    rows, missing = [], []
    for s in seeds:
        for k in (0.0,) + KAPPAS:
            d = os.path.join(args.base, f"seed{s:03d}", f"k{k:.2f}")
            X = np.asarray(torch.load(os.path.join(d, "inputs.pt"), weights_only=False)["X"], float)
            for g in (("warp",) if k == 0 else ("warp", "twin")):
                fa, fb, ft = (os.path.join(d, x) for x in (f"{g}_arrays.npz", f"{g}_boot_arrays.npz", f"tax_{g}.json"))
                if not all(os.path.exists(x) for x in (fa, fb, ft)):
                    missing.append(f"{s} {k} {g}")
                    continue
                T = json.load(open(ft))
                sg, sb, ig, ib = sigma_delta(fa), sigma_delta(fb), iso_sd(fa), iso_sd(fb)
                rows.append(dict(seed=s, kappa=k, graph=g, tax=T[f"d{args.dim}"]["tax_mean"], tax_d4=T["d4"]["tax_mean"],
                                 excess=sg - sb, sigma_delta=sg, sigma_delta_boot=sb, iso_excess=ig - ib, iso_graph=ig, iso_boot=ib,
                                 **competitors(fa, X)))
    assert not missing, f"missing instances (all 84 are required): {missing}"
    R = dict(analysis="tax experiment", dim=args.dim, n_graphs=len(rows), seeds=seeds)
    get = lambda s, k, g, key: next(r[key] for r in rows if r["seed"] == s and r["kappa"] == k and r["graph"] == g)
    contrast = lambda key: [np.mean([get(s, k, "warp", key) - get(s, k, "twin", key) for k in KAPPAS]) for s in seeds]
    R["H1"] = one_sided_t(contrast("tax"))
    R["H1"]["tax_at_kappa0_mean"] = float(np.mean([get(s, 0.0, "warp", "tax") for s in seeds]))
    R["H1"]["passes"] = bool(R["H1"]["p"] < 0.05 and R["H1"]["positive"] >= 10)
    R["H2"] = one_sided_t(contrast("excess"))
    COMPS = ("homophily_X", "direct", "sigma_sp", "sigma_link", "sd_log_degree", "clustering")
    R["H2_competitors"] = {c: one_sided_t(contrast(c)) for c in COMPS}
    R["H2_terms"] = dict(graph_term=one_sided_t(contrast("sigma_delta")), bootstrap_term=one_sided_t(contrast("sigma_delta_boot")),
                         note="excess contrast = graph term - bootstrap term")
    R["H2_iso_secondary"] = dict(excess=one_sided_t(contrast("iso_excess")), graph_term=one_sided_t(contrast("iso_graph")),
                                 bootstrap_term=one_sided_t(contrast("iso_boot")))
    R["H1_d4"] = one_sided_t(contrast("tax_d4"))
    comps = COMPS
    rho = {key: [] for key in ("excess",) + comps}
    for s in seeds:
        sub = [r for r in rows if r["seed"] == s]
        tax = [r["tax"] for r in sub]
        for key in rho:
            rho[key].append(float(stats.spearmanr([r[key] for r in sub], tax)[0]))
    R["H3"] = one_sided_t(rho["excess"])
    R["H3_ranking"] = {c: dict(mean_rho=float(np.nanmean(rho[c])),
                               abs_diff=one_sided_t(np.abs(rho["excess"]) - np.abs(np.array(rho[c])))) for c in comps}
    ps = np.array([R["H2"]["p"], R["H3"]["p"]])
    o = np.argsort(ps)
    adj = np.empty(2)
    adj[o[0]] = min(1, 2 * ps[o[0]])
    adj[o[1]] = min(1, max(adj[o[0]], ps[o[1]]))
    R["H2"]["p_holm"], R["H3"]["p_holm"] = float(adj[0]), float(adj[1])
    R["H2"]["supported"] = bool(adj[0] < 0.05 and R["H2"]["positive"] >= 10)
    R["H3"]["supported"] = bool(adj[1] < 0.05)
    R["H2_also_separating"] = {c: ("warp > twin" if v["positive"] >= 10 else "twin > warp") for c, v in R["H2_competitors"].items()
                               if v["positive"] >= 10 or v["n"] - v["positive"] >= 10}          # two-sided
    within = {}
    for k in KAPPAS:                                                                         # secondary
        ex = [get(s, k, g, "excess") - get(s, 0.0, "warp", "excess") for s in seeds for g in ("warp", "twin")]
        tx = [get(s, k, g, "tax") - get(s, 0.0, "warp", "tax") for s in seeds for g in ("warp", "twin")]
        r_, p_ = stats.spearmanr(ex, tx)
        within[f"k{k:.2f}"] = dict(spearman=float(r_), p=float(p_), n=len(ex))
    R["H3_within_kappa_secondary"] = within
    R["by_kappa"] = {f"{g}_k{k:.2f}": {key: float(np.mean([r[key] for r in rows if r["kappa"] == k and r["graph"] == g]))
                                       for key in ("tax", "excess", "sigma_delta") + comps}
                     for k in (0.0,) + KAPPAS for g in (("warp",) if k == 0 else ("warp", "twin"))}
    json.dump(jsonable(dict(R, rows=rows)), open(os.path.join(args.base, f"analysis_d{args.dim}.json"), "w"), indent=1)
    print(json.dumps(jsonable(R), indent=1))


if __name__ == "__main__":
    main()
