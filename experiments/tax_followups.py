"""
Further analyses of the tax (Supplementary Section S5, Table S3).

  degcorr   tax with degree-corrected distance decoders (node biases b_i + b_j in the link logit), shared vs
            separate, d = 2, on the graphs of the tax experiment (warp and twin)
  equaldim  tax at equal TOTAL dimension: shared 4-D code vs separate 2-D + 2-D codes
  gen       new graphs for seeds 601-612, kappa in {1.0, 1.5, 2.0}:
              areafree  global warp with the area field switched off (anisotropy and orientation only)
              degtwin   degree-preserving scattered twin: double-edge swaps of the aligned graph until its mean
                        log X-distance over links reaches the (standard) warp's
  taxnew    tax (d = 2, model seed 0) on the new graphs; taxnew_dc the same with degree-corrected decoders
  truth     true Delta_2(d_X, d_graph) and Delta_2(d_attr, d_graph) for every warp graph
  aggregate summary JSON (read by tables_supp.py and figures_paper.py)

usage: python experiments/tax_followups.py degcorr|equaldim --dir results/synthetic/globaltax/seed601/k1.50 --seed 601 --graph warp
       python experiments/tax_followups.py gen --seed 601 --kappa 1.5
       python experiments/tax_followups.py taxnew|taxnew_dc --seed 601 --kappa 1.5 --graph areafree|degtwin
       python experiments/tax_followups.py truth | aggregate
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
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ROOT, WORK, jsonable, triu
from experiments import global_tax as GT

BASE = os.path.join(WORK, "results", "synthetic", "globaltax")
OUT = os.path.join(WORK, "results", "synthetic", "globaltax_followups")
torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------- degree-corrected fit
def fit_dc(dim, iu, y, train, logdx, lam_link, lam_attr, Z0, seed, n):
    torch.manual_seed(seed)
    Z = torch.nn.Parameter(torch.as_tensor(Z0[:, :dim].copy()) + 0.01 * torch.randn(n, dim))
    nb = torch.nn.Parameter(torch.zeros(n))
    b, w, c = (torch.nn.Parameter(torch.tensor(v)) for v in (0.0, 0.0, 0.0))
    opt = torch.optim.Adam([Z, nb, b, w, c], lr=0.02)
    I, Jx = torch.as_tensor(iu[0]), torch.as_tensor(iu[1])
    yt, tr, lx = torch.as_tensor(y), torch.as_tensor(train), torch.as_tensor(logdx)
    pw = torch.tensor(float((y[train] == 0).sum() / max((y[train] == 1).sum(), 1)))
    prev = None
    for ep in range(GT.MAX_EPOCHS):
        opt.zero_grad()
        d2 = ((Z[I] - Z[Jx]) ** 2).sum(-1)
        loss = torch.tensor(0.0)
        if lam_link > 0:
            logit = b + nb[I] + nb[Jx] - torch.exp(w) * d2
            loss = loss + lam_link * torch.nn.functional.binary_cross_entropy_with_logits(logit[tr], yt[tr], pos_weight=pw)
        if lam_attr > 0:
            loss = loss + lam_attr * ((0.5 * torch.log(d2 + 1e-12) - lx - c) ** 2).mean()
        loss.backward()
        opt.step()
        if ep % GT.CHECK == 0:
            v = float(loss)
            if prev is not None and abs(prev - v) <= GT.REL_TOL * abs(prev):
                break
            prev = v
    Zn, nbn = Z.detach().numpy(), nb.detach().numpy()
    score = nbn[iu[0]] + nbn[iu[1]] - float(torch.exp(w)) * ((Zn[iu[0]] - Zn[iu[1]]) ** 2).sum(-1)
    return Zn, score


def setup(A, X, seed, dim):
    rng = np.random.default_rng([seed, 2222])
    iu, y, train, test, val = GT.split(A, rng)
    logdx = np.log(np.linalg.norm(X[iu[0]] - X[iu[1]], axis=1) + 1e-12)
    Xc = X - X.mean(0)
    Z0 = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:dim].T
    Z0 = Z0 / Z0.std()
    return iu, y, train, test, val, logdx, Z0, GT.knn(X)


def tax_degcorr(A, X, seed, dim=2, ms=0):
    n = A.shape[0]
    iu, y, train, (tp, tn), (vp, vn), logdx, Z0, nnX = setup(A, X, seed, dim)
    Zg, sg = fit_dc(dim, iu, y, train, logdx, 1.0, 0.0, Z0, ms, n)
    Za, _ = fit_dc(dim, iu, y, train, logdx, 0.0, 1.0, Z0, ms, n)
    L_s, Lv_s, R_s = GT.auc(sg[tp], sg[tn]), GT.auc(sg[vp], sg[vn]), GT.recall10(Za, nnX)
    best = None
    for lam in GT.LAMBDAS:
        Zm, sm = fit_dc(dim, iu, y, train, logdx, 1.0, lam, Z0, ms, n)
        Rm = GT.recall10(Zm, nnX)
        crit = max(Lv_s - GT.auc(sm[vp], sm[vn]), R_s - Rm)
        if best is None or crit < best[0]:
            best = (crit, max(L_s - GT.auc(sm[tp], sm[tn]), R_s - Rm), lam)
    return dict(tax=best[1], lam=best[2], sep_link=L_s, sep_attr=R_s)


def tax_equaldim(A, X, seed, ms=0):
    """shared 4-D code vs separate 2-D link code + 2-D attribute code (equal total dimension)."""
    iu, y, train, (tp, tn), (vp, vn), logdx, Z0, nnX = setup(A, X, seed, 4)
    Zg, sg, _ = GT.fit(2, iu, y, train, logdx, 1.0, 0.0, Z0, ms)
    Za, _, _ = GT.fit(2, iu, y, train, logdx, 0.0, 1.0, Z0, ms)
    L_s, Lv_s, R_s = GT.auc(sg[tp], sg[tn]), GT.auc(sg[vp], sg[vn]), GT.recall10(Za, nnX)
    best = None
    for lam in GT.LAMBDAS:
        Zm, sm, _ = GT.fit(4, iu, y, train, logdx, 1.0, lam, Z0, ms)
        Rm = GT.recall10(Zm, nnX)
        crit = max(Lv_s - GT.auc(sm[vp], sm[vn]), R_s - Rm)
        if best is None or crit < best[0]:
            best = (crit, max(L_s - GT.auc(sm[tp], sm[tn]), R_s - Rm), lam)
    return dict(tax=best[1], lam=best[2], sep_link=L_s, sep_attr=R_s)


# ----------------------------------------------------------------------------- new graphs
def gen(seed, kappa):
    from aligning.synthetic import SyntheticConfig, ManifoldSpec, generate, true_geodesic_distances, true_metric_fn
    from aligning.twist import TwistSpec, generate_twisted
    from experiments.run_twist import GRAPH
    from experiments.run_global_twist import fourier_field, geodesics, mean_log_xdist
    import networkx as nx
    out = os.path.join(OUT, f"seed{seed:03d}", f"k{kappa:.2f}")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(10_000 + seed)
    manifold = ManifoldSpec.sample(rng, 20)
    Z = rng.normal(size=(500, 2))
    d_attr = true_geodesic_distances(manifold, Z)
    cfg = SyntheticConfig(graph=GRAPH)
    X = generate(cfg, seed, d_true=d_attr, manifold=manifold, Z=Z)["X"]
    # area-free warp: same fields as run_global_twist (same rng stream), area field set to 0
    frng = np.random.default_rng([seed, 2222])
    fa, fs, ft = (fourier_field(frng, Z) for _ in range(3))
    G_attr = true_metric_fn(manifold)

    def fn(u):
        G = G_attr(u)
        th = np.pi * ft(u)
        a = torch.stack([torch.cos(th), torch.sin(th)], -1)
        lam, mu = torch.exp(kappa * fs(u) / np.sqrt(2)), torch.exp(-kappa * fs(u) / np.sqrt(2))
        Ga = torch.einsum("nij,nj->ni", G, a)
        m = Ga / torch.sqrt(torch.einsum("ni,ni->n", a, Ga)).unsqueeze(-1)
        return mu[:, None, None] * G + (lam - mu)[:, None, None] * m.unsqueeze(-1) * m.unsqueeze(-2)
    d_af = geodesics(fn, Z)
    A_af = generate_twisted(cfg, seed, TwistSpec(kind="none"), manifold, Z, d_graph=d_af)["A"]
    # degree-preserving twin of the STANDARD warp's homophily
    std = torch.load(os.path.join(BASE, f"seed{seed:03d}", f"k{kappa:.2f}", "inputs.pt"), weights_only=False)
    target = mean_log_xdist(np.asarray(std["A_warp"]), X)
    A0 = generate_twisted(cfg, seed, TwistSpec(kind="none"), manifold, Z, d_graph=d_attr)["A"]
    G = nx.from_numpy_array((A0 > 0).astype(int))
    srng = np.random.default_rng([seed, 4444, int(round(100 * kappa))])
    swaps = 0
    while mean_log_xdist(nx.to_numpy_array(G, nodelist=range(500)), X) < target and swaps < 200_000:
        nx.double_edge_swap(G, nswap=20, max_tries=10_000, seed=int(srng.integers(2 ** 31)))
        swaps += 20
    A_dt = nx.to_numpy_array(G, nodelist=range(500))
    info = dict(seed=seed, kappa=kappa, target_homophily=target, homophily_degtwin=mean_log_xdist(A_dt, X),
                homophily_areafree=mean_log_xdist(A_af, X), swaps=swaps,
                sd_log_degree=dict(areafree=float(np.log(np.maximum((A_af > 0).sum(1), 1)).std()),
                                   degtwin=float(np.log(np.maximum((A_dt > 0).sum(1), 1)).std()),
                                   aligned=float(np.log(np.maximum((A0 > 0).sum(1), 1)).std())))
    torch.save(dict(X=X, A_areafree=A_af.astype(np.float32), A_degtwin=A_dt.astype(np.float32), d_areafree=triu(d_af).astype(np.float32)),
               os.path.join(out, "inputs.pt"))
    json.dump(jsonable(info), open(os.path.join(out, "instance.json"), "w"))
    print(json.dumps(jsonable(info)))


def truth():
    rows = []
    for d in sorted(glob.glob(os.path.join(BASE, "seed6*", "k*"))):
        inp = torch.load(os.path.join(d, "inputs.pt"), weights_only=False)
        X = np.asarray(inp["X"], float)
        da, dg = inp["d_attr"].astype(float), inp["d_graph"].astype(float)
        dx = np.linalg.norm(X[:, None] - X[None], axis=-1)[np.triu_indices(len(X), 1)]
        rows.append(dict(dir=d, seed=int(d.split("seed")[1][:3]), kappa=float(d.split("/k")[-1]),
                         delta2_attr_graph=float((np.log(dg) - np.log(da)).std()), delta2_X_graph=float((np.log(dg) - np.log(dx)).std())))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "truth.json"), "w"), indent=1)
    print(len(rows))


def aggregate():
    """Per seed: mean over kappa in {1, 1.5, 2} of contrasts (one-sided t over 12 seeds). All tax values use model seed 0
    where the tax-experiment value is compared with a new single-seed computation."""
    seeds = list(range(601, 613))
    K = [1.0, 1.5, 2.0]
    J = lambda *p: json.load(open(os.path.join(*p)))
    p22 = lambda s, k, g: J(BASE, f"seed{s:03d}", f"k{k:.2f}", f"tax_{g}.json")["d2"]["tax"][0]       # model seed 0
    fu = lambda s, k, name: J(OUT, f"seed{s:03d}", f"k{k:.2f}", name)

    def t1(x):
        x = np.asarray(x, float)
        r = stats.ttest_1samp(x, 0.0, alternative="greater")
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        return dict(mean=float(x.mean()), ci95=[float(x.mean() - h), float(x.mean() + h)], p=float(r.pvalue), positive=int((x > 0).sum()), n=len(x))
    R = {}
    R["degcorr_warp_minus_twin"] = t1([np.mean([fu(s, k, "degcorr_warp.json")["tax"] - fu(s, k, "degcorr_twin.json")["tax"] for k in K]) for s in seeds])
    R["degcorr_warp_by_kappa"] = {k: float(np.mean([fu(s, k, "degcorr_warp.json")["tax"] for s in seeds])) for k in [0.0] + K}
    R["degcorr_twin_by_kappa"] = {k: float(np.mean([fu(s, k, "degcorr_twin.json")["tax"] for s in seeds])) for k in K}
    R["equaldim_warp_minus_twin"] = t1([np.mean([fu(s, k, "equaldim_warp.json")["tax"] - fu(s, k, "equaldim_twin.json")["tax"] for k in K]) for s in seeds])
    R["equaldim_warp_by_kappa"] = {k: float(np.mean([fu(s, k, "equaldim_warp.json")["tax"] for s in seeds])) for k in [0.0] + K}
    R["equaldim_twin_by_kappa"] = {k: float(np.mean([fu(s, k, "equaldim_twin.json")["tax"] for s in seeds])) for k in K}
    af = lambda s, k: J(OUT, f"seed{s:03d}", f"k{k:.2f}", "tax_areafree.json")["tax"]
    dt = lambda s, k: J(OUT, f"seed{s:03d}", f"k{k:.2f}", "tax_degtwin.json")["tax"]
    R["areafree_minus_aligned"] = t1([np.mean([af(s, k) - p22(s, 0.0, "warp") for k in K]) for s in seeds])
    R["areafree_by_kappa"] = {k: float(np.mean([af(s, k) for s in seeds])) for k in K}
    R["warp_minus_degtwin"] = t1([np.mean([p22(s, k, "warp") - dt(s, k) for k in K]) for s in seeds])
    R["degtwin_by_kappa"] = {k: float(np.mean([dt(s, k) for s in seeds])) for k in K}
    R["areafree_minus_degtwin"] = t1([np.mean([af(s, k) - dt(s, k) for k in K]) for s in seeds])
    inst = [fu(s, k, "instance.json") for s in seeds for k in K]
    R["areafree_homophily_mean"] = float(np.mean([i["homophily_areafree"] for i in inst]))
    R["degtwin_homophily_gap_max"] = float(np.max([abs(i["homophily_degtwin"] - i["target_homophily"]) for i in inst]))
    R["sd_log_degree"] = {g: float(np.mean([i["sd_log_degree"][g] for i in inst])) for g in ("areafree", "degtwin", "aligned")}
    tr = {(t["seed"], t["kappa"]): t for t in J(OUT, "truth.json")}
    R["true_delta_attr_graph_by_kappa"] = {k: float(np.mean([tr[(s, k)]["delta2_attr_graph"] for s in seeds])) for k in [0.0] + K}
    xs = [tr[(s, k)]["delta2_attr_graph"] for s in seeds for k in [0.0] + K]
    ys = [J(BASE, f"seed{s:03d}", f"k{k:.2f}", "tax_warp.json")["d2"]["tax_mean"] for s in seeds for k in [0.0] + K]
    R["spearman_tax_true_delta_warp"] = float(stats.spearmanr(xs, ys)[0])
    afd = lambda s, k: J(OUT, f"seed{s:03d}", f"k{k:.2f}", "taxdc_areafree.json")["tax"]
    dtd = lambda s, k: J(OUT, f"seed{s:03d}", f"k{k:.2f}", "taxdc_degtwin.json")["tax"]
    R["degcorr_areafree_minus_degtwin"] = t1([np.mean([afd(s, k) - dtd(s, k) for k in K]) for s in seeds])
    R["degcorr_areafree_by_kappa"] = {k: float(np.mean([afd(s, k) for s in seeds])) for k in K}
    R["degcorr_degtwin_by_kappa"] = {k: float(np.mean([dtd(s, k) for s in seeds])) for k in K}
    R["degcorr_areafree_minus_aligned"] = t1([np.mean([afd(s, k) - fu(s, 0.0, "degcorr_warp.json")["tax"] for k in K]) for s in seeds])
    json.dump(jsonable(R), open(os.path.join(OUT, "aggregate.json"), "w"), indent=1)
    print(json.dumps(jsonable(R), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["degcorr", "equaldim", "gen", "taxnew", "taxnew_dc", "truth", "aggregate"])
    ap.add_argument("--dir")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--kappa", type=float)
    ap.add_argument("--graph")
    a = ap.parse_args()
    if a.mode in ("degcorr", "equaldim"):
        inp = torch.load(os.path.join(a.dir, "inputs.pt"), weights_only=False)
        X, A = np.asarray(inp["X"], float), np.asarray(inp[f"A_{a.graph}"], float)
        r = tax_degcorr(A, X, a.seed) if a.mode == "degcorr" else tax_equaldim(A, X, a.seed)
        rel = os.path.relpath(a.dir, BASE)
        out = os.path.join(OUT, rel)
        os.makedirs(out, exist_ok=True)
        json.dump(jsonable(r), open(os.path.join(out, f"{a.mode}_{a.graph}.json"), "w"))
    elif a.mode == "gen":
        gen(a.seed, a.kappa)
    elif a.mode == "taxnew":
        d = os.path.join(OUT, f"seed{a.seed:03d}", f"k{a.kappa:.2f}")
        inp = torch.load(os.path.join(d, "inputs.pt"), weights_only=False)
        X, A = np.asarray(inp["X"], float), np.asarray(inp[f"A_{a.graph}"], float)
        r = GT.tax(A, X, a.seed, dim=2, model_seeds=(0,))
        json.dump(jsonable(dict(tax=r["tax_mean"], sep_link=r["sep_link"][0], sep_attr=r["sep_attr"][0], lam=r["lam"][0])),
                  open(os.path.join(d, f"tax_{a.graph}.json"), "w"))
    elif a.mode == "taxnew_dc":
        d = os.path.join(OUT, f"seed{a.seed:03d}", f"k{a.kappa:.2f}")
        inp = torch.load(os.path.join(d, "inputs.pt"), weights_only=False)
        X, A = np.asarray(inp["X"], float), np.asarray(inp[f"A_{a.graph}"], float)
        json.dump(jsonable(tax_degcorr(A, X, a.seed)), open(os.path.join(d, f"taxdc_{a.graph}.json"), "w"))
    elif a.mode == "truth":
        truth()
    else:
        aggregate()


if __name__ == "__main__":
    main()
