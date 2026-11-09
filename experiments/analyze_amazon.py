"""
Analysis of the product network (Fig. 4, Supplementary Section S6): the buy-after-viewing layer (S) vs the
also-bought-only layer (C) on Amazon Musical Instruments.

Statistic: for an ordered category pair (X, Y) with control Y' (the eligible category at the closest median
attribute distance to X, excluding X's specified partners in either orientation), per seed
    DiD = [m^C(X,Y) - m^C(X,Y')] - [m^S(X,Y) - m^S(X,Y')],
m = block median of a pair distortion over the pairs allowed by a mask; seed-averaged; T = mean over the kept pairs.
Null: category relabelling, which keeps the sharing pattern of the specified pairs. Secondary: random ordered pairs.

usage: python experiments/analyze_amazon.py [--base results/amazon] [--data data/Amazon/Musical_Instruments] [--rewire 100]
       python experiments/analyze_amazon.py --loo        (leave-one-pair-out table)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ROOT, WORK, jsonable

P_REG = [
    ("Microphones & Accessories > Microphones", "Microphones & Accessories > Accessories"),
    ("Microphones & Accessories > Microphones", "Live Sound & Stage > Stage & Studio Cables"),
    ("Guitars > Acoustic Guitars", "Instrument Accessories > Guitar & Bass Accessories"),
    ("Guitars > Electric Guitars", "Instrument Accessories > Guitar & Bass Accessories"),
    ("Amplifiers & Effects > Guitar Effects", "Live Sound & Stage > Stage & Studio Cables"),
    ("Amplifiers & Effects > Guitar & Bass Amplifiers", "Instrument Accessories > Guitar & Bass Accessories"),
    ("Drums & Percussion > Hand Percussion", "Instrument Accessories > Drum & Percussion Accessories"),
    ("Drums & Percussion > Drum Sets & Set Components", "Instrument Accessories > Drum & Percussion Accessories"),
    ("Wind & Woodwind Instruments > Folk & World", "Instrument Accessories > Wind & Woodwind Accessories"),
    ("Band & Orchestra > Wind & Woodwinds", "Instrument Accessories > Wind & Woodwind Accessories"),
    ("Band & Orchestra > Orchestral Strings", "Instrument Accessories > Orchestral Strings Accessories"),
    ("Stringed Instruments > Folk & World", "Instrument Accessories > Folk & World Instrument Accessories"),
    ("Studio Recording Equipment > Computer Recording", "Microphones & Accessories > Microphones"),
]
MIN_CAT, N_NULL, K_NN, NULL_SEED = 5, 5000, 10, 18_000
MDE_FACTOR = 2.49            # z_0.95 + z_0.80


# ----------------------------------------------------------------------------- inputs and distortions
def load_runs(base, attributes, variant):
    out = {}
    for f in sorted(glob.glob(os.path.join(base, attributes, variant, "seed*_arrays.npz"))):
        z = np.load(f)
        out[int(os.path.basename(f)[4:7])] = dict(d1=squareform(z["d1"].astype(float)), d2=squareform(z["d2"].astype(float)),
                                                  major_z=z["major_z"], mom_major_z=z["mom_major_z"], Z1=z["Z1"].astype(float))
    return out


def centred(L, off):
    L = L.copy()
    L[~off] = np.nan
    return L - np.median(L[off & np.isfinite(L)])


def rel_log_distortion(D, D1, off):
    with np.errstate(divide="ignore", invalid="ignore"):
        return centred(np.log(D) - np.log(D1), off)


def capped(SP, plus=0.0):
    """disconnected pairs -> layer max finite (+ plus)"""
    SP = SP.copy()
    fin = np.isfinite(SP)
    SP[~fin] = SP[fin].max() + plus
    return SP


def hops(A):
    return capped(shortest_path(A.astype(float), directed=False, unweighted=True), plus=1.0)


# ----------------------------------------------------------------------------- corridor statistic
class Corridor:
    def __init__(self, labels, cats, partners, P_kept, md):
        self.lab = np.array(labels)
        self.cats, self.partners, self.P, self.md = cats, partners, P_kept, md
        self.ix = {c: np.flatnonzero(self.lab == c) for c in cats}

    def blocks(self, M, mask):
        out = {}
        for X in self.cats:
            for Y in self.cats:
                if X != Y:
                    sub = M[np.ix_(self.ix[X], self.ix[Y])]
                    ok = mask[np.ix_(self.ix[X], self.ix[Y])] & np.isfinite(sub)
                    out[(X, Y)] = float(np.median(sub[ok])) if ok.any() else np.nan
        return out

    def control(self, X, Y, excl):
        cand = [Z for Z in self.cats if Z not in (X, Y) and Z not in excl]
        return min(cand, key=lambda Z: (abs(self.md[(X, Z)] - self.md[(X, Y)]), Z))

    def did(self, MC, MS, X, Y, Yc):
        return np.array([(mc[(X, Y)] - mc[(X, Yc)]) - (ms[(X, Y)] - ms[(X, Yc)]) for mc, ms in zip(MC, MS)])   # per seed

    def T(self, MC, MS, mapping=None):
        m = (lambda c: c) if mapping is None else mapping.__getitem__
        per_pair, per_seed = {}, []
        for A, B in self.P:
            X, Y = m(A), m(B)
            Yc = self.control(X, Y, {m(Z) for Z in self.partners.get(A, ())})
            per_pair[(A, B, Yc)] = self.did(MC, MS, X, Y, Yc)
        D = np.array(list(per_pair.values()))                     # (pairs, seeds)
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(D, 1)
        return float(np.nanmean(avg)), np.nanmean(D, 0), per_pair, int(np.isfinite(avg).sum())

    def test(self, dC, dS, mask, rng, n_null=N_NULL):
        seeds = sorted(dC)
        MC = [self.blocks(dC[s], mask) for s in seeds]
        MS = [self.blocks(dS[s], mask) for s in seeds]
        T, T_seed, per_pair, nfin = self.T(MC, MS)
        null = np.array([self.T(MC, MS, dict(zip(self.cats, rng.permutation(self.cats))))[0] for _ in range(n_null)])
        null = null[np.isfinite(null)]
        # secondary null: random ordered pairs (distinct unordered pairs within a draw), symmetric partner exclusion
        Pu = {frozenset(p) for p in P_REG}
        pairs = [(X, Y) for X in self.cats for Y in self.cats if X != Y and frozenset((X, Y)) not in Pu]
        avg = {}
        for X, Y in pairs:
            with np.errstate(invalid="ignore"):
                avg[(X, Y)] = float(np.nanmean(self.did(MC, MS, X, Y, self.control(X, Y, self.partners.get(X, set())))))
        keys = sorted({frozenset(p) for p in pairs}, key=sorted)
        by = {k: [p for p in pairs if frozenset(p) == k] for k in keys}
        null2 = []
        for _ in range(n_null):
            vals = [avg[by[keys[k]][rng.integers(len(by[keys[k]]))]] for k in rng.choice(len(keys), len(self.P), replace=False)]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                null2.append(np.mean(vals))
        null2 = np.array(null2)
        return dict(T=T, p=float((1 + np.sum(null <= T)) / (len(null) + 1)), n_null=int(len(null)),
                    seeds_negative=int(np.sum(T_seed < 0)), n_seeds=len(seeds), T_per_seed=T_seed.tolist(), finite_P_pairs=nfin,
                    null_sd=float(null.std(ddof=1)), mde_80=float(MDE_FACTOR * null.std(ddof=1)),
                    null_q05=float(np.quantile(null, 0.05)), null_q50=float(np.median(null)),
                    p_random_pair_null=float((1 + np.sum(null2 <= T)) / (len(null2) + 1)),
                    per_pair={f"{A} || {B} (ctrl {Yc})": float(np.nanmean(v)) for (A, B, Yc), v in per_pair.items()})


# ----------------------------------------------------------------------------- leave one pair out
def leave_one_out(labels, cats, partners, P_kept, md, dC, dS, mask, n_null=2000):
    """Supplementary Table (leave-one-pair-out): drop each specified pair in turn and recompute T and its
    category-relabelling null (n_null relabellings, seed NULL_SEED + 100 + i for the i-th dropped pair)."""
    seeds = sorted(dC)
    out = []
    for i, p in enumerate(P_kept):
        cor = Corridor(labels, cats, partners, [q for q in P_kept if q != p], md)
        MC = [cor.blocks(dC[s], mask) for s in seeds]
        MS = [cor.blocks(dS[s], mask) for s in seeds]
        T = cor.T(MC, MS)[0]
        rng = np.random.default_rng(NULL_SEED + 100 + i)
        null = np.array([cor.T(MC, MS, dict(zip(cats, rng.permutation(cats))))[0] for _ in range(n_null)])
        null = null[np.isfinite(null)]
        out.append(dict(dropped=f"{p[0]} || {p[1]}", T=T, p=float((1 + np.sum(null <= T)) / (len(null) + 1)),
                        q05=float(np.quantile(null, 0.05))))
    return out


# ----------------------------------------------------------------------------- helpers
def knn_agreement(D, labels):
    lab = np.array(labels)
    ok = lab != ""
    Dm = D[np.ix_(ok, ok)].copy()
    np.fill_diagonal(Dm, np.inf)
    nn = np.argsort(Dm, 1, kind="stable")[:, :K_NN]
    L = lab[ok]
    hit = (L[nn] == L[:, None]).mean(1)
    return float(hit.mean()), float(np.mean([hit[L == c].mean() for c in np.unique(L)]))


def paired(a, b):
    d = np.array(a) - np.array(b)
    t = stats.ttest_1samp(d, 0.0, alternative="greater")
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean_diff=float(d.mean()), ci95=[float(d.mean() - h), float(d.mean() + h)], p=float(t.pvalue),
                seeds_positive=int((d > 0).sum()), n=len(d))


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def rewire(A, rng_seed):
    import networkx as nx
    G = nx.from_numpy_array(A.astype(int))
    m = G.number_of_edges()
    nx.double_edge_swap(G, nswap=10 * m, max_tries=1000 * m, seed=rng_seed)
    return nx.to_numpy_array(G, nodelist=range(A.shape[0])) > 0


def config_enrichment(A, labels, P_kept):
    lab = np.array(labels)
    deg = A.sum(1)
    m2 = deg.sum()
    E = np.outer(deg, deg) / m2
    inP = np.zeros_like(A, bool)
    for a, b in P_kept:
        inP |= np.outer(lab == a, lab == b) | np.outer(lab == b, lab == a)
    iu = np.triu(np.ones_like(A, bool), 1)
    lab_ok = (lab[:, None] != "") & (lab[None, :] != "") & (lab[:, None] != lab[None, :])
    r = lambda M: float(A[M & iu].sum() / E[M & iu].sum())
    return dict(P_blocks=r(inP), other_cross_category=r(lab_ok & ~inP))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(WORK, "results", "amazon"))
    ap.add_argument("--data", default=os.path.join(WORK, "data", "Amazon", "Musical_Instruments"))
    ap.add_argument("--rewire", type=int, default=100)
    ap.add_argument("--loo", action="store_true", help="only the leave-one-pair-out table (writes loo_pairs.json)")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    labels = meta["level3"]
    lab = np.array(labels)
    S, C, Cf, C2 = (np.load(os.path.join(args.data, f"A_{k}.npy")) > 0 for k in ("S", "C", "C_full", "C2"))
    n = S.shape[0]
    off = ~np.eye(n, dtype=bool)
    unlinked = off & ~S & ~C & ~Cf
    cats = sorted(c for c in set(labels) if c and np.sum(lab == c) >= MIN_CAT)
    P_kept = [p for p in P_REG if p[0] in cats and p[1] in cats]
    partners = {}
    for A_, B_ in P_REG:                                          # symmetric
        partners.setdefault(A_, set()).add(B_)
        partners.setdefault(B_, set()).add(A_)
    R = dict(analysis="product network", n=n, eligible_categories=len(cats),
             P_kept=[list(p) for p in P_kept], P_dropped=[list(p) for p in P_REG if p not in P_kept])
    ix = {c: np.flatnonzero(lab == c) for c in cats}
    R["P_link_counts"] = {f"{a} || {b}": dict(C=int(C[np.ix_(ix[a], ix[b])].sum()), S=int(S[np.ix_(ix[a], ix[b])].sum()),
                                              C_full=int(Cf[np.ix_(ix[a], ix[b])].sum())) for a, b in P_kept}

    runs = {v: load_runs(args.base, "text", v) for v in ("S", "C", "S_boot", "C_boot")}
    seeds = sorted(set.intersection(*[set(r) for r in runs.values()]))
    R["seeds"] = seeds
    for s in seeds:
        assert all(np.allclose(runs[v][s]["d1"], runs["S"][s]["d1"]) for v in runs), s
    d1 = {s: runs["S"][s]["d1"] for s in seeds}
    d1n = np.mean([d1[s] / np.median(d1[s][off]) for s in seeds], 0)
    delta = {v: {s: rel_log_distortion(runs[v][s]["d2"], d1[s], off) for s in seeds} for v in runs}

    def md_for(mask, D=d1n):
        cor = Corridor(labels, cats, partners, P_kept, None)
        return cor.blocks(D, mask)
    COR_all = Corridor(labels, cats, partners, P_kept, md_for(off))
    COR_unl = Corridor(labels, cats, partners, P_kept, md_for(unlinked))      # controls matched on the mask
    if args.loo:
        L = leave_one_out(labels, cats, partners, P_kept, md_for(off), delta["C"], delta["S"], off)
        json.dump(jsonable(L), open(os.path.join(args.base, "loo_pairs.json"), "w"), indent=1)
        print(json.dumps(jsonable(L), indent=1))
        return

    # ---- H1 manipulation check and layer facts
    iu = np.triu(off)
    ratio = [float(np.median(d1[s][C & iu]) / np.median(d1[s][S & iu])) for s in seeds]

    def same_cat(A):
        i, j = np.where(np.triu(A, 1))
        ok = (lab[i] != "") & (lab[j] != "")
        return float(np.mean(lab[i][ok] == lab[j][ok]))
    R["H1_manipulation_check"] = dict(ratio_median_d1_C_over_S=float(np.mean(ratio)), per_seed=ratio, passes=bool(np.mean(ratio) > 1),
                                      same_category_share_S=same_cat(S), same_category_share_C=same_cat(C),
                                      share_S_links_in_also_bought=float((S & Cf & iu).sum() / (S & iu).sum()),
                                      share_S_links_bought_together=float((S & C2 & iu).sum() / (S & iu).sum()),
                                      config_enrichment_C=config_enrichment(C, labels, P_kept),
                                      config_enrichment_S=config_enrichment(S, labels, P_kept))

    # ---- H2, H2u (primary null: relabelling)
    R["H2"] = COR_all.test(delta["C"], delta["S"], off, np.random.default_rng(NULL_SEED))
    R["H2u"] = COR_unl.test(delta["C"], delta["S"], unlinked, np.random.default_rng(NULL_SEED + 1))

    # ---- graph reference row: log hop count, no geometry; degree-preserving rewirings
    hC, hS = np.log(hops(C)), np.log(hops(S))
    hopC, hopS = {s: centred(hC, off) for s in seeds}, {s: centred(hS, off) for s in seeds}
    R["graph_reference_hop"] = dict(all=COR_all.test(hopC, hopS, off, np.random.default_rng(NULL_SEED + 2)),
                                    unlinked=COR_unl.test(hopC, hopS, unlinked, np.random.default_rng(NULL_SEED + 3)))
    for k in ("all", "unlinked"):
        R["graph_reference_hop"][k].pop("T_per_seed")
    Trw = []
    one = [seeds[0]]
    for r in range(args.rewire):
        Cr, Sr = rewire(C, 18_100 + r), rewire(S, 28_100 + r)
        hc, hs = {one[0]: centred(np.log(hops(Cr)), off)}, {one[0]: centred(np.log(hops(Sr)), off)}
        MC, MS = [COR_all.blocks(hc[one[0]], off)], [COR_all.blocks(hs[one[0]], off)]
        Trw.append(COR_all.T(MC, MS)[0])
    Trw = np.array(Trw)
    R["degree_rewiring_control"] = dict(hop_T_observed=R["graph_reference_hop"]["all"]["T"], rewired_q05=float(np.quantile(Trw, 0.05)),
                                        rewired_q50=float(np.median(Trw)), n=len(Trw),
                                        passes=bool(R["graph_reference_hop"]["all"]["T"] < np.quantile(Trw, 0.05)))

    # ---- H3 label transfer
    ag = {k: [] for k in ("d1", "S", "S_boot", "C", "C_boot", "graph_S_hop", "graph_sp_d1_S")}
    for s in seeds:
        ag["d1"].append(knn_agreement(d1[s], labels))
        for v in ("S", "S_boot", "C", "C_boot"):
            ag[v].append(knn_agreement(runs[v][s]["d2"], labels))
        ag["graph_S_hop"].append(knn_agreement(hops(S) + 1e-3 * d1[s] / d1[s][off].max(), labels))
        ag["graph_sp_d1_S"].append(knn_agreement(capped(shortest_path(np.where(S, d1[s], 0.0), directed=False)), labels))
    R["knn_agreement"] = {k: dict(micro=float(np.mean([a for a, _ in v])), macro=float(np.mean([b for _, b in v]))) for k, v in ag.items()}
    mic = {k: [a for a, _ in v] for k, v in ag.items()}
    R["H3a"] = paired(mic["S"], mic["S_boot"])
    R["H3b"] = paired(mic["S"], mic["d1"])
    R["H3c"] = paired(mic["S"], mic["C"])
    R["H3c"]["note"] = "manipulation check"

    fam = ["H2", "H2u", "H3a", "H3b", "H3c"]
    for h, a in zip(fam, holm(np.array([R[h]["p"] for h in fam]))):
        R[h]["p_holm"] = float(a)
    for h in ("H2", "H2u"):
        R[h]["supported"] = bool(R[h]["p_holm"] < 0.05 and R[h]["T"] < 0 and R[h]["seeds_negative"] >= 8)
    for h in ("H3a", "H3b", "H3c"):
        R[h]["supported"] = bool(R[h]["p_holm"] < 0.05 and R[h]["seeds_positive"] >= 8)

    # ---- gate: popularity-only attributes
    pop = {v: load_runs(args.base, "popularity", v) for v in ("S", "C")}
    ps_ = sorted(set(pop["S"]) & set(pop["C"]))
    if ps_:
        d1p = {s: pop["S"][s]["d1"] for s in ps_}
        dp = {v: {s: rel_log_distortion(pop[v][s]["d2"], d1p[s], off) for s in ps_} for v in pop}
        d1pn = np.mean([d1p[s] / np.median(d1p[s][off]) for s in ps_], 0)
        CORp = Corridor(labels, cats, partners, P_kept, Corridor(labels, cats, partners, P_kept, None).blocks(d1pn, off))
        g = CORp.test(dp["C"], dp["S"], off, np.random.default_rng(NULL_SEED + 4))
        fires = bool(g["p"] < 0.05 or (R["H2"]["T"] < 0 and g["T"] <= 0.5 * R["H2"]["T"]))
        R["gate_popularity"] = dict(T=g["T"], p=g["p"], seeds=ps_, fires=fires, note="low-powered")
    else:
        R["gate_popularity"] = dict(fires=None, note="no popularity runs found")

    R["licence_carries_complement_structure"] = bool(R["H1_manipulation_check"]["passes"] and R["H2"]["supported"] and R["H2u"]["supported"]
                                                     and R["gate_popularity"].get("fires") is False and R["degree_rewiring_control"]["passes"])
    R["licence_label_transfer"] = bool(R["H3a"]["supported"])

    # ---- comparisons: shortest-path and hop distortion, capped
    C1 = {}
    for name in ("sp", "hop"):
        dc, ds = {}, {}
        for s in seeds:
            for A, store in ((C, dc), (S, ds)):
                SP = capped(shortest_path(np.where(A, d1[s], 0.0), directed=False)) if name == "sp" else hops(A)
                store[s] = rel_log_distortion(SP, d1[s], off)
        C1[name] = dict(all=COR_all.test(dc, ds, off, np.random.default_rng(NULL_SEED + 5)),
                        unlinked=COR_unl.test(dc, ds, unlinked, np.random.default_rng(NULL_SEED + 6)))
        for k in ("all", "unlinked"):
            C1[name][k].pop("per_pair")
    R["comparisons_graph_distortion"] = C1

    # ---- direction (descriptive; undefined axes dropped)
    def folded(u, v):
        c = np.abs(u @ v) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v) + 1e-300)
        return np.degrees(np.arccos(np.clip(c, 0, 1)))
    dirs = {k: [] for k in ("C_twist", "S_twist", "C_moment", "S_moment")}
    for s in seeds:
        for A_, B_ in P_kept:
            nodes = np.r_[ix[A_], ix[B_]]
            Z = runs["C"][s]["Z1"]
            axis = Z[ix[B_]].mean(0) - Z[ix[A_]].mean(0)
            for key, v, fld in (("C_twist", "C", "major_z"), ("S_twist", "S", "major_z"), ("C_moment", "C", "mom_major_z"), ("S_moment", "S", "mom_major_z")):
                u = runs[v][s][fld][nodes]
                u = u[np.all(np.isfinite(u), 1) & (np.linalg.norm(u, axis=1) > 0)]
                if len(u):
                    dirs[key].append(float(np.mean(folded(u, axis))))
    R["direction_descriptive"] = {k: float(np.mean(v)) for k, v in dirs.items()}
    R["direction_descriptive"]["note"] = "folded angle in [0, 90]; uniform-chance mean 45; descriptive"

    json.dump(jsonable(R), open(os.path.join(args.base, "analysis.json"), "w"), indent=1)
    show = {k: R[k] for k in ("H1_manipulation_check", "gate_popularity", "degree_rewiring_control",
                              "licence_carries_complement_structure", "licence_label_transfer", "knn_agreement")}
    for h in fam:
        show[h] = {k: v for k, v in R[h].items() if k not in ("per_pair", "T_per_seed")}
    show["graph_reference_hop"] = {k: {kk: R["graph_reference_hop"][k][kk] for kk in ("T", "p", "null_sd")} for k in ("all", "unlinked")}
    show["comparisons_graph_distortion"] = {nm: {k: {kk: v[k][kk] for kk in ("T", "p", "finite_P_pairs")} for k in v} for nm, v in C1.items()}
    print(json.dumps(jsonable(show), indent=1))


if __name__ == "__main__":
    main()
