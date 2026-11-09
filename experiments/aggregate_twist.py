"""
Aggregation of the planted-disagreement experiment (Fig. 3, Supplementary Section S4).

Estimands
  E1  planted strain G_attr^{-1} G_graph (arrays t_* written by run_twist.py); region = nodes with phi >= 0.5
  E2  strain of the graph geometry relative to the LEARNED attribute geometry, G1^{-1} G_graph (pulled to latent
      coordinates with the exact encoder Jacobian). In design "none", E2 = Phase 1's own metric error.

Strain estimators
  raw          the twist G1^{-1} G2                                                              PRIMARY
  boot         the twist of the bootstrap graph of "none" (carries no information on Phase 1's error)
  moment       training-free link-moment tensor: area = 0.5 (log det C0 - log det C),
               anisotropy from eig(C0, C), axis = major axis of the G1-symmetrised C0 C^{-1} (NaN where indefinite);
               C and C0 pooled over the node and its 10 nearest latent neighbours before the invariants
  oracle       G2_none^{-1} G2 (needs the truth; descriptive reference only)

Paired references: every test compares a quantity with the same quantity for a reference estimator that saw the
same nodes and Phase 1 but not the information under test:
  * design "none": the bootstrap twist (H1-H3);
  * twisted designs: the "none" twist, which carries the repair of Phase 1 but not the planted twist (H1, H2, H4,
    H5b, H6, H7).
Tests: paired Wilcoxon signed-rank over seeds, Holm over the family (31 rows; H2[shear] is null by construction
because the shear preserves area).

usage: python experiments/aggregate_twist.py --tag twist [--arms bce1,mse15] [--blocks twist_blocks]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import scipy.linalg as sla
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aligning import baselines as B
from aligning import heat
from aligning.evaluation import summarize, paired_test, holm
from aligning.strain import axis_error_deg, generalized_eig, invariants, link_moments
from aligning.synthetic import ManifoldSpec
from aligning.twist import TwistSpec, true_strain, graph_metric_fn
from experiments.common import ROOT, WORK, jsonable

DESIGNS = ("shear", "barrier", "corridor")
REGION_PHI = 0.5
E2_THR = 0.25          # nodes whose target anisotropy is at least 0.25 (eigenvalue ratio >= 1.65)
MIN_NODES = 10         # fewer selected nodes -> NaN
LOCAL_Q = 0.05         # pair endpoint: attribute-local pairs
FID_Q = 0.10           # fidelity endpoints: local pairs
N_STRATA = 10
ROT = np.radians(np.arange(5, 180, 5))
MOMENT_K = 10


# ----------------------------------------------------------------------------- helpers
def strained(Gnum, Gden):
    L, V = generalized_eig(Gnum, Gden)
    inv = invariants(L)
    return dict(aniso=inv["aniso"], area=inv["area"], major_z=V[:, :, 0])


def g_angle(v, w, G):
    """Angle (deg, [0, 90]) between latent directions v and w in the metric G (per node)."""
    num = np.abs(np.einsum("ni,nij,nj->n", v, G, w))
    den = np.sqrt(np.einsum("ni,nij,nj->n", v, G, v) * np.einsum("ni,nij,nj->n", w, G, w))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arccos(np.clip(num / den, 0.0, 1.0)))


def learned_relative_truth(manifold, tw, Zt, J, G1):
    """E2 = G1^{-1} G_graph with G_graph pulled to latent coordinates (J^{-T} G_graph(u) J^{-1})."""
    with torch.no_grad():
        Gg = graph_metric_fn(manifold, tw)(torch.as_tensor(Zt, dtype=torch.float64)).numpy()
    Ji = np.linalg.inv(J)
    Gz = np.einsum("nji,njk,nkl->nil", Ji, Gg, Ji)
    return strained(Gz, G1)


def moment_invariants(Z, C, C0, G1, deg, smooth_k=MOMENT_K):
    """Link-moment competitor. See the module docstring."""
    n = Z.shape[0]
    ok = deg >= 2
    Cp, C0p = C, C0
    if smooth_k:
        _, nn = cKDTree(Z).query(Z, k=min(smooth_k + 1, n))
        w = ok.astype(float)[nn]
        Cp = (C[nn] * w[..., None, None]).sum(1) / np.maximum(w.sum(1), 1)[:, None, None]
        C0p = C0[nn].mean(1)
        ok = w.sum(1) >= 1
    area = np.full(n, np.nan)
    aniso = np.full(n, np.nan)
    major = np.full((n, 2), np.nan)
    for i in np.nonzero(ok)[0]:
        dc, d0 = np.linalg.det(Cp[i]), np.linalg.det(C0p[i])
        if not (dc > 0 and d0 > 0):
            continue
        lam = sla.eigh(C0p[i], Cp[i], eigvals_only=True)
        if np.any(lam <= 0):
            continue
        area[i] = 0.5 * (np.log(d0) - np.log(dc))
        aniso[i] = 0.5 * np.log(lam.max() / lam.min())
        W = sla.sqrtm(G1[i]).real
        Sm = W @ (C0p[i] @ np.linalg.inv(Cp[i])) @ np.linalg.inv(W)
        ev, evec = np.linalg.eigh(0.5 * (Sm + Sm.T))
        if ev.min() > 0:
            major[i] = np.linalg.solve(W, evec[:, -1])
    return dict(area=area, aniso=aniso, major_z=major)


def _sq(v, n):
    M = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    M[iu] = v
    return M + M.T


def moment_of(a, meta_d, Z1, G1):
    """Recompute the corrected moment competitor from the saved arrays of one graph."""
    A = a["A"].astype(float)
    n = A.shape[0]
    D1 = _sq(a["d1"].astype(float), n)
    t1 = heat.heat_times(A, 1, 4.0)[0]
    s0, a0 = meta_d["moment_calibration"]["scale"], meta_d["moment_calibration"]["amp"]
    P1 = a0 * np.exp(-((s0 * D1) ** 2) / (4.0 * t1))
    np.fill_diagonal(P1, 0.0)
    nb = (A > 0).astype(float)
    return moment_invariants(Z1, link_moments(Z1, nb), link_moments(Z1, P1), G1, nb.sum(1))


def stratified_spearman(x, y, s, k=N_STRATA):
    """Pairs are split into k quantile strata of s; Spearman in each stratum with
    >= 10 pairs and non-constant x and y; size-weighted mean; NaN below 5k pairs."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    x, y, s = x[ok], y[ok], s[ok]
    if x.size < 5 * k:
        return np.nan
    edges = np.quantile(s, np.linspace(0, 1, k + 1))
    b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, k - 1)
    rs, ws = [], []
    for j in range(k):
        m = b == j
        if m.sum() >= 10 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
            rs.append(spearmanr(x[m], y[m]).correlation)
            ws.append(m.sum())
    return float(np.average(rs, weights=ws)) if rs else np.nan


def fidelity(d, ref, q=FID_Q):
    loc = ref <= np.quantile(ref, q)
    return float(spearmanr(d, ref).correlation), float(spearmanr(d[loc], ref[loc]).correlation)


def auroc(y, s):
    s = np.asarray(s, float)
    s = np.where(np.isfinite(s), s, np.nanmedian(s))
    return float(roc_auc_score(y, s))


def dir_err(est, ref, G, sel):
    return float(np.nanmedian(g_angle(est["major_z"], ref["major_z"], G)[sel])) if sel.sum() >= MIN_NODES else np.nan


def rho_on(x, y, sel):
    ok = sel & np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() >= MIN_NODES else np.nan


def load_seed(sd):
    p1 = torch.load(os.path.join(sd, "phase1.pt"), weights_only=False)
    arr = {os.path.basename(f)[:-len("_arrays.npz")]: dict(np.load(f)) for f in glob.glob(os.path.join(sd, "*_arrays.npz"))}
    meta = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(sd, "*.json")) if "FAILED" not in f}
    return p1, arr, meta


# ----------------------------------------------------------------------------- per seed
def per_seed(sd: str) -> dict:
    p1, arr, meta = load_seed(sd)
    seed = int(os.path.basename(sd)[4:])
    J, Z1, Zt = p1["J_dzdu"], p1["Z1"], p1["Z_true"]
    manifold = ManifoldSpec.sample(np.random.default_rng(10_000 + seed), 20)
    out = {}
    if "none" not in arr:
        return out
    nu, nb = arr["none"], arr.get("none_boot")
    G1 = nu["G1"]
    n = Z1.shape[0]
    raw_none = strained(nu["G2"], G1)
    mom_none = moment_of(nu, meta["none"], Z1, G1)
    cond = np.linalg.cond(J)
    # ------------------------------------------------ design "none": recovery of Phase 1's error, repair
    r = dict(frac_cond_J_gt_100=float(np.mean(cond > 100)))
    e2n = learned_relative_truth(manifold, TwistSpec("none"), Zt, J, G1)
    seln = e2n["aniso"] >= E2_THR
    alln = np.ones(n, bool)
    r["n_e2_selected"] = int(seln.sum())
    r["e2_raw_dir"] = dir_err(raw_none, e2n, G1, seln)
    r["e2_raw_area_rho"] = rho_on(raw_none["area"], e2n["area"], alln)
    r["e2_moment_dir"] = dir_err(mom_none, e2n, G1, seln)
    r["e2_moment_area_rho"] = rho_on(mom_none["area"], e2n["area"], alln)
    da = nu["d_attr"].astype(float)
    r["fid_attr_d1_all"], r["fid_attr_d1_loc"] = fidelity(nu["d1"].astype(float), da)
    r["fid_attr_d2_all"], r["fid_attr_d2_loc"] = fidelity(nu["d2"].astype(float), da)
    r["loss_drop"] = float(1 - meta["none"]["phase2_summary"]["loss_p2"] / meta["none"]["phase2_summary"]["loss_p1"])
    if nb is not None:
        boot = strained(nb["G2"], G1)
        mom_boot = moment_of(nb, meta["none_boot"], Z1, G1)
        r["e2_boot_dir"] = dir_err(boot, e2n, G1, seln)
        r["e2_boot_area_rho"] = rho_on(boot["area"], e2n["area"], alln)
        r["e2_moment_boot_dir"] = dir_err(mom_boot, e2n, G1, seln)
        r["e2_moment_boot_area_rho"] = rho_on(mom_boot["area"], e2n["area"], alln)
        r["fid_attr_d2boot_all"], r["fid_attr_d2boot_loc"] = fidelity(nb["d2"].astype(float), da)
        r["loss_drop_boot"] = float(1 - meta["none_boot"]["phase2_summary"]["loss_p2"] / meta["none_boot"]["phase2_summary"]["loss_p1"])
    out["none"] = r
    # ------------------------------------------------ twisted designs
    iu = np.triu_indices(n, 1)
    for d in DESIGNS:
        if d not in arr:
            continue
        a = arr[d]
        tw = TwistSpec(**meta[d]["twist"])
        region = a["t_phi"] >= REGION_PHI
        y = region.astype(int)
        Ga, tax = a["G_attr"], a["t_major_u"]
        raw = strained(a["G2"], G1)
        mom = moment_of(a, meta[d], Z1, G1)
        oracle = strained(a["G2"], nu["G2"])
        o = dict(n_region=int(region.sum()), angle=float(tw.angle))
        # E2 restricted to the region, against the "none" twist (H1, H2) and the moment (C1, C2)
        e2 = learned_relative_truth(manifold, tw, Zt, J, G1)
        sel = region & (e2["aniso"] >= E2_THR)
        o["n_e2_region_selected"] = int(sel.sum())
        for k, est in dict(raw=raw, none=raw_none, moment=mom, moment_none=mom_none).items():
            o[f"e2r_{k}_dir"] = dir_err(est, e2, G1, sel)
            o[f"e2r_{k}_area_rho"] = rho_on(est["area"], e2["area"], region)
        # toward the graph geometry (H4), paired with the "none" twist
        dg = a["d_graph"].astype(float)
        o["fid_graph_d1_all"], o["fid_graph_d1_loc"] = fidelity(a["d1"].astype(float), dg)
        o["fid_graph_d2_all"], o["fid_graph_d2_loc"] = fidelity(a["d2"].astype(float), dg)
        o["fid_graph_d2none_all"], o["fid_graph_d2none_loc"] = fidelity(nu["d2"].astype(float), dg)
        # planted strain E1: direction (H5a vs rotation chance, H5b vs the "none" twist), localisation (H6)
        decoys = [true_strain(manifold, TwistSpec(kind=tw.kind, kappa=tw.kappa, centre=tw.centre, angle=tw.angle + r_,
                                                   width=tw.width), Zt)["major_u"] for r_ in ROT]
        for k, est, ref in (("raw", raw, raw_none), ("moment", mom, mom_none), ("oracle", oracle, None)):
            e = axis_error_deg(est["major_z"], J, tax, Ga)
            o[f"dir_{k}"] = float(np.nanmedian(e[region]))
            o[f"dir_{k}_chance"] = float(np.mean([np.nanmedian(axis_error_deg(est["major_z"], J, dc, Ga)[region]) for dc in decoys]))
            o[f"loc_{k}"] = auroc(y, est["aniso"])
            o[f"loc_area_{k}"] = auroc(y, np.abs(est["area"] - np.nanmedian(est["area"])))
            if ref is not None:
                o[f"dir_{k}_null"] = float(np.nanmedian(axis_error_deg(ref["major_z"], J, tax, Ga)[region]))
                o[f"loc_{k}_null"] = auroc(y, ref["aniso"])
                o[f"loc_area_{k}_null"] = auroc(y, np.abs(ref["area"] - np.nanmedian(ref["area"])))
        o["loc_raw_abs"] = abs(o["loc_raw"] - 0.5)                       # descriptive
        o["loc_raw_null_abs"] = abs(o["loc_raw_null"] - 0.5)
        for k, s, s0 in (("direct", a["p1_edge_length"], nu["p1_edge_length"]),
                         ("degree", B.degree_deviation(a["A"]), B.degree_deviation(nu["A"])),
                         ("cos_homophily", a["local_cosine_homophily"], nu["local_cosine_homophily"])):
            o[f"loc_{k}"], o[f"loc_{k}_null"] = auroc(y, s), auroc(y, s0)
        # pair distortion (H7): delta against the truth, paired with the "none" twist
        dA, d1 = a["d_attr"].astype(float), a["d1"].astype(float)
        psel = (region[iu[0]] | region[iu[1]]) & (dA <= np.quantile(dA, LOCAL_Q))
        truth = np.log(dg) - np.log(dA)
        strat = np.log(d1[psel])
        pairs = dict(raw=(np.log(a["d2"]) - np.log(a["d1"]), np.log(nu["d2"]) - np.log(nu["d1"])),
                     sp_d1=(np.log(np.maximum(a["sp_d1"], a["d1"])) - np.log(a["d1"]), np.log(np.maximum(nu["sp_d1"], nu["d1"])) - np.log(nu["d1"])),
                     hop=(np.log(a["hop"]) - np.log(a["d1"]), np.log(nu["hop"]) - np.log(nu["d1"])),
                     oracle=(np.log(a["d2"]) - np.log(nu["d2"]), None))
        for k, (e, e0) in pairs.items():
            o[f"pair_{k}"] = stratified_spearman(e[psel].astype(float), truth[psel], strat)
            if e0 is not None:
                o[f"pair_{k}_null"] = stratified_spearman(e0[psel].astype(float), truth[psel], strat)
        o["n_pairs"] = int(psel.sum())
        o["epochs"] = meta[d]["phase2_summary"]["epochs"]
        o["max_epochs"] = meta[d]["phase2"]["epochs"]
        out[d] = o
    return out


def collect(tag: str, arm: str):
    rows = {d: [] for d in ("none",) + DESIGNS}
    for sd in sorted(glob.glob(os.path.join(WORK, "results", "synthetic", tag, arm, "seed*"))):
        if not os.path.exists(os.path.join(sd, "phase1.pt")):
            continue
        for d, v in per_seed(sd).items():
            rows[d].append(dict(v, seed=int(os.path.basename(sd)[4:])))
    return rows


# ----------------------------------------------------------------------------- block designs (H8, H9)
def block_endpoints(tag: str):
    base = os.path.join(WORK, "results", "synthetic", tag)
    rows = []
    for sd in sorted(glob.glob(os.path.join(base, "seed*"))):
        r = dict(seed=int(os.path.basename(sd)[4:]))
        for c in ("D70", "D70N", "BAND", "BANDN", "R70", "R70N"):
            f = os.path.join(sd, f"{c}.json")
            if os.path.exists(f):
                m = json.load(open(f))["metrics"]
                r[f"{c}_edge_contraction"] = m["distortion_edge_contraction"]["auroc"]
                r[f"{c}_direct"] = m["p1_edge_length"]["auroc"]
        p1f = os.path.join(sd, "phase1.npz")
        for c in ("R70N", "D70N", "BANDN"):
            fa = os.path.join(sd, f"{c}_arrays.npz")
            if os.path.exists(fa) and os.path.exists(p1f):
                dt = np.load(p1f)["d_true"].astype(float)
                a = np.load(fa)
                r[f"{c}_fid_d1_loc"] = fidelity(a["d1"].astype(float), dt)[1]
                r[f"{c}_fid_d2_loc"] = fidelity(a["d2"].astype(float), dt)[1]
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------- family of tests
def _diff(a, b):
    return [x - y for x, y in zip(a, b)]


def _one_sample(x, ref):
    return paired_test(x, [ref] * len(x))


def family(rows: dict, block_rows: list) -> dict:
    col = lambda d, k: [r.get(k, np.nan) for r in rows.get(d, [])]
    H, C = {}, {}
    # design none: recovery of Phase 1's error and repair, against the bootstrap twist
    H["H1 E2 direction: twist < bootstrap twist [none]"] = (paired_test(col("none", "e2_raw_dir"), col("none", "e2_boot_dir")), "lt")
    H["H2 E2 area: twist > bootstrap twist [none]"] = (paired_test(col("none", "e2_raw_area_rho"), col("none", "e2_boot_area_rho")), "gt")
    H["H3 repair: twist > bootstrap twist [none]"] = (paired_test(col("none", "fid_attr_d2_loc"), col("none", "fid_attr_d2boot_loc")), "gt")
    C["C1 E2 direction gain: twist - moment [none]"] = paired_test(_diff(col("none", "e2_raw_dir"), col("none", "e2_boot_dir")),
                                                                    _diff(col("none", "e2_moment_dir"), col("none", "e2_moment_boot_dir")))
    C["C2 E2 area gain: twist - moment [none]"] = paired_test(_diff(col("none", "e2_raw_area_rho"), col("none", "e2_boot_area_rho")),
                                                               _diff(col("none", "e2_moment_area_rho"), col("none", "e2_moment_boot_area_rho")))
    for d in DESIGNS:
        H[f"H1 E2 direction in region: twist < none twist [{d}]"] = (paired_test(col(d, "e2r_raw_dir"), col(d, "e2r_none_dir")), "lt")
        if d != "shear":   # the shear preserves area, so H2[shear] is null by construction
            H[f"H2 E2 area in region: twist > none twist [{d}]"] = (paired_test(col(d, "e2r_raw_area_rho"), col(d, "e2r_none_area_rho")), "gt")
        H[f"H4 toward graph geometry: twist > none twist [{d}]"] = (paired_test(col(d, "fid_graph_d2_loc"), col(d, "fid_graph_d2none_loc")), "gt")
        H[f"H5a planted direction < rotation chance [{d}]"] = (paired_test(col(d, "dir_raw"), col(d, "dir_raw_chance")), "lt")
        H[f"H5b planted direction < none twist [{d}]"] = (paired_test(col(d, "dir_raw"), col(d, "dir_raw_null")), "lt")
        H[f"H6 planted localisation > none twist [{d}]"] = (paired_test(col(d, "loc_raw"), col(d, "loc_raw_null")), "gt")
        H[f"H7 planted pair distortion > none twist [{d}]"] = (paired_test(col(d, "pair_raw"), col(d, "pair_raw_null")), "gt")
        C[f"C1 E2 direction gain: twist - moment [{d}]"] = paired_test(_diff(col(d, "e2r_raw_dir"), col(d, "e2r_none_dir")),
                                                                        _diff(col(d, "e2r_moment_dir"), col(d, "e2r_moment_none_dir")))
        C[f"C2 E2 area gain: twist - moment [{d}]"] = paired_test(_diff(col(d, "e2r_raw_area_rho"), col(d, "e2r_none_area_rho")),
                                                                   _diff(col(d, "e2r_moment_area_rho"), col(d, "e2r_moment_none_area_rho")))
        for b in ("sp_d1", "hop"):
            C[f"C3 pair gain: twist - {b} [{d}]"] = paired_test(_diff(col(d, "pair_raw"), col(d, "pair_raw_null")),
                                                                _diff(col(d, f"pair_{b}"), col(d, f"pair_{b}_null")))
        C[f"C4 localisation gain: twist - direct [{d}]"] = paired_test(_diff(col(d, "loc_raw"), col(d, "loc_raw_null")),
                                                                       _diff(col(d, "loc_direct"), col(d, "loc_direct_null")))
    bc = lambda k: [r.get(k, np.nan) for r in block_rows]
    for d in ("D70", "BAND", "R70"):
        H[f"H8 edge contraction > paired null [{d}]"] = (paired_test(bc(f"{d}_edge_contraction"), bc(f"{d}N_edge_contraction")), "gt")
        H[f"H8 edge contraction > 0.5 [{d}]"] = (_one_sample(bc(f"{d}_edge_contraction"), 0.5), "gt")
    for d in ("D70N", "BANDN"):
        H[f"H9 repair (local) [{d}]"] = (paired_test(bc(f"{d}_fid_d2_loc"), bc(f"{d}_fid_d1_loc")), "gt")
    # Holm over the full family: a NaN p (missing data) counts as p = 1 and keeps its place
    pH = {k: (v[0]["p"] if np.isfinite(v[0]["p"]) else 1.0) for k, v in H.items()}
    pC = {k: (v["p"] if np.isfinite(v["p"]) else 1.0) for k, v in C.items()}
    adjH, adjC = holm(pH), holm(pC)
    out = {"hypotheses": {}, "comparisons": {}, "family_size": len(H), "comparison_family_size": len(C)}
    for k, (t, side) in H.items():
        ok_side = (t["ci_hi"] < 0) if side == "lt" else (t["ci_lo"] > 0)
        out["hypotheses"][k] = dict(t, p_holm=adjH[k], predicted=side,
                                    supported=bool(np.isfinite(t["p"]) and ok_side and adjH[k] < 0.05))
    for k, t in C.items():
        out["comparisons"][k] = dict(t, p_holm=adjC[k])
    # which conclusions each set of results supports
    def sup(row, designs):
        return all(out["hypotheses"].get(f"{row} [{d}]", {}).get("supported", False) for d in designs)
    out["licences"] = {
        # T1 = H5a and H5b, T2 = H6, T3 = H7, each in all three designs
        "thesis (H5a, H5b, H6, H7 in shear, barrier and corridor)": bool(
            sup("H5a planted direction < rotation chance", DESIGNS) and sup("H5b planted direction < none twist", DESIGNS)
            and sup("H6 planted localisation > none twist", DESIGNS) and sup("H7 planted pair distortion > none twist", DESIGNS)),
        "beyond repair (H1 and H4 in all three twisted designs)": bool(
            sup("H1 E2 direction in region: twist < none twist", DESIGNS) and sup("H4 toward graph geometry: twist > none twist", DESIGNS)),
        "repair (H1, H2, H3 in design none)": bool(all(v["supported"] for k, v in out["hypotheses"].items() if k.endswith("[none]"))),
    }
    return out


def t4(rows_b, rows_m):
    """Prediction T4: the likelihood advantage in the planted pair endpoint, defined as in H4
    (pair_raw minus its paired null), is larger in corridor than in shear and than in barrier. Attributed to
    objective AND kernel (bce1: K = 1; mse15: K = 15)."""
    adv = {}
    for d in DESIGNS:
        g = lambda rs: {r["seed"]: r.get("pair_raw", np.nan) - r.get("pair_raw_null", np.nan) for r in rs.get(d, [])}
        b, m = g(rows_b), g(rows_m)
        adv[d] = {s: b[s] - m[s] for s in b if s in m}
    out = {}
    for other in ("shear", "barrier"):
        seeds = sorted(set(adv["corridor"]) & set(adv[other]))
        t = paired_test([adv["corridor"][s] for s in seeds], [adv[other][s] for s in seeds])
        out[f"corridor - {other}"] = dict(t, supported=bool(np.isfinite(t["ci_lo"]) and t["ci_lo"] > 0))
    return out


# ----------------------------------------------------------------------------- reporting
def fmt(s):
    if not isinstance(s, dict) or not s.get("n"):
        return "n/a"
    return f"{s['mean']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"


def summarise(rows):
    out = {}
    for d, rs in rows.items():
        if rs:
            keys = sorted({k for r in rs for k in r if k != "seed"})
            out[d] = {k: summarize([r.get(k, np.nan) for r in rs]) for k in keys}
            if d != "none":
                out[d]["frac_at_epoch_cap"] = float(np.mean([r["epochs"] >= r["max_epochs"] for r in rs]))
    return out


def family_md(F, arm_label="likelihood arm"):
    L = [f"## Family of tests ({arm_label}; Holm over all {F['family_size']} H rows)", "",
         "| hypothesis | n | mean difference [95% t] | p (Wilcoxon) | p (Holm) | supported |", "|---|---|---|---|---|---|"]
    for k, t in F["hypotheses"].items():
        L.append(f"| {k} | {t['n']} | {t['mean_diff']:+.3f} [{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}] | {t['p']:.2g} | {t['p_holm']:.2g} | {'yes' if t['supported'] else 'no'} |")
    L += ["", "Conclusions supported:", ""]
    for k, v in F.get("licences", {}).items():
        L.append(f"* {k}: **{'yes' if v else 'no'}**")
    L += ["", f"## Comparisons (two-sided; null-subtracted gains; Holm within this family of {F['comparison_family_size']})", "",
          "| comparison | n | mean difference [95% t] | p | p (Holm) |", "|---|---|---|---|---|"]
    for k, t in F["comparisons"].items():
        L.append(f"| {k} | {t['n']} | {t['mean_diff']:+.3f} [{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}] | {t['p']:.2g} | {t['p_holm']:.2g} |")
    return "\n".join(L) + "\n"


def arm_md(S, arm):
    L = [f"## Descriptive tables, arm {arm}", ""]
    if "none" in S:
        s = S["none"]
        L += ["### none (graph from the attribute geometry)", "",
              "| quantity | twist | bootstrap twist | Phase 1 | moment | moment (bootstrap graph) |", "|---|---|---|---|---|---|",
              f"| E2 direction error (deg, G1 metric) | {fmt(s.get('e2_raw_dir'))} | {fmt(s.get('e2_boot_dir'))} | | {fmt(s.get('e2_moment_dir'))} | {fmt(s.get('e2_moment_boot_dir'))} |",
              f"| E2 area Spearman | {fmt(s.get('e2_raw_area_rho'))} | {fmt(s.get('e2_boot_area_rho'))} | | {fmt(s.get('e2_moment_area_rho'))} | {fmt(s.get('e2_moment_boot_area_rho'))} |",
              f"| Spearman with true attribute geodesics, local | {fmt(s.get('fid_attr_d2_loc'))} | {fmt(s.get('fid_attr_d2boot_loc'))} | {fmt(s.get('fid_attr_d1_loc'))} | | |",
              f"| same, all pairs | {fmt(s.get('fid_attr_d2_all'))} | {fmt(s.get('fid_attr_d2boot_all'))} | {fmt(s.get('fid_attr_d1_all'))} | | |",
              f"| Phase 2 loss reduction | {fmt(s.get('loss_drop'))} | {fmt(s.get('loss_drop_boot'))} | | | |", "",
              f"Fraction of nodes with cond(dz/du) > 100: {fmt(s.get('frac_cond_J_gt_100'))}.", ""]
    for d in DESIGNS:
        if d not in S:
            continue
        s = S[d]
        L += [f"### {d} (region {s['n_region']['mean']:.0f} nodes; {100 * s['frac_at_epoch_cap']:.0f}% of runs at the epoch cap)", "",
              "| quantity | twist | none twist (paired null) | moment | moment on none | oracle excess (reference) |", "|---|---|---|---|---|---|",
              f"| E2 direction in region (deg) | {fmt(s.get('e2r_raw_dir'))} | {fmt(s.get('e2r_none_dir'))} | {fmt(s.get('e2r_moment_dir'))} | {fmt(s.get('e2r_moment_none_dir'))} | |",
              f"| E2 area Spearman in region | {fmt(s.get('e2r_raw_area_rho'))} | {fmt(s.get('e2r_none_area_rho'))} | {fmt(s.get('e2r_moment_area_rho'))} | {fmt(s.get('e2r_moment_none_area_rho'))} | |",
              f"| planted direction (deg; true attribute metric) | {fmt(s.get('dir_raw'))} | {fmt(s.get('dir_raw_null'))} | {fmt(s.get('dir_moment'))} | {fmt(s.get('dir_moment_null'))} | {fmt(s.get('dir_oracle'))} |",
              f"| its rotation chance | {fmt(s.get('dir_raw_chance'))} | | {fmt(s.get('dir_moment_chance'))} | | {fmt(s.get('dir_oracle_chance'))} |",
              f"| planted localisation AUROC (anisotropy) | {fmt(s.get('loc_raw'))} | {fmt(s.get('loc_raw_null'))} | {fmt(s.get('loc_moment'))} | {fmt(s.get('loc_moment_null'))} | {fmt(s.get('loc_oracle'))} |",
              f"| planted localisation AUROC (\\|area change\\|) | {fmt(s.get('loc_area_raw'))} | {fmt(s.get('loc_area_raw_null'))} | {fmt(s.get('loc_area_moment'))} | {fmt(s.get('loc_area_moment_null'))} | {fmt(s.get('loc_area_oracle'))} |",
              f"| planted pair distortion (stratified Spearman) | {fmt(s.get('pair_raw'))} | {fmt(s.get('pair_raw_null'))} | | | {fmt(s.get('pair_oracle'))} |",
              f"| Spearman with graph geodesics, local (Phase 1: {fmt(s.get('fid_graph_d1_loc'))}) | {fmt(s.get('fid_graph_d2_loc'))} | {fmt(s.get('fid_graph_d2none_loc'))} | | | |", "",
              f"Other estimators, planted pair distortion (with paired null): sp_d1 {fmt(s.get('pair_sp_d1'))} ({fmt(s.get('pair_sp_d1_null'))}); "
              f"hop {fmt(s.get('pair_hop'))} ({fmt(s.get('pair_hop_null'))}). Localisation AUROC (with paired null): direct measurement "
              f"{fmt(s.get('loc_direct'))} ({fmt(s.get('loc_direct_null'))}); degree {fmt(s.get('loc_degree'))} ({fmt(s.get('loc_degree_null'))}); "
              f"local cosine homophily {fmt(s.get('loc_cos_homophily'))} ({fmt(s.get('loc_cos_homophily_null'))}). "
              f"|AUROC − 0.5| of the twist: {fmt(s.get('loc_raw_abs'))} vs paired null {fmt(s.get('loc_raw_null_abs'))}.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="twist")
    ap.add_argument("--arms", default="bce1,mse15")
    ap.add_argument("--blocks", default="twist_blocks", help="run_synthetic.py tag of the likelihood block designs")
    args = ap.parse_args()
    allsum, md = {}, [f"# Metric-twist benchmark: {args.tag}", ""]
    rows_by_arm = {}
    for arm in args.arms.split(","):
        rows = collect(args.tag, arm)
        if not any(rows.values()):
            continue
        rows_by_arm[arm] = rows
        allsum[arm] = dict(summary=summarise(rows), rows=rows)
    if "bce1" in rows_by_arm:
        blocks = block_endpoints(args.blocks)
        allsum["blocks"] = blocks
        allsum["family"] = family(rows_by_arm["bce1"], blocks)
        md.append(family_md(allsum["family"]))
    if "bce1" in rows_by_arm and "mse15" in rows_by_arm:
        allsum["T4"] = t4(rows_by_arm["bce1"], rows_by_arm["mse15"])
        md.append("## Prediction T4 (likelihood + K = 1 vs squared error + K = 15; planted pair endpoint minus its paired null)\n")
        for k, t in allsum["T4"].items():
            md.append(f"* {k}: {t['mean_diff']:+.3f} [{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}], p = {t['p']:.2g}, supported = {t['supported']}")
        md.append("")
    if "mse15" in rows_by_arm:
        F2 = family(rows_by_arm["mse15"], [])
        allsum["family_mse15_descriptive"] = F2
        md.append("## Same rows, squared-error arm (descriptive; H8/H9 not run in this arm)\n")
        md.append(family_md(F2, "squared-error arm, descriptive").replace("## Family", "### Descriptive"))
    for arm in rows_by_arm:
        md.append(arm_md(allsum[arm]["summary"], arm))
    base = os.path.join(WORK, "results", "synthetic", args.tag)
    with open(os.path.join(base, "summary.json"), "w") as f:
        json.dump(jsonable(allsum), f, indent=1)
    with open(os.path.join(base, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
