"""
Figures of the planted-disagreement experiment.

fig_twist_tissot_<seed>   strain "indicatrices" on the Phase 1 latent plane, one row per design:
                          truth | cross-entropy twist (K=1) | squared-error twist | link moment
                          tensor (Phase 1 only). Ellipse = principal stretches of the strain, normalised
                          to unit area (shape = anisotropy, long axis = direction the graph lengthens);
                          colour = log-area change (estimates centred at their median: the global scale
                          is free). Estimated anisotropy is exaggerated by the stated factor so that its
                          95th percentile matches the truth's (as in engineering deformation plots).
fig_twist_sheet_<seed>    the manifold as a rubber sheet: 2-D isometric embedding (classical MDS,
                          Procrustes-aligned) of the Phase 1 geometry, of the twisted geometry and of the
                          true graph geometry, with a latent grid carried along by thin-plate interpolation.
fig_twist_pairs           the pair distortion delta = log d2 - log d1 against the true distortion.
fig_twist_summary         endpoint estimates with 95% t-intervals over seeds, per design and estimator.
forest_figure             all tests of the experiment (Supplementary Fig. S1; called by figures_paper.py).

usage: python experiments/figures_twist.py --tag twist --seed 301
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import torch
import warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ROOT, WORK

DESIGNS = ("shear", "barrier", "corridor")
ORANGE, BLUE, GREEN, INK, MUTED = "#d55e00", "#0072b2", "#009e73", "#222222", "#8a8a8a"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})


def _sq(v, n):
    M = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    M[iu] = v
    return M + M.T


def farthest_points(Z, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(Z)))]
    d = np.linalg.norm(Z - Z[idx[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(Z - Z[j], axis=1))
    return np.array(idx)


def load(tag, arm, seed, design):
    d = os.path.join(WORK, "results", "synthetic", tag, arm, f"seed{seed:03d}")
    a = dict(np.load(os.path.join(d, f"{design}_arrays.npz")))
    p1 = torch.load(os.path.join(d, "phase1.pt"), weights_only=False)
    return a, p1


CAP = 1.4     # display cap on the (exaggerated) log semi-axis ratio; capped glyphs get a dark edge


def glyphs(ax, Z, major_lat, aniso, area, sel, exaggerate=1.0, norm=None, size=None, cmap="RdBu_r"):
    if size is None:
        size = 0.55 * np.median(np.sort(np.linalg.norm(Z[sel][:, None] - Z[sel][None], axis=-1), 1)[:, 1])
    ok = sel[np.isfinite(aniso[sel]) & np.all(np.isfinite(major_lat[sel]), 1)]
    cm = plt.get_cmap(cmap)
    for i in ok:
        ang = np.degrees(np.arctan2(major_lat[i, 1], major_lat[i, 0]))
        e = exaggerate * aniso[i]
        capped = e > CAP
        e = min(e, CAP)
        w, h = 2 * size * np.exp(e / 2), 2 * size * np.exp(-e / 2)
        ax.add_patch(Ellipse(Z[i], w, h, angle=ang, facecolor=cm(norm(area[i])), edgecolor=INK if capped else "#555555",
                             lw=0.6 if capped else 0.25, alpha=0.95))
    pad = 1.6 * size
    ax.set_xlim(Z[:, 0].min() - pad, Z[:, 0].max() + pad)
    ax.set_ylim(Z[:, 1].min() - pad, Z[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def region_contour(ax, Z, phi):
    try:
        ax.tricontour(Z[:, 0], Z[:, 1], phi, levels=[0.5], colors=[INK], linewidths=0.8, linestyles="--")
    except Exception:
        pass


def _e2(tag, arm, seed, design, a, p1):
    from aligning.synthetic import ManifoldSpec
    from aligning.twist import TwistSpec
    from experiments.aggregate_twist import learned_relative_truth
    meta = json.load(open(os.path.join(WORK, "results", "synthetic", tag, arm, f"seed{seed:03d}", f"{design}.json")))
    man = ManifoldSpec.sample(np.random.default_rng(10_000 + seed), 20)
    return learned_relative_truth(man, TwistSpec(**meta["twist"]), p1["Z_true"], p1["J_dzdu"], a["G1"])


def load_meta(tag, arm, seed, design):
    return json.load(open(os.path.join(WORK, "results", "synthetic", tag, arm, f"seed{seed:03d}", f"{design}.json")))


def corrected_moment(tag, arm, seed, design, a, p1):
    from experiments.aggregate_twist import moment_of
    return moment_of(a, load_meta(tag, arm, seed, design), p1["Z1"], a["G1"])


def tissot_figure(tag, seed, arms=("bce1", "mse15"), designs=("none",) + DESIGNS, out=None, exaggerate=False):
    """Columns: planted strain (E1) | strain relative to the learned geometry (E2, what Phase 2 can see) |
    likelihood twist | squared-error twist | link moment tensor."""
    from experiments.aggregate_twist import strained
    cols = [("planted\n(graph vs true attributes)", "E1"), ("target\n(graph vs learned attributes)", "E2"),
            ("twist\nlikelihood, K = 1", "bce1"), ("twist\nsquared error", "mse15"),
            ("link moment tensor\nPhase 1 only", "moment")]
    cols = [c for c in cols if c[1] in ("E1", "E2", "moment") or c[1] in arms]
    Zr = load(tag, arms[0], seed, designs[0])[1]["Z1"]
    asp = (np.ptp(Zr[:, 1]) + 0.3) / (np.ptp(Zr[:, 0]) + 0.3)
    w_col = 2.15
    fig, axes = plt.subplots(len(designs), len(cols), figsize=(w_col * len(cols) + 0.9, w_col * asp * len(designs) + 0.9),
                             gridspec_kw=dict(wspace=0.04, hspace=0.08))
    axes = np.atleast_2d(axes)
    for r, design in enumerate(designs):
        a0, p1 = load(tag, arms[0], seed, design)
        Z, J = p1["Z1"], p1["J_dzdu"]
        sel = farthest_points(Z, 150, seed)
        e2 = _e2(tag, arms[0], seed, design, a0, p1)
        ref = np.quantile(e2["aniso"], 0.95)
        for c, (title, src) in enumerate(cols):
            ax = axes[r, c]
            if src == "E1":
                aniso, area, maj = a0["t_aniso"], a0["t_area"], np.einsum("nij,nj->ni", J, a0["t_major_u"])
            elif src == "E2":
                aniso, area, maj = e2["aniso"], e2["area"], e2["major_z"]
            elif src == "moment":
                m = corrected_moment(tag, arms[0], seed, design, a0, p1)
                aniso, area, maj = m["aniso"], m["area"], m["major_z"]
            else:
                a = a0 if src == arms[0] else load(tag, src, seed, design)[0]
                st = strained(a["G2"], a["G1"])
                aniso, area, maj = st["aniso"], st["area"], st["major_z"]
            area = area - np.nanmedian(area) if src != "E1" else area
            amax = 1.5
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-amax, vmax=amax)
            q = np.nanquantile(aniso, 0.95) if src != "E1" else 1.0
            ex = float(np.clip(ref / q, 0.25, 8)) if (exaggerate and src not in ("E1", "E2") and q > 0) else 1.0
            glyphs(ax, Z, maj, aniso, area, sel, exaggerate=ex, norm=norm)
            if design != "none":
                region_contour(ax, Z, a0["t_phi"])
            elif src == "E1":
                ax.text(0.5, 0.5, "no planted strain\n(graph follows the\nattribute geometry)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color=MUTED)
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 0:
                ax.text(-0.04, 0.5, design, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=9)
            if abs(ex - 1) > 0.05:
                ax.text(0.0, 0.0, f"anisotropy ×{ex:.1f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=6, color=MUTED)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-1.5, vmax=1.5))
    cb = fig.colorbar(sm, ax=axes, shrink=0.35, pad=0.01, aspect=25)
    cb.set_label("log-area change (centred)\nblue: graph contracts · red: graph expands", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.text(0.02, 0.005, f"Seed {seed}. Each ellipse is a node's strain on the Phase 1 latent plane, normalised to unit area: long axis = "
             "direction in which the graph lengthens distances relative to the rest; dashed line: planted region (φ = 0.5). "
             + ("Estimates' anisotropy is exaggerated as stated (supplementary version); " if exaggerate else "No exaggeration. ") + "Thick-edged glyphs are capped for display.", fontsize=6.5, color=MUTED, wrap=True)
    out = out or os.path.join(WORK, "results", "synthetic", tag, "figures", f"fig_twist_tissot_seed{seed}" + ("_exaggerated" if exaggerate else ""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def cmds(D, k=2):
    n = D.shape[0]
    Jc = np.eye(n) - 1.0 / n
    Bm = -0.5 * Jc @ (D ** 2) @ Jc
    w, v = np.linalg.eigh(Bm)
    o = np.argsort(w)[::-1][:k]
    return v[:, o] * np.sqrt(np.clip(w[o], 0, None))


def procrustes_to(Y, X):
    """Similarity transform of Y onto X (rotation/reflection, scale, translation)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, s, Vt = np.linalg.svd(Yc.T @ Xc)
    R = U @ Vt
    scale = s.sum() / (Yc ** 2).sum()
    return scale * Yc @ R + mx


def local_embedding(D, k=12):
    """Isomap-style 2-D embedding: k-nearest-neighbour graph of D, shortest paths, classical MDS. Uses only
    local distances, so a few badly inflated long distances cannot dominate the picture."""
    from scipy.sparse.csgraph import shortest_path
    n = D.shape[0]
    nn = np.argsort(D, 1)[:, 1:k + 1]
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    W[rows, nn.ravel()] = D[rows, nn.ravel()]
    W = np.maximum(W, W.T)
    G = shortest_path(W, directed=False)
    G[~np.isfinite(G)] = np.nanmax(G[np.isfinite(G)])
    return cmds(G)


def sheet_mesh(Z):
    from scipy.spatial import Delaunay
    tri = Delaunay(Z).simplices
    e = np.concatenate([np.linalg.norm(Z[tri[:, a]] - Z[tri[:, b]], axis=1)[:, None] for a, b in ((0, 1), (1, 2), (0, 2))], 1)
    keep = e.max(1) < 2.5 * np.median(e)
    return tri[keep]


def tri_area(Y, tri):
    a, b, c = Y[tri[:, 0]], Y[tri[:, 1]], Y[tri[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def spring_edges(Z, tri, k=8):
    E = set()
    for t in tri:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = sorted((int(t[a]), int(t[b])))
            E.add((i, j))
    from scipy.spatial import cKDTree
    _, nn = cKDTree(Z).query(Z, k=k + 1)
    for i in range(Z.shape[0]):
        for j in nn[i, 1:]:
            E.add(tuple(sorted((i, int(j)))))
    return np.array(sorted(E))


def relax_sheet(X0, E, rest, iters=300):
    """Stress majorisation (SMACOF, Guttman transform) of a spring mesh: springs on the edges E with rest
    lengths `rest` and weights rest^-2, started from X0. Only local springs: the sheet bends and stretches
    locally, it does not fold to honour long distances."""
    n = X0.shape[0]
    i, j = E[:, 0], E[:, 1]
    w = 1.0 / rest ** 2
    V = np.zeros((n, n))
    V[i, j] = V[j, i] = -w
    V[np.arange(n), np.arange(n)] = -V.sum(1)
    Vp = np.linalg.pinv(V)
    X = X0.copy()
    for _ in range(iters):
        d = np.linalg.norm(X[i] - X[j], axis=1)
        b = -w * rest / np.maximum(d, 1e-12)
        Bm = np.zeros((n, n))
        Bm[i, j] = Bm[j, i] = b
        Bm[np.arange(n), np.arange(n)] = -Bm.sum(1)
        X = Vp @ (Bm @ X)
    return X


def _relaxed_sheets(Z, E, Dref, Dothers, tear):
    """Relax a reference sheet to Dref, then each other geometry from it. Returns sheets, torn masks."""
    r1 = Dref[E[:, 0], E[:, 1]]
    X1 = relax_sheet((Z - Z.mean(0)) * np.median(r1) / np.median(np.linalg.norm(Z[E[:, 0]] - Z[E[:, 1]], axis=1)), E, r1)
    sheets, torn = [X1], [np.zeros(len(E), bool)]
    for Dm in Dothers:
        r = Dm[E[:, 0], E[:, 1]]
        r = r * np.median(r1) / np.median(r)
        ratio = r / r1
        cut = ratio > tear * np.median(ratio)
        sheets.append(procrustes_to(relax_sheet(X1, E[~cut], r[~cut]), X1))
        torn.append(cut)
    return sheets, torn


def _draw_sheet(ax, Y, tri, A0, torn, eset, phi, design, title, ref=False):
    from matplotlib.collections import PolyCollection
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.2, vmax=1.2)
    la = np.log(np.maximum(tri_area(Y, tri), 1e-12) / np.maximum(A0, 1e-12))
    la -= np.median(la)
    holes = np.array([any(torn[eset[tuple(sorted((int(tr[x]), int(tr[y]))))]] for x, y in ((0, 1), (1, 2), (0, 2))) for tr in tri])
    fc = plt.get_cmap("RdBu_r")(norm(la)) if not ref else np.tile(np.array([[0.86, 0.86, 0.84, 1.0]]), (len(tri), 1))
    fc[holes] = (1, 1, 1, 0)
    ax.add_collection(PolyCollection(Y[tri], facecolors=fc, edgecolors="#ffffff", linewidths=0.2))
    if design != "none":
        ax.scatter(Y[phi >= 0.5, 0], Y[phi >= 0.5, 1], s=2.2, c=INK, lw=0, zorder=3)
    if holes.any():
        ax.text(0.02, 0.0, f"{int(holes.sum())} torn triangles", transform=ax.transAxes, fontsize=6.5, color=MUTED)
    ax.set_xlim(Y[:, 0].min(), Y[:, 0].max()); ax.set_ylim(Y[:, 1].min(), Y[:, 1].max())
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(title, fontsize=8)


def sheet_figure(tag, seed, design, arms=("bce1", "mse15"), out=None, tear=5.0):
    """The attribute manifold as a rubber sheet (spring mesh: Delaunay triangulation of the nodes on the
    Phase 1 latent plane + 8 nearest neighbours; stress majorisation). Top row, the truth: the true attribute
    geometry relaxed to the true graph geometry (the planted twist alone). Bottom row, the method: the Phase 1
    geometry relaxed to each twisted geometry. Triangles: log-area change relative to the row's first sheet
    (median 0). Springs stretched > `tear` x their typical ratio are cut (holes)."""
    a, p1 = load(tag, arms[0], seed, design)
    Z = p1["Z1"]
    n = Z.shape[0]
    tri = sheet_mesh(Z)
    E = spring_edges(Z, tri)
    eset = {tuple(e): k for k, e in enumerate(E)}
    Zt = p1["Z_true"]                      # the truth row uses the true manifold's own mesh
    tri_t = sheet_mesh(Zt)
    E_t = spring_edges(Zt, tri_t)
    eset_t = {tuple(e): k for k, e in enumerate(E_t)}
    phi = a["t_phi"]
    truth, ttorn = _relaxed_sheets(Zt, E_t, _sq(a["d_attr"], n), [_sq(a["d_graph"], n)], tear)
    d2s = [_sq((a if arm == arms[0] else load(tag, arm, seed, design)[0])["d2"], n) for arm in arms]
    meth, mtorn = _relaxed_sheets(Z, E, _sq(a["d1"], n), d2s, tear)
    sc_t = np.sqrt(((meth[0] - meth[0].mean(0)) ** 2).sum() / ((truth[0] - truth[0].mean(0)) ** 2).sum())
    Rt = procrustes_to(truth[0], meth[0])                       # same nodes: orient the truth row like the method row
    U, _, Vt = np.linalg.svd((truth[0] - truth[0].mean(0)).T @ (Rt - Rt.mean(0)))
    truth = [(T - truth[0].mean(0)) @ (U @ Vt) * sc_t + meth[0].mean(0) for T in truth]
    fig, axes = plt.subplots(2, 1 + len(arms), figsize=(2.75 * (1 + len(arms)), 5.4), gridspec_kw=dict(wspace=0.06, hspace=0.18))
    _draw_sheet(axes[0, 0], truth[0], tri_t, tri_area(truth[0], tri_t), ttorn[0], eset_t, phi, design, "true attribute geometry", ref=True)
    _draw_sheet(axes[0, 1], truth[1], tri_t, tri_area(truth[0], tri_t), ttorn[1], eset_t, phi, design, "true graph geometry\n(planted twist)")
    for c in range(2, 1 + len(arms)):
        axes[0, c].axis("off")
    axes[0, 2].text(0.02, 0.62, "Top: the truth. The graph was generated\nfrom a twisted copy of the attribute\ngeometry.\n"
                    "Bottom: the method. Phase 2 twists the\nlearned attribute manifold until its\ndiffusion geometry fits the graph.",
                    transform=axes[0, 2].transAxes, fontsize=7.2, color=INK, va="center")
    _draw_sheet(axes[1, 0], meth[0], tri, tri_area(meth[0], tri), mtorn[0], eset, phi, design, "Phase 1 geometry\n(learned from attributes)", ref=True)
    for k, arm in enumerate(arms):
        _draw_sheet(axes[1, 1 + k], meth[1 + k], tri, tri_area(meth[0], tri), mtorn[1 + k], eset, phi, design,
                    {"bce1": "twisted onto the graph\nlikelihood, K = 1", "mse15": "twisted onto the graph\nsquared error"}[arm])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.2, vmax=1.2))
    cax = axes[0, 2].inset_axes([0.05, 0.3, 0.75, 0.06])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("log-area change vs the row's first sheet\nblue: contracted · red: stretched", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5)
    fig.text(0.02, 0.02, f"{design}, seed {seed}. Spring mesh on the nodes (Delaunay + 8 nearest neighbours) relaxed to the geodesic lengths of each "
             "geometry by stress majorisation, starting from the row's reference sheet (same median spring length). Black dots: planted region. "
             f"Holes: springs stretched > {tear:g}x the typical ratio (the sheet tears).", fontsize=6.5, color=MUTED, wrap=True)
    out = out or os.path.join(WORK, "results", "synthetic", tag, "figures", f"fig_twist_sheet_{design}_seed{seed}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return out, dict(truth=truth, method=meth, torn_truth=ttorn, torn_method=mtorn), tri


PLAIN_NAMES = {"H1": "direction", "H2": "area", "H3": "repair", "H4": "approach to graph geometry", "H5a": "vs rotation chance",
               "H5b": "vs untwisted design", "H6": "localisation", "H7": "pair distortion", "H8": "detection", "H9": "repair"}
PLAIN_DESIGNS = {"none": "", "shear": "shear", "barrier": "barrier", "corridor": "corridor", "R70": "scattered",
                 "D70": "compact block", "BAND": "band", "D70N": "compact block", "BANDN": "band"}
PLAIN_GROUPS = ["No planted disagreement: twist axis vs Phase 1's own error (deg; minus bootstrap twist)",
                "No planted disagreement: area of Phase 1's error, and repair (\u0394 Spearman; minus bootstrap twist)",
                "Planted designs: twist direction vs target strain (deg; minus untwisted design)",
                "Planted designs: area of the target strain, and approach to the graph geometry (\u0394 Spearman)",
                "Planted direction (deg)",
                "Planted localisation (\u0394 AUROC) and pair distortion (\u0394 Spearman); minus untwisted design",
                "Edge contraction detects misaligned nodes (\u0394 AUROC)",
                "Repair with other aligned graphs (\u0394 Spearman)"]


def forest_figure(tag, out=None, plain=False):
    """All tests of the planted-disagreement experiment (likelihood arm): mean paired difference with its 95% t-interval, one
    row per hypothesis, grouped by endpoint; filled marker = supported (interval on the predicted side and Holm
    p < 0.05). Panels have their own x-scales (degrees vs correlation vs AUROC differences)."""
    S = json.load(open(os.path.join(WORK, "results", "synthetic", tag, "summary.json")))
    H = S["family"]["hypotheses"]
    groups = [("Untwisted design: Phase 1's own error, direction (deg; twist minus bootstrap twist)", ("H1",), "none"),
              ("Untwisted design: error area (H2) and repair (H3) (Δ Spearman; twist minus bootstrap twist)", ("H2", "H3"), "none"),
              ("Beyond repair: strain direction in the planted region (deg; minus the untwisted design's twist)", ("H1",), "twisted"),
              ("Beyond repair: strain area (H2) and approach to the graph geometry (H4) (Δ Spearman)", ("H2", "H4"), "twisted"),
              ("Planted direction (deg). H5a: minus rotation chance; H5b: minus the untwisted design's twist", ("H5a", "H5b"), "twisted"),
              ("Planted localisation (H6, Δ AUROC) and pair distortion δ (H7, Δ Spearman); minus the untwisted twist", ("H6", "H7"), "twisted"),
              ("Edge contraction detects misaligned nodes (Δ AUROC). R70 scattered; D70, BAND blocks", ("H8",), "any"),
              ("Repair in other realisations and widths (Δ Spearman, d2 vs d1)", ("H9",), "any")]
    def pick(ids, scope):
        out = []
        for k, v in H.items():
            hid, des = k.split()[0], k[k.rfind("[") + 1:-1]
            if hid in ids and (scope == "any" or (scope == "none") == (des == "none")):
                out.append((k, v))
        return out
    fig, axes = plt.subplots(len(groups), 1, figsize=(6.4, 1.05 + 0.33 * len(H)),
                             gridspec_kw=dict(height_ratios=[max(1, len(pick(g[1], g[2]))) for g in groups], hspace=1.0))
    for g_, (ax, (title, ids, scope)) in enumerate(zip(axes, groups)):
        rows = pick(ids, scope)
        for y, (k, v) in enumerate(rows[::-1]):
            col = BLUE if v["supported"] else MUTED
            ax.plot([v["ci_lo"], v["ci_hi"]], [y, y], color=col, lw=1.6, solid_capstyle="round")
            ax.plot(v["mean_diff"], y, "o", ms=5.5, mfc=col if v["supported"] else "white", mec=col, mew=1.4)
        ax.axvline(0, color=INK, lw=0.7)
        ax.set_yticks(range(len(rows)))
        if plain:
            labs = [", ".join(x for x in (PLAIN_NAMES[k.split()[0]], PLAIN_DESIGNS[k[k.rfind('[') + 1:-1]]) if x)
                    + (" (>0.5)" if "> 0.5" in k else "") for k, _ in rows[::-1]]
            title = PLAIN_GROUPS[g_]
        else:
            labs = [f"{k.split()[0]} {k[k.rfind('[') + 1:-1]}" + (" (>0.5)" if "> 0.5" in k else "") for k, _ in rows[::-1]]
        ax.set_yticklabels(labs, fontsize=7)
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(-0.7, len(rows) - 0.3)
    if not plain:
        fig.text(0.0, -0.01, "Mean paired difference over seeds with its 95% t-interval. Filled: supported (interval on the predicted side and "
                 f"Holm-adjusted Wilcoxon p < 0.05; {len(H)} rows).", fontsize=6.5, color=MUTED, wrap=True)
    out = out or os.path.join(WORK, "results", "synthetic", tag, "figures", "fig_twist_family")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="twist")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arms", default="bce1,mse15")
    args = ap.parse_args()
    arms = tuple(args.arms.split(","))
    print(tissot_figure(args.tag, args.seed, arms))
    print(tissot_figure(args.tag, args.seed, arms, exaggerate=True))
    for d in DESIGNS:
        print(sheet_figure(args.tag, args.seed, d, arms)[0])
    if os.path.exists(os.path.join(WORK, "results", "synthetic", args.tag, "summary.json")):
        print(forest_figure(args.tag))


if __name__ == "__main__":
    main()
