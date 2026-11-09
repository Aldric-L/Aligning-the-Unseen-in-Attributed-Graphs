"""
Figures of the paper and of the Supplementary Information. Every number is read from results/; nothing is typed by
hand. Vector PDF, Type 42 fonts, Okabe-Ito colours (colour-blind safe), 180 mm width.

  fig2_tax.pdf              the cost of a shared representation: structured warp vs scattered twin
  fig3a_strain.pdf          planted disagreements: strain ellipses (seed 301)
  fig3b_recovery.pdf        planted disagreements: direction, localisation, pair distortion and repair
  fig4_amazon.pdf           product network: complement vs control categories
  figS1_results.pdf         planted disagreements: all tests (Supplementary Fig. S1)
  figS_followups.pdf        further analyses of the tax
  figS_amazon_transfer.pdf  product network: 10-NN category agreement

usage: python experiments/figures_paper.py [--out DIR]      (default: outputs/figures/)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ROOT, WORK

OUT = os.path.join(ROOT, "outputs", "figures")
MM = 1 / 25.4
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7", sky="#56B4E9",
          yellow="#F0E442", grey="#7F7F7F", black="#000000")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
                     "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300})


def panel(ax, letter, x=-0.22, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def ci95(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), h


# ----------------------------------------------------------------------------- Fig. 2: the tax
def draw_network(ax, Z, A, colour, title, sub, max_edges=None, rng=None, thr=None):
    from matplotlib.collections import LineCollection
    i, j = np.where(np.triu(A > 0, 1))
    if max_edges is not None and len(i) > max_edges:
        k = (rng or np.random.default_rng(0)).choice(len(i), max_edges, replace=False)
        i, j = i[k], j[k]
    segs = np.stack([Z[i], Z[j]], 1)
    long_ = np.linalg.norm(Z[i] - Z[j], axis=1) > thr
    ax.add_collection(LineCollection(segs[~long_], colors="#4D4D4D", linewidths=0.25, alpha=0.35, zorder=1))
    ax.add_collection(LineCollection(segs[long_], colors=OI["red"], linewidths=0.35, alpha=0.45, zorder=1))
    sc = ax.scatter(Z[:, 0], Z[:, 1], s=5, c=colour, cmap="viridis", lw=0, zorder=2)
    lo, hi = Z.min(0), Z.max(0)
    pad = 0.04 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=7, pad=2)
    ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color="#333333")
    return sc


def fig2():
    import torch
    R = json.load(open(os.path.join(WORK, "results", "synthetic", "globaltax", "analysis_d2.json")))
    rows = R["rows"]
    seeds = sorted({r["seed"] for r in rows})
    K = [0.0, 1.0, 1.5, 2.0]
    get = lambda s, k, g, key: next(r[key] for r in rows if r["seed"] == s and r["kappa"] == k and r["graph"] == g)
    truth_f = os.path.join(WORK, "results", "synthetic", "globaltax_followups", "truth.json")
    truth = {(t["seed"], t["kappa"]): t for t in json.load(open(truth_f))} if os.path.exists(truth_f) else {}
    fig = plt.figure(figsize=(180 * MM, 128 * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.38)
    top = outer[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.32], wspace=0.08)
    bot = outer[1].subgridspec(1, 5, width_ratios=[1, 1, 1, 0.42, 1.05], wspace=0.8)
    # ---- top row: the three kinds of graph (seed 601, kappa = 2)
    base = os.path.join(WORK, "results", "synthetic", "globaltax", "seed601")
    i2 = torch.load(os.path.join(base, "k2.00", "inputs.pt"), weights_only=False)
    i0 = torch.load(os.path.join(base, "k0.00", "inputs.pt"), weights_only=False)
    info = json.load(open(os.path.join(base, "k2.00", "instance.json")))
    Z = np.asarray(i2["Z_true"], float)
    X = np.asarray(i2["X"], float)
    Xc = X - X.mean(0)
    colour = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][0]
    rng = np.random.default_rng(1)
    D0 = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
    thr = float(np.quantile(D0[np.triu_indices(len(Z), 1)], 0.25))
    panels = [(np.asarray(i0["A_warp"]), "aligned", f"links follow attributes\n(attribute distance of links {info['homophily_aligned']:.2f})"),
              (np.asarray(i2["A_warp"]), "structured warp", f"links follow a deformed geometry\n(attribute distance of links {info['homophily_target']:.2f})"),
              (np.asarray(i2["A_twin"]), "scattered twin", f"links rewired at random, same homophily\n(attribute distance of links {info['homophily_twin']:.2f})")]
    for k_, (A, t_, sub) in enumerate(panels):
        ax = fig.add_subplot(top[k_])
        sc = draw_network(ax, Z, A, colour, t_, sub, max_edges=2500, rng=rng, thr=thr)
        if k_ == 0:
            panel(ax, "a", x=-0.02, y=1.0)
    from matplotlib.lines import Line2D
    axl = fig.add_subplot(top[3])
    axl.axis("off")
    cax = axl.inset_axes([0.05, 0.55, 0.12, 0.38])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_ticks([])
    cax.set_title("node\nattributes", fontsize=5.5, loc="left")
    cax.text(1.4, 1.0, "similar\ncolour =\nsimilar\nattributes", transform=cax.transAxes, fontsize=5, va="top")
    axl.legend(handles=[Line2D([], [], color="#4D4D4D", lw=1, alpha=0.6, label="short link"),
                        Line2D([], [], color=OI["red"], lw=1, label="long link")],
               loc="lower left", bbox_to_anchor=(0.0, 0.05), frameon=False, fontsize=5.5, handlelength=1.2)
    axes = [fig.add_subplot(bot[i]) for i in (0, 1, 2, 4)]
    # b: tax by kappa
    ax = axes[0]
    for g, col, name in (("warp", OI["blue"], "structured warp"), ("twin", OI["orange"], "scattered twin")):
        for s in seeds:
            ax.plot(K, [get(s, k, "warp" if k == 0 else g, "tax") for k in K], color=col, alpha=0.15, lw=0.5)
        m = [ci95([get(s, k, "warp" if k == 0 else g, "tax") for s in seeds]) for k in K]
        ax.errorbar(K, [a for a, _ in m], yerr=[h for _, h in m], color=col, marker="o", ms=3, lw=1.3, capsize=1.5, label=name)
    ax.axhline(0, color=OI["grey"], lw=0.5, ls=":")
    ax.set_xlabel(r"warp strength $\kappa$")
    ax.set_ylabel("cost of sharing one code\n(tax, task units)")
    ax.set_xticks(K)
    ax.legend(frameon=False, loc="upper left", fontsize=5.5)
    panel(ax, "b")
    # c: paired differences
    ax = axes[1]
    for key, col, mk, lab in (("tax", OI["black"], "o", "tax"), ("excess", OI["blue"], "s", r"twist excess $\sigma_\delta$")):
        diffs = np.array([[get(s, k, "warp", key) - get(s, k, "twin", key) for k in K[1:]] for s in seeds])
        m = [ci95(diffs[:, i]) for i in range(3)]
        off = -0.06 if key == "tax" else 0.06
        ax.errorbar(np.array(K[1:]) + off, [a for a, _ in m], yerr=[h for _, h in m], fmt=mk, ms=3, color=col, capsize=1.5, lw=1, label=lab)
    ax.axhline(0, color=OI["grey"], lw=0.5, ls=":")
    ax.set_xticks(K[1:])
    ax.set_xlabel(r"warp strength $\kappa$")
    ax.set_ylabel("warp $-$ twin (paired by seed)")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5)
    panel(ax, "c")
    # d: tax vs true disagreement
    ax = axes[2]
    if truth:
        for k, col in zip(K, ["#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]):
            ax.scatter([truth[(s, k)]["delta2_attr_graph"] for s in seeds], [get(s, k, "warp", "tax") for s in seeds], s=7, color=col, lw=0,
                       label=rf"warp $\kappa$={k:g}")
        taf_f = os.path.join(WORK, "results", "synthetic", "globaltax_followups", "truth_areafree.json")
        if os.path.exists(taf_f):
            taf = {(t["seed"], t["kappa"]): t["delta2_attr_areafree"] for t in json.load(open(taf_f))}
            ax.scatter([taf[(s, k)] for s in seeds for k in K[1:]],
                       [json.load(open(os.path.join(WORK, "results", "synthetic", "globaltax_followups", f"seed{s}", f"k{k:.2f}", "tax_areafree.json")))["tax"]
                        for s in seeds for k in K[1:]], s=7, facecolors="white", edgecolors=OI["green"], lw=0.6, label="area-free warp")
        allx = [truth[(s, k)]["delta2_attr_graph"] for s in seeds for k in K]
        ally = [get(s, k, "warp", "tax") for s in seeds for k in K]
        ax.text(0.03, 0.97, rf"$\rho$ = {stats.spearmanr(allx, ally)[0]:.2f}", transform=ax.transAxes, va="top", fontsize=6)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=4.8, handletextpad=0.1, labelspacing=0.2)
    ax.axhline(0, color=OI["grey"], lw=0.5, ls=":")
    ax.set_xlabel(r"true disagreement $\Delta(d_A,d_G)$")
    ax.set_ylabel("tax")
    panel(ax, "d")
    # e: competitors
    ax = axes[3]
    names = [("excess", r"twist excess $\sigma_\delta$"), ("sigma_sp", r"path distortion $\sigma_{sp}$"),
             ("sd_log_degree", "degree heterogeneity"), ("clustering", "clustering"),
             ("sigma_link", "link-length spread"), ("direct", "mean link length"), ("homophily_X", "attribute distance of links")]
    vals = []
    for key, lab in names:
        rho = [stats.spearmanr([r[key] for r in rows if r["seed"] == s], [r["tax"] for r in rows if r["seed"] == s])[0] for s in seeds]
        vals.append((lab, *ci95(rho), key == "excess"))
    vals.sort(key=lambda v: v[1])
    for yi, (lab, m, h, tw) in enumerate(vals):
        ax.errorbar(m, yi, xerr=h, fmt="o", ms=3, color=OI["blue"] if tw else OI["black"], mfc="white" if tw else OI["black"], capsize=1.5, lw=1)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([v[0] for v in vals])
    ax.axvline(0, color=OI["grey"], lw=0.5, ls=":")
    ax.set_xlabel(r"within-seed $\rho$ with tax")
    ax.set_xlim(-1, 1)
    panel(ax, "e", x=-1.05)
    fig.savefig(os.path.join(OUT, "fig2_tax.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 3: twist on planted disagreement
def strain_panel(ax, Z, major, aniso, phi, idx, norm, cmap, r=0.16, a_max=1.2):
    from matplotlib.patches import Ellipse
    from matplotlib.collections import PatchCollection
    pats, cols = [], []
    for i in idx:
        v = major[i]
        if not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0:
            v, a = np.array([1.0, 0.0]), 0.0
        else:
            a = float(aniso[i]) if np.isfinite(aniso[i]) else 0.0
        a = min(a, a_max)              # display cap: the colour bar saturates at the same value
        ang = np.degrees(np.arctan2(v[1], v[0]))
        pats.append(Ellipse(Z[i], 2 * r * np.exp(a / 2), 2 * r * np.exp(-a / 2), angle=ang))
        cols.append(a)
    pc = PatchCollection(pats, cmap=cmap, norm=norm, lw=0)
    pc.set_array(np.array(cols))
    ax.add_collection(pc)
    if phi is not None and phi.max() > 0:
        ax.tricontour(Z[:, 0], Z[:, 1], phi, levels=[0.5], colors="k", linewidths=0.6, linestyles="--")
    lo, hi = Z[idx].min(0) - r * np.exp(a_max / 2), Z[idx].max(0) + r * np.exp(a_max / 2)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return pc


def fig3():
    import torch
    from matplotlib.colors import Normalize
    base = os.path.join(WORK, "results", "synthetic", "twist", "bce1", "seed301")
    Z = np.asarray(torch.load(os.path.join(base, "phase1.pt"), weights_only=False)["Z_true"], float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Z), 220, replace=False)
    norm, cmap = Normalize(0, 1.2), plt.get_cmap("viridis")
    D = ["shear", "barrier", "corridor"]
    cols = [("planted strain", "t_major_u", "t_aniso"), ("twist (estimated)", "major_u", "aniso"),
            ("link-moment tensor\n(training-free)", "mom_major_u", "mom_aniso")]
    fig, axes = plt.subplots(3, 3, figsize=(120 * MM, 118 * MM), gridspec_kw=dict(wspace=0.04, hspace=0.06))
    for r_, d in enumerate(D):
        z = np.load(os.path.join(base, f"{d}_arrays.npz"))
        for c_, (title, km, ka) in enumerate(cols):
            ax = axes[r_, c_]
            pc = strain_panel(ax, Z, z[km], z[ka], z["t_phi"], idx, norm, cmap)
            if r_ == 0:
                ax.set_title(title, fontsize=7)
            if c_ == 0:
                ax.text(-0.08, 0.5, d, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=7)
    cb = fig.colorbar(pc, ax=axes, fraction=0.025, pad=0.02, extend="max", shrink=0.6, anchor=(0, 0.9))
    cb.set_label("anisotropy (half log ratio of the strain eigenvalues)", fontsize=6)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.legend_handler import HandlerPatch

    class _HE(HandlerPatch):
        def create_artists(self, legend, orig, x0, y0, w, h, fs, trans):
            e = Ellipse((x0 + w / 2, y0 + h / 2), w * orig.width, h * orig.height, color=orig.get_facecolor())
            e.set_transform(trans)
            return [e]
    hs = [Ellipse((0, 0), 0.45, 0.9, color=cmap(norm(0.0))), Ellipse((0, 0), 1.2, 0.35, color=cmap(norm(1.2))),
          Line2D([], [], color="k", ls="--", lw=0.8)]
    fig.legend(hs, ["isotropic: same stretch\nin every direction", "anisotropic: long axis =\nmost stretched direction",
                    "planted region"], handler_map={Ellipse: _HE()}, loc="lower left", bbox_to_anchor=(0.905, 0.1),
               frameon=False, fontsize=5.5, handlelength=1.8, labelspacing=1.0, title="one ellipse per node", title_fontsize=5.8)
    cb.ax.tick_params(labelsize=5.5)
    axes[0, 0].text(-0.18, 1.08, "a", transform=axes[0, 0].transAxes, fontsize=8, fontweight="bold")
    fig.savefig(os.path.join(OUT, "fig3a_strain.pdf"), bbox_inches="tight")
    plt.close(fig)

    S = json.load(open(os.path.join(WORK, "results", "synthetic", "twist", "summary.json")))["bce1"]["summary"]
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 50 * MM), gridspec_kw=dict(wspace=0.7))
    x = np.arange(3)

    def series(ax, key, col, mk, lab, off, mfc=None):
        m = [S[d][key]["mean"] for d in D]
        lo = [S[d][key]["mean"] - S[d][key]["ci_lo"] for d in D]
        hi = [S[d][key]["ci_hi"] - S[d][key]["mean"] for d in D]
        ax.errorbar(x + off, m, yerr=[lo, hi], fmt=mk, ms=3.2, color=col, mfc=mfc or col, capsize=1.5, lw=0.9, label=lab)
    leg = dict(frameon=False, loc="upper left", fontsize=5.2, ncol=1, handletextpad=0.3, borderaxespad=0.2)
    ax = axes[0]
    series(ax, "dir_raw", OI["blue"], "o", "twist", -0.2)
    series(ax, "dir_moment", OI["green"], "s", "link-moment tensor", 0.0)
    series(ax, "dir_raw_chance", OI["grey"], "_", "rotation chance", 0.2)
    ax.set_ylabel("planted-axis error (deg)")
    ax.legend(**leg)
    ax.set_ylim(0, 80)
    panel(ax, "b")
    ax = axes[1]
    series(ax, "loc_raw", OI["blue"], "o", "twist", -0.2)
    series(ax, "loc_moment", OI["green"], "s", "link-moment tensor", 0.0)
    series(ax, "loc_direct", OI["red"], "D", "direct link length", 0.2)
    series(ax, "loc_raw_null", OI["grey"], "_", "twist, untwisted design", -0.2)
    ax.axhline(0.5, color=OI["grey"], lw=0.5, ls=":")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("localisation AUROC")
    ax.legend(**dict(leg, loc="lower left"))
    panel(ax, "c")
    ax = axes[2]
    series(ax, "pair_raw", OI["blue"], "o", r"twist $\delta$", -0.2)
    series(ax, "pair_sp_d1", OI["green"], "s", "shortest-path distortion", 0.0)
    series(ax, "pair_hop", OI["red"], "D", "hop distortion", 0.2)
    series(ax, "pair_raw_null", OI["grey"], "_", "twist, untwisted design", -0.2)
    ax.axhline(0, color=OI["grey"], lw=0.5, ls=":")
    ax.set_ylim(-0.25, 1.15)
    ax.set_ylabel(r"Spearman with true $\delta^*$")
    ax.legend(**leg)
    panel(ax, "d")
    for ax in axes[:3]:
        ax.set_xticks(x)
        ax.set_xticklabels(D)
        ax.set_xlim(-0.5, 2.5)
    ax = axes[3]
    N = S["none"]
    pts = [("Phase 1", "fid_attr_d1_loc", OI["grey"]), ("bootstrap\ntwist", "fid_attr_d2boot_loc", OI["sky"]), ("twist", "fid_attr_d2_loc", OI["blue"])]
    for i, (lab, key, col) in enumerate(pts):
        ax.errorbar(i, N[key]["mean"], yerr=[[N[key]["mean"] - N[key]["ci_lo"]], [N[key]["ci_hi"] - N[key]["mean"]]], fmt="o", ms=4,
                    color=col, capsize=2, lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([p[0] for p in pts], fontsize=5.5, rotation=30, ha="right")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.6, 0.9)
    ax.set_ylabel("local agreement with true\nattribute geodesics (Spearman)")
    panel(ax, "e")
    fig.savefig(os.path.join(OUT, "fig3b_recovery.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 4: real systems
SHORT = {"Microphones & Accessories > Microphones": "microphones", "Microphones & Accessories > Accessories": "mic accessories",
         "Live Sound & Stage > Stage & Studio Cables": "stage cables", "Guitars > Acoustic Guitars": "acoustic guitars",
         "Guitars > Electric Guitars": "electric guitars", "Instrument Accessories > Guitar & Bass Accessories": "guitar accessories",
         "Amplifiers & Effects > Guitar Effects": "guitar effects", "Amplifiers & Effects > Guitar & Bass Amplifiers": "amplifiers",
         "Drums & Percussion > Hand Percussion": "hand percussion", "Instrument Accessories > Drum & Percussion Accessories": "drum accessories",
         "Drums & Percussion > Drum Sets & Set Components": "drum sets", "Wind & Woodwind Instruments > Folk & World": "folk wind",
         "Instrument Accessories > Wind & Woodwind Accessories": "wind accessories", "Band & Orchestra > Wind & Woodwinds": "band woodwinds",
         "Band & Orchestra > Orchestral Strings": "orchestral strings", "Instrument Accessories > Orchestral Strings Accessories": "string accessories",
         "Stringed Instruments > Folk & World": "folk strings", "Instrument Accessories > Folk & World Instrument Accessories": "folk accessories",
         "Studio Recording Equipment > Computer Recording": "audio interfaces"}


def fig4():
    sys.path.insert(0, ROOT)
    from experiments.analyze_amazon import P_REG
    A = json.load(open(os.path.join(WORK, "results", "amazon", "analysis.json")))
    data = os.path.join(WORK, "data", "Amazon", "Musical_Instruments")
    lab = np.array(json.load(open(os.path.join(data, "meta.json")))["level3"])
    Pset = {frozenset(p) for p in P_REG}
    fig = plt.figure(figsize=(180 * MM, 72 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.75)
    # a: what each layer links
    ax = fig.add_subplot(gs[0])
    seg_cols = [OI["grey"], OI["orange"], "#DDDDDD"]
    seg_names = ["same category", "complement categories\n(specified in advance)", "other categories"]
    layers = [("A_S", "viewed, then bought", "e.g. microphone – microphone"),
              ("A_C", "bought together", "e.g. microphone – stage cable")]
    shares = []
    for yi, (f, name, ex) in zip([1, 0], layers):
        Aly = np.load(os.path.join(data, f + ".npy")) > 0
        i_, j_ = np.where(np.triu(Aly, 1))
        same = (lab[i_] == lab[j_]) & (lab[i_] != "")
        comp = np.array([frozenset((a, b)) in Pset for a, b in zip(lab[i_], lab[j_])]) & ~same
        sh = [same.mean(), comp.mean(), (~same & ~comp).mean()]
        shares.append(sh)
        left = 0.0
        for k, (v, c) in enumerate(zip(sh, seg_cols)):
            ax.barh(yi, v, left=left, height=0.5, color=c, edgecolor="white", lw=1)
            if v > 0.06:
                ax.text(left + v / 2, yi, f"{100 * v:.0f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if k < 2 else OI["black"], fontweight="bold")
            left += v
        ax.text(-0.02, yi + 0.06, name, ha="right", va="center", fontsize=6.8, fontweight="bold")
        ax.text(-0.02, yi - 0.17, f"{len(i_):,} links\n{ex}", ha="right", va="center", fontsize=5.5, color="#444444")
    xs = shares[0][0] + shares[0][1] / 2
    ax.annotate(f"{100 * shares[0][1]:.0f}%", xy=(xs, 1.25), xytext=(xs, 1.42), ha="center", fontsize=6, color=OI["orange"],
                fontweight="bold", arrowprops=dict(arrowstyle="-", color=OI["orange"], lw=0.7))
    ax.text(shares[1][0] + shares[1][1] / 2, -0.33, f"{shares[1][1] / shares[0][1]:.1f}\u00d7 more links between\ncomplementary categories", ha="center", va="top", fontsize=5.8,
            color=OI["orange"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, 1.75)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.set_xlabel("share of links joining two products of...", fontsize=6.3)
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=c) for c in seg_cols], ["same category", "complement categories", "other categories"],
              loc="upper left", bbox_to_anchor=(-0.02, 1.02), ncol=3, frameon=False, fontsize=5.5, handlelength=1,
              columnspacing=0.6, handletextpad=0.4)
    ax.set_title("Same products, two kinds of links", fontsize=7.5, fontweight="bold", loc="left", x=-0.6)
    panel(ax, "a", x=-0.72, y=1.03)
    # b: the twist pulls complementary categories together
    ax = fig.add_subplot(gs[1])
    pp = []
    for (k, v), (k2, c) in zip(A["H2"]["per_pair"].items(), A["P_link_counts"].items()):
        a, rest = k.split(" || ")
        b = rest.split(" (ctrl")[0]
        pp.append((f"{SHORT.get(a, a)} – {SHORT.get(b, b)}", 100 * (np.exp(v) - 1), c["C"] + c["S"] + c["C_full"] == 0))
    pp.sort(key=lambda t: t[1])
    y = np.arange(len(pp))[::-1]
    for yi, (_, v, nolink) in zip(y, pp):
        col = OI["orange"] if v < 0 else OI["grey"]
        ax.plot([0, v], [yi, yi], color=col, lw=1.4, zorder=2, solid_capstyle="butt")
        ax.scatter(v, yi, s=16, color=col, zorder=3, lw=0.9, facecolors="white" if nolink else col, edgecolors=col)
    T = 100 * (np.exp(A["H2"]["T"]) - 1)
    ax.axvline(0, color=OI["black"], lw=0.6)
    ax.axvline(T, color=OI["black"], lw=0.9, ls="--", zorder=1)
    ax.text(T - 0.3, len(pp) - 0.5, f"average {T:.1f}%\np = {A['H2']['p_holm']:.3f}", ha="right", va="center", fontsize=5.8)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in pp], fontsize=5.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-17, 6)
    ax.set_ylim(-1.6, len(pp) - 0.1)
    ax.set_xlabel("change in distance between the two categories (%)\nbought-together twist vs viewed-then-bought twist", fontsize=6.3)
    ax.annotate("", xy=(-16, -1.15), xytext=(-1, -1.15), arrowprops=dict(arrowstyle="->", color=OI["orange"], lw=1))
    ax.text(-8.5, -1.0, "pulled together", ha="center", va="bottom", fontsize=5.8, color=OI["orange"])
    ax.annotate("", xy=(5.6, -1.15), xytext=(1, -1.15), arrowprops=dict(arrowstyle="->", color=OI["grey"], lw=1))
    ax.text(3.3, -1.0, "apart", ha="center", va="bottom", fontsize=5.8, color=OI["grey"])
    ax.set_title("Twisting onto bought-together links pulls\ncomplementary categories together", fontsize=7.5,
                 fontweight="bold", loc="left", x=-0.62)
    panel(ax, "b", x=-0.72, y=1.03)
    fig.savefig(os.path.join(OUT, "fig4_amazon.pdf"), bbox_inches="tight")
    plt.close(fig)
    # Supplementary: category transfer (formerly Fig. 4c)
    fig, ax = plt.subplots(figsize=(80 * MM, 50 * MM))
    K = A["knn_agreement"]
    rows = [("Phase 1 (attributes only)", "d1", OI["grey"]), ("twist, bootstrap graph", "S_boot", OI["sky"]),
            ("twist, buy-after-viewing", "S", OI["blue"]), ("twist, also-bought-only", "C", OI["orange"]),
            ("buy-after-viewing graph\n(hop distance, no attributes)", "graph_S_hop", OI["black"])]
    for yi, (lab_, key, col) in enumerate(rows[::-1]):
        ax.scatter(K[key]["micro"], yi, s=14, color=col, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=5.5)
    ax.set_xlabel("10-NN category agreement")
    ax.set_xlim(0.2, 0.7)
    fig.savefig(os.path.join(OUT, "figS_amazon_transfer.pdf"), bbox_inches="tight")
    plt.close(fig)


def figS_followups():
    F = os.path.join(WORK, "results", "synthetic", "globaltax_followups")
    P = os.path.join(WORK, "results", "synthetic", "globaltax")
    seeds, K = range(601, 613), [1.0, 1.5, 2.0]
    J = lambda *p: json.load(open(os.path.join(*p)))
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 52 * MM), gridspec_kw=dict(wspace=0.45))
    series = {
        "a": [("structured warp", lambda s, k: J(P, f"seed{s}", f"k{k:.2f}", "tax_warp.json")["d2"]["tax"][0], OI["blue"], "o"),
              ("scattered twin", lambda s, k: J(P, f"seed{s}", f"k{k:.2f}", "tax_twin.json")["d2"]["tax"][0], OI["orange"], "o"),
              ("area-free warp", lambda s, k: J(F, f"seed{s}", f"k{k:.2f}", "tax_areafree.json")["tax"], OI["green"], "s"),
              ("degree-preserving twin", lambda s, k: J(F, f"seed{s}", f"k{k:.2f}", "tax_degtwin.json")["tax"], OI["purple"], "s")],
        "b": [("structured warp", lambda s, k: J(F, f"seed{s}", f"k{k:.2f}", "degcorr_warp.json")["tax"], OI["blue"], "o"),
              ("scattered twin", lambda s, k: J(F, f"seed{s}", f"k{k:.2f}", "degcorr_twin.json")["tax"], OI["orange"], "o"),
              ("area-free warp", lambda s, k: J(F, f"seed{s}", f"k{k:.2f}", "taxdc_areafree.json")["tax"], OI["green"], "s"),
              ("degree-preserving twin", lambda s, k: J(F, f"seed{s}", f"k{k:.2f}", "taxdc_degtwin.json")["tax"], OI["purple"], "s")],
        "c": [("structured warp", lambda s, k: J(F, f"seed{s}", f"k{k:.2f}", "equaldim_warp.json")["tax"], OI["blue"], "o"),
              ("scattered twin", lambda s, k: J(F, f"seed{s}", f"k{k:.2f}", "equaldim_twin.json")["tax"], OI["orange"], "o")]}
    titles = {"a": "distance decoder, 2-D codes", "b": "degree-corrected decoder, 2-D codes", "c": "equal total dimension (4-D vs 2+2)"}
    for ax, key in zip(axes, "abc"):
        for j, (lab, f, col, mk) in enumerate(series[key]):
            m = [ci95([f(s, k) for s in seeds]) for k in K]
            ax.errorbar(np.array(K) + 0.04 * (j - 1.5), [a for a, _ in m], yerr=[h for _, h in m], fmt=mk, ms=3, color=col, capsize=1.5, lw=1,
                        label=lab, mfc="white" if mk == "s" else col)
        ax.axhline(0, color=OI["grey"], lw=0.5, ls=":")
        ax.set_xticks(K)
        ax.set_xlabel(r"warp strength $\kappa$")
        ax.set_title(titles[key], fontsize=6)
        panel(ax, key)
    axes[0].set_ylabel("tax of a shared code (task units)")
    axes[0].legend(frameon=False, fontsize=5, loc="upper left")
    fig.savefig(os.path.join(OUT, "figS_followups.pdf"), bbox_inches="tight")
    plt.close(fig)


def figS1():
    from experiments.figures_twist import forest_figure
    forest_figure("twist", out=os.path.join(OUT, "figS1_results"), plain=True)   


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    OUT = os.path.abspath(ap.parse_args().out)
    os.makedirs(OUT, exist_ok=True)
    fig2()
    fig3()
    fig4()
    figS1()
    figS_followups()
    print("ok", OUT)
