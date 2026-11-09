"""
Supplementary tables, written from results/ (no hand-typed numbers).

  tabS_estimators.tex    geodesic estimators against a reference solver (Supplementary Table S2)
  tabS_followups.tex     further analyses of the tax (Supplementary Table S3)
  tabS_tax_cells.tex     tax experiment, cell means
  tabS_amazon_pairs.tex  product network, per-pair difference-in-differences
  tabS_amazon_loo.tex    product network, leave-one-pair-out

usage: python experiments/tables_supp.py [--out DIR]      (default: outputs/tables/)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ROOT, WORK

OUT = os.path.join(ROOT, "outputs", "tables")
J = lambda *p: json.load(open(os.path.join(WORK, *p)))


def tex(s):
    return (s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace(">", "$>$").replace("<", "$<$")
            .replace("−", "$-$").replace("δ", "$\\delta$"))


def p_fmt(p):
    return f"{p:.2g}" if p >= 1e-3 else "$<10^{-3}$"


def write(name, body):
    open(os.path.join(OUT, name), "w").write(body)


def estimators():
    E = J("results", "estimator_validation.json")
    rows = []
    for e in E:
        cells = []
        for k in ("hybrid", "grid", "line"):
            v = e[k]
            cells.append(f"{100 * v['median_abs_rel']:.2f} / {100 * v['p95_abs_rel']:.1f} / {100 * v['max_abs_rel']:.1f}")
        rows.append(f"{tex(e['label'])} & " + " & ".join(cells) + " \\\\")
    write("tabS_estimators.tex", "\\begin{tabular}{lccc}\n\\toprule\nMetric & hybrid (used) & grid only & straight line \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def followups():
    R = J("results", "synthetic", "globaltax_followups", "aggregate.json")
    rows = [("Warp $-$ degree-preserving twin (distance decoder)", "warp_minus_degtwin"),
            ("Area-free warp $-$ aligned graph", "areafree_minus_aligned"),
            ("Area-free warp $-$ degree-preserving twin", "areafree_minus_degtwin"),
            ("Degree-corrected: warp $-$ twin", "degcorr_warp_minus_twin"),
            ("Degree-corrected: area-free warp $-$ degree-preserving twin", "degcorr_areafree_minus_degtwin"),
            ("Degree-corrected: area-free warp $-$ aligned graph", "degcorr_areafree_minus_aligned"),
            ("Equal total dimension: warp $-$ twin", "equaldim_warp_minus_twin")]
    body = []
    for lab, k in rows:
        v = R[k]
        body.append(f"{lab} & {v['mean']:+.4f} & [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] & {v['positive']}/{v['n']} & {v['p']:.2g} \\\\")
    write("tabS_followups.tex", "\\begin{tabular}{p{7.2cm}rcrr}\n\\toprule\nContrast (mean over $\\kappa\\in\\{1,1.5,2\\}$ per seed) & mean & 95\\% CI & seeds $>0$ & $p$ \\\\\n\\midrule\n"
          + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")


def tax_cells():
    R = J("results", "synthetic", "globaltax", "analysis_d2.json")
    rows = []
    for k, v in R["by_kappa"].items():
        g, kap = k.split("_k")
        rows.append(f"{g} & {kap} & {v['tax']:.4f} & {v['excess']:.3f} & {v['sigma_delta']:.3f} & {v['sigma_sp']:.3f} & "
                    f"{v['sd_log_degree']:.3f} & {v['clustering']:.3f} & {v['homophily_X']:.3f} \\\\")
    write("tabS_tax_cells.tex", "\\begin{tabular}{lcrrrrrrr}\n\\toprule\ngraph & $\\kappa$ & tax ($d=2$) & excess $\\sigma_\\delta$ & $\\sigma_\\delta$ & $\\sigma_{sp}$ & SD log deg. & clustering & homophily \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def amazon():
    A = J("results", "amazon", "analysis.json")
    rows = []
    for (k, v), (k2, c) in zip(A["H2"]["per_pair"].items(), A["P_link_counts"].items()):
        a, rest = k.split(" || ")
        b, ctrl = rest.split(" (ctrl ")
        rows.append(f"{tex(a).replace(' $>$ ', ' / ')} & {tex(b).replace(' $>$ ', ' / ')} & {tex(ctrl.rstrip(')')).replace(' $>$ ', ' / ')} & {v:+.3f} & {c['C']} & {c['S']} \\\\")
    write("tabS_amazon_pairs.tex", "\\begin{tabular}{p{3.2cm}p{3.4cm}p{3.2cm}rrr}\n\\toprule\ninstrument category $A$ & accessory category $B$ & control $B'$ & DiD & $C$ links & $S$ links \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    loo = J("results", "amazon", "loo_pairs.json")
    lrows = [f"{tex(r['dropped']).replace(' $>$ ', ' / ').replace(' || ', ' -- ')} & {r['T']:+.4f} & {r['q05']:+.4f} & {p_fmt(r['p'])} \\\\" for r in loo]
    write("tabS_amazon_loo.tex", "\\begin{tabular}{p{9.0cm}rrr}\n\\toprule\nPair left out & $T$ & null 5\\% & $p$ \\\\\n\\midrule\n" + "\n".join(lrows) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    OUT = os.path.abspath(ap.parse_args().out)
    os.makedirs(OUT, exist_ok=True)
    estimators()
    followups()
    tax_cells()
    amazon()
    print("ok", OUT)
