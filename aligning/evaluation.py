"""
Evaluation utilities.

Threshold-free metrics: AUROC and average precision (AP), positive class = perturbed nodes, score
orientation fixed in advance (high = anomalous) for every method.
Threshold-dependent metrics: precision/recall/F1 of the nodes flagged by the Iglewicz–Hoaglin rule
(modified z-score of the node score > 3.5), and R-precision (precision among the top-m nodes,
m = number of positives). Oracle "max-F1" thresholds are not used.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from .scores import label_free_flags


def detection_metrics(score: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    out = {}
    if np.all(np.isfinite(score)) and 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, score))
        out["ap"] = float(average_precision_score(y, score))
        m = int(y.sum())
        # R-precision with ties at the cut-off averaged over all tie orderings
        thr = np.sort(score)[::-1][m - 1]
        above = score > thr
        at = score == thr
        n_take = m - above.sum()
        exp_pos = y[above].sum() + n_take * (y[at].sum() / at.sum())
        out["r_precision"] = float(exp_pos / m)
        flags = label_free_flags(score)
        tp = int((flags & (y == 1)).sum())
        out["n_flagged"] = int(flags.sum())
        out["precision_rule"] = tp / flags.sum() if flags.sum() else 0.0
        out["recall_rule"] = tp / m
        p, r = out["precision_rule"], out["recall_rule"]
        out["f1_rule"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    else:
        for k in ["auroc", "ap", "r_precision", "precision_rule", "recall_rule", "f1_rule"]:
            out[k] = float("nan")
        out["n_flagged"] = 0
    out["prevalence"] = float(y.mean())
    return out


def bootstrap_ci(x: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, x.size), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def summarize(x: Sequence[float]) -> Dict[str, float]:
    """Mean, sd and a 95% t-interval for the mean over seeds (the percentile bootstrap
    under-covers with n = 10, simulated coverage 0.89-0.91; the t-interval gives 0.95-0.96)."""
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, ci_lo=np.nan, ci_hi=np.nan, min=np.nan, max=np.nan)
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
    h = float(stats.t.ppf(0.975, x.size - 1) * sd / np.sqrt(x.size)) if x.size > 1 else np.nan
    return dict(n=int(x.size), mean=m, sd=sd, ci_lo=m - h, ci_hi=m + h, min=float(x.min()), max=float(x.max()))


def paired_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test of a - b (paired by seed); returns mean diff, CI, p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if d.size < 3 or np.allclose(d, 0):
        return dict(n=int(d.size), mean_diff=float(d.mean()) if d.size else np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    sm = summarize(d)
    p = float(stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
    return dict(n=int(d.size), mean_diff=sm["mean"], ci_lo=sm["ci_lo"], ci_hi=sm["ci_hi"], p=p)


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted([(k, v) for k, v in pvals.items() if np.isfinite(v)], key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    for k, v in pvals.items():
        if not np.isfinite(v):
            adj[k] = float("nan")
    return adj
