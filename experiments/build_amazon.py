"""
Amazon product network: buy-after-viewing (S) and also-bought-only (C) layers (Methods, "Amazon product network").

Data: the 2014 Amazon product graph (McAuley et al. 2015), per-category metadata, one Python-dict literal per line
(parsed with ast.literal_eval only; no code is executed). Place the raw file in data/Amazon/raw/.

  categories  eligible node set (rule independent of either layer's content) and its category-path counts.
              Used to fix the complement pairs before anything else. No text, no graph-attribute comparison.
  build       node set, layers S (buy_after_viewing), C (also_bought, first k entries), C' (bought_together),
              attributes (TF-IDF of title + description -> truncated SVD -> standardised), labels (category path).

usage: python experiments/build_amazon.py categories|build [--category Musical_Instruments] [--n 1200]
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import ROOT, WORK

RAW = os.path.join(WORK, "data", "Amazon", "raw")
OUT = os.path.join(WORK, "data", "Amazon")
RANK_KEY = {"Musical_Instruments": "Musical Instruments", "Electronics": "Electronics"}


def read(category):
    with gzip.open(os.path.join(RAW, f"meta_{category}.json.gz"), "rt") as f:
        for line in f:
            yield ast.literal_eval(line)


def eligible(category, n):
    """Items with a title, a description, a salesRank in the category, and both buy_after_viewing and also_bought
    lists; the n best-ranked. The rule uses list *presence* only, not list content."""
    key = RANK_KEY[category]
    items = []
    for d in read(category):
        rel = d.get("related", {})
        rank = d.get("salesRank", {}).get(key)
        if d.get("title") and d.get("description") and rank is not None and rel.get("buy_after_viewing") and rel.get("also_bought"):
            items.append(d)
    items.sort(key=lambda d: (d["salesRank"][RANK_KEY[category]], d["asin"]))
    return items[:n], len(items)


def path_of(d, category):
    """The category path under the top-level category (the first path that starts with it)."""
    top = RANK_KEY[category]
    for p in d.get("categories", []):
        if p and p[0] == top:
            return tuple(p[1:])
    return tuple()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["categories", "build"])
    ap.add_argument("--category", default="Musical_Instruments")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--svd", type=int, default=32)
    args = ap.parse_args()
    items, n_elig = eligible(args.category, args.n)
    paths = [path_of(d, args.category) for d in items]
    if args.mode == "categories":
        print(f"eligible {n_elig}; selected {len(items)}")
        for lev in (1, 2):
            c = Counter(p[:lev] for p in paths)
            print(f"\nlevel {lev + 1} paths (count):")
            for k, v in c.most_common():
                print(f"  {v:5d}  {' > '.join(k) if k else '(none)'}")
        return
    build(args, items, paths)


def build(args, items, paths):
    """S = buy_after_viewing; C = first k in-set also_bought entries that are not S-links (disjoint);
    final node set = iterated removal of items isolated in S or C; TF-IDF fitted on the final set."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    asin_all = [d["asin"] for d in items]
    idx = {a: i for i, a in enumerate(asin_all)}
    n0 = len(items)

    def lists(key):
        return [[idx[b] for b in d.get("related", {}).get(key, []) if b in idx and b != d["asin"]] for d in items]

    def sym(E):
        A = np.zeros((n0, n0))
        for i, lst in enumerate(E):
            A[i, lst] = 1.0
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0.0)
        return A

    S = sym(lists("buy_after_viewing"))
    ab = lists("also_bought")
    C = sym([[b for b in lst if S[i, b] == 0][:args.k] for i, lst in enumerate(ab)])
    C_full = sym(ab)
    C2 = sym(lists("bought_together"))
    keep = np.ones(n0, bool)
    while True:
        Sk, Ck = S[np.ix_(keep, keep)], C[np.ix_(keep, keep)]
        bad = (Sk.sum(1) == 0) | (Ck.sum(1) == 0)
        if not bad.any():
            break
        keep[np.flatnonzero(keep)[bad]] = False
    sel = np.flatnonzero(keep)
    items = [items[i] for i in sel]
    paths = [paths[i] for i in sel]
    S, C, C_full, C2 = (M[np.ix_(sel, sel)] for M in (S, C, C_full, C2))
    assert not (S * C).any()
    text = [(d["title"] + ". " + d["description"]) for d in items]
    tf = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.5, stop_words="english", strip_accents="unicode")
    Xs = TruncatedSVD(args.svd, random_state=0).fit_transform(tf.fit_transform(text))
    X = (Xs - Xs.mean(0)) / Xs.std(0)
    key = RANK_KEY[args.category]

    def price(d):
        try:
            return float(d.get("price"))
        except (TypeError, ValueError):
            return np.nan
    pr = np.array([price(d) for d in items])
    miss = ~np.isfinite(pr) | (pr <= 0)
    lp = np.log(np.where(miss, np.nanmedian(pr[~miss]), pr))
    P = np.column_stack([np.log([d["salesRank"][key] for d in items]), lp, miss.astype(float),
                         np.log1p([len(d["related"].get("also_bought", [])) for d in items]),
                         np.log1p([len(d["related"].get("buy_after_viewing", [])) for d in items])])
    P = (P - P.mean(0)) / np.where(P.std(0) > 0, P.std(0), 1.0)
    out = os.path.join(OUT, args.category)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "X.npy"), X)
    np.save(os.path.join(out, "X_popularity.npy"), P)
    for nm, A in (("S", S), ("C", C), ("C2", C2), ("C_full", C_full)):
        np.save(os.path.join(out, f"A_{nm}.npy"), A)
    lv = lambda p, k: " > ".join(p[:k + 1]) if len(p) > k else ""
    meta = dict(asin=[d["asin"] for d in items], title=[d["title"] for d in items],
                level2=[lv(p, 0) for p in paths], level3=[lv(p, 1) for p in paths],
                salesrank=[d["salesRank"][key] for d in items])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    info = dict(n_candidates=n0, n=len(items), svd=args.svd, k=args.k, vocabulary=len(tf.vocabulary_),
                edges={nm: int(np.triu(A, 1).sum()) for nm, A in (("S", S), ("C", C), ("C2", C2), ("C_full", C_full))},
                mean_degree={nm: float(A.sum(1).mean()) for nm, A in (("S", S), ("C", C))},
                level3_counts=dict(Counter(meta["level3"]).most_common()))
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "level3_counts"}, indent=1))
    print("\n".join(f"{v:5d} {k}" for k, v in info["level3_counts"].items()))


if __name__ == "__main__":
    main()
