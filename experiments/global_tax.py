"""
The cost of a shared representation (the "tax") for distance-based embeddings (Methods, "Tax measurement").

For a graph A and attributes X: separate codes (z_g on links only, z_a on attribute log-stress only) and shared codes
(one z on L_link + lam L_attr, lam over a grid), all trained to convergence from a PCA start. Per task:
  link   = test AUC of held-out links vs held-out non-links (distance kernel)
  attr   = recall@10 of X-space neighbours
tax = min over lam of max(link_sep - link_shared(lam), attr_sep - attr_shared(lam)), with lam chosen on validation
links. Dimensions: shared d vs separate (d, d).

usage (import): from experiments.global_tax import tax
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats

torch.set_default_dtype(torch.float64)
LAMBDAS = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
MAX_EPOCHS, CHECK, REL_TOL = 10000, 200, 1e-5


def auc(pos, neg):
    r = stats.rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def knn(M, k=10):
    D = ((M[:, None] - M[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, 1)[:, :k]


def recall10(Z, nnX):
    nz = knn(Z)
    return float(np.mean([len(set(nnX[i]) & set(nz[i])) / 10 for i in range(len(Z))]))


def split(A, rng):
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    y = (A[iu] > 0).astype(float)
    pos, neg = rng.permutation(np.flatnonzero(y == 1)), rng.permutation(np.flatnonzero(y == 0))
    m = len(pos) // 10
    test, val = (pos[:m], neg[:m]), (pos[m:2 * m], neg[m:2 * m])
    train = np.ones(len(y), bool)
    train[np.r_[pos[:2 * m], neg[:2 * m]]] = False
    return iu, y, train, test, val


def fit(dim, iu, y, train, logdx, lam_link, lam_attr, Z0, seed):
    torch.manual_seed(seed)
    Z = torch.nn.Parameter(torch.as_tensor(Z0[:, :dim].copy()) + 0.01 * torch.randn(Z0.shape[0], dim))
    b, w, c = (torch.nn.Parameter(torch.tensor(v)) for v in (0.0, 0.0, 0.0))
    opt = torch.optim.Adam([Z, b, w, c], lr=0.02)
    I, J = torch.as_tensor(iu[0]), torch.as_tensor(iu[1])
    yt, tr, lx = torch.as_tensor(y), torch.as_tensor(train), torch.as_tensor(logdx)
    pw = torch.tensor(float((y[train] == 0).sum() / max((y[train] == 1).sum(), 1)))
    prev = None
    for ep in range(MAX_EPOCHS):
        opt.zero_grad()
        d2 = ((Z[I] - Z[J]) ** 2).sum(-1)
        loss = torch.tensor(0.0)
        if lam_link > 0:
            loss = loss + lam_link * torch.nn.functional.binary_cross_entropy_with_logits((b - torch.exp(w) * d2)[tr], yt[tr], pos_weight=pw)
        if lam_attr > 0:
            loss = loss + lam_attr * ((0.5 * torch.log(d2 + 1e-12) - lx - c) ** 2).mean()
        loss.backward()
        opt.step()
        if ep % CHECK == 0:
            v = float(loss)
            if prev is not None and abs(prev - v) <= REL_TOL * abs(prev):
                break
            prev = v
    Zn = Z.detach().numpy()
    return Zn, -((Zn[iu[0]] - Zn[iu[1]]) ** 2).sum(-1), ep + 1


def tax(A, X, seed, dim=4, model_seeds=(0, 1, 2)):
    rng = np.random.default_rng([seed, 2222])
    iu, y, train, (tp, tn), (vp, vn) = split(A, rng)
    logdx = np.log(np.linalg.norm(X[iu[0]] - X[iu[1]], axis=1) + 1e-12)
    Xc = X - X.mean(0)
    Z0 = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:dim].T
    Z0 = Z0 / Z0.std()
    nnX = knn(X)
    out = dict(tax=[], sep_link=[], sep_attr=[], lam=[], epochs_max=0)
    for ms in model_seeds:
        Zg, sg, e1 = fit(dim, iu, y, train, logdx, 1.0, 0.0, Z0, ms)
        Za, _, e2 = fit(dim, iu, y, train, logdx, 0.0, 1.0, Z0, ms)
        L_s, R_s = auc(sg[tp], sg[tn]), recall10(Za, nnX)
        Lv_s = auc(sg[vp], sg[vn])
        best = None
        for lam in LAMBDAS:
            Zm, sm, e3 = fit(dim, iu, y, train, logdx, 1.0, lam, Z0, ms)
            Rm = recall10(Zm, nnX)
            crit = max(Lv_s - auc(sm[vp], sm[vn]), R_s - Rm)                 # chosen on validation links
            if best is None or crit < best[0]:
                best = (crit, max(L_s - auc(sm[tp], sm[tn]), R_s - Rm), lam)
            out["epochs_max"] = max(out["epochs_max"], e1, e2, e3)
        out["tax"].append(best[1])
        out["lam"].append(best[2])
        out["sep_link"].append(L_s)
        out["sep_attr"].append(R_s)
    out["tax_mean"] = float(np.mean(out["tax"]))
    return out
