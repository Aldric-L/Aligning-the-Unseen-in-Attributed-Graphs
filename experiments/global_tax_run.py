"""
The tax of one graph (warp or twin) of one instance of the tax experiment.
usage: python experiments/global_tax_run.py --dir results/synthetic/globaltax/seed601/k1.50 --seed 601 --graph warp
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.global_tax import tax
from experiments.common import jsonable

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--graph", required=True, choices=["warp", "twin"])
a = ap.parse_args()
inp = torch.load(os.path.join(a.dir, "inputs.pt"), weights_only=False)
X, A = np.asarray(inp["X"], float), np.asarray(inp[f"A_{a.graph}"], float)
out = dict(d2=tax(A, X, a.seed, dim=2, model_seeds=(0, 1)), d4=tax(A, X, a.seed, dim=4, model_seeds=(0,)))
json.dump(jsonable(out), open(os.path.join(a.dir, f"tax_{a.graph}.json"), "w"))
