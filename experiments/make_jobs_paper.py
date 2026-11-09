"""
Job lists of every run behind the paper. Each list is a text file of shell commands, one per job, run in parallel by
run_queue.py. Lists must be run in the order below (later lists read the outputs of earlier ones); the aggregation,
figure and table steps are listed in REPRODUCING.md.

  results/jobs_planted.txt     planted disagreements: seeds 301-320, both objectives, plus block designs
  results/jobs_pilots.txt      development pilots: twist seeds 900-902 (incl. longer budget), tax seeds 951-953
  results/jobs_tax_gen.txt     tax experiment: graphs of seeds 601-612, kappa in {0, 1, 1.5, 2}
  results/jobs_tax.txt         tax experiment: tax of each graph and twist of each graph and its bootstrap
  results/jobs_fu_gen.txt      further analyses of the tax: area-free warps and degree-preserving twins
  results/jobs_fu.txt          further analyses of the tax: degree-corrected decoders, equal total dimension, new graphs
  results/jobs_amazon.txt      product network: ten Phase 1 seeds x {S, C} layers x {twist, bootstrap}
  results/jobs_estimators.txt  geodesic estimators against a reference solver

usage: python experiments/make_jobs_paper.py
       python experiments/run_queue.py results/jobs_planted.txt 8      (8 parallel workers; one thread each)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.abspath(os.environ.get("ALIGNING_WORKDIR", ROOT))
PY = sys.executable
ENV = "OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1"
TAX_SEEDS = range(601, 613)
KAPPAS = (0.0, 1.0, 1.5, 2.0)


def job_list(name, cmds):
    lines = []
    for cmd, log in cmds:
        os.makedirs(os.path.join(WORK, os.path.dirname(log)), exist_ok=True)
        lines.append(f"cd {ROOT} && {ENV} {PY} {cmd} > {os.path.join(WORK, log)} 2>&1")
    open(os.path.join(WORK, "results", name), "w").write("\n".join(lines) + "\n")
    print(f"{name}: {len(lines)} jobs")


os.makedirs(os.path.join(WORK, "results"), exist_ok=True)

planted = []
for s in range(301, 321):
    planted.append((f"experiments/run_synthetic.py --seed {s} --conditions R70,R70N,D70,D70N,BAND,BANDN --p2 K=1,loss=bce "
                    f"--tag twist_blocks --no-pygod", f"results/synthetic/twist_blocks/logs/seed{s}.log"))
    for arm in ("bce1", "mse15"):
        planted.append((f"experiments/run_twist.py --seed {s} --arm {arm} --designs none,shear,barrier,corridor,none_boot --tag twist",
                        f"results/synthetic/twist/logs/{arm}_seed{s}.log"))
job_list("jobs_planted.txt", planted)

pilots = []
for s in (900, 901, 902):
    for arm in ("bce1", "mse15"):
        pilots.append((f"experiments/run_twist.py --seed {s} --arm {arm} --designs "
                       f"none,shear,barrier,corridor,none_boot,shear_boot,barrier_boot,corridor_boot --tag twist_pilot",
                       f"results/synthetic/twist_pilot/logs/{arm}_seed{s}.log"))
for s in (900, 901):
    pilots.append((f"experiments/run_twist.py --seed {s} --arm bce1 --designs none,shear,barrier --epochs 1200 --tag twist_pilot_e1200",
                   f"results/synthetic/twist_pilot_e1200/logs/seed{s}.log"))
for s in (951, 952, 953):
    for k in (0.0, 0.3, 0.6, 1.0, 1.5):
        pilots.append((f"experiments/run_global_twist.py --seed {s} --kappa {k} --variants '' --tag globaltax_pilot",
                       f"results/synthetic/globaltax_pilot/logs/gen_s{s}_k{k:.2f}.log"))
job_list("jobs_pilots.txt", pilots)

job_list("jobs_tax_gen.txt", [(f"experiments/run_global_twist.py --seed {s} --kappa {k} --variants ''",
                               f"results/synthetic/globaltax/logs/gen_s{s}_k{k:.2f}.log") for s in TAX_SEEDS for k in KAPPAS])

tax = []
for s in TAX_SEEDS:
    for k in KAPPAS:
        d = f"results/synthetic/globaltax/seed{s}/k{k:.2f}"
        graphs = ("warp",) if k == 0 else ("warp", "twin")
        variants = "warp,warp_boot" if k == 0 else "warp,warp_boot,twin,twin_boot"
        tax.append((f"experiments/run_global_twist.py --seed {s} --kappa {k} --variants {variants}",
                    f"results/synthetic/globaltax/logs/twist_s{s}_k{k:.2f}.log"))
        for g in graphs:
            tax.append((f"experiments/global_tax_run.py --dir {os.path.join(WORK, d)} --seed {s} --graph {g}",
                        f"results/synthetic/globaltax/logs/tax_s{s}_k{k:.2f}_{g}.log"))
job_list("jobs_tax.txt", tax)

job_list("jobs_fu_gen.txt", [(f"experiments/tax_followups.py gen --seed {s} --kappa {k}",
                              f"results/synthetic/globaltax_followups/logs/gen_{s}_{k:.2f}.log") for s in TAX_SEEDS for k in KAPPAS[1:]])

fu = []
for s in TAX_SEEDS:
    for k in KAPPAS:
        d = os.path.join(WORK, f"results/synthetic/globaltax/seed{s}/k{k:.2f}")
        for g in (("warp",) if k == 0 else ("warp", "twin")):
            for mode in ("degcorr", "equaldim"):
                fu.append((f"experiments/tax_followups.py {mode} --dir {d} --seed {s} --graph {g}",
                           f"results/synthetic/globaltax_followups/logs/{mode}_{s}_{k:.2f}_{g}.log"))
        if k > 0:
            for mode in ("taxnew", "taxnew_dc"):
                for g in ("areafree", "degtwin"):
                    fu.append((f"experiments/tax_followups.py {mode} --seed {s} --kappa {k} --graph {g}",
                               f"results/synthetic/globaltax_followups/logs/{mode}_{s}_{k:.2f}_{g}.log"))
job_list("jobs_fu.txt", fu)

amazon = []
for s in range(1, 11):
    for v in ("S", "C", "S_boot", "C_boot"):
        amazon.append((f"experiments/run_amazon_twist.py --seed {s} --variant {v}", f"results/amazon/logs/text_{v}_seed{s}.log"))
for s in range(1, 6):
    for v in ("S", "C"):
        amazon.append((f"experiments/run_amazon_twist.py --seed {s} --variant {v} --attributes popularity",
                       f"results/amazon/logs/popularity_{v}_seed{s}.log"))
job_list("jobs_amazon.txt", amazon)

job_list("jobs_estimators.txt", [("experiments/validate_estimators.py", "results/logs/validate_estimators.log")])
