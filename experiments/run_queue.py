"""Run a job file (one shell command per line) with N parallel workers; log start/end/exit code.
usage: python experiments/run_queue.py results/jobs_all.txt 16"""
import subprocess, sys, time, datetime
from concurrent.futures import ThreadPoolExecutor
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
log = open(sys.argv[1].replace(".txt", "_status.log"), "a", buffering=1)
def run(i):
    t0 = time.time()
    log.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} START {i} {jobs[i][150:260]}\n")
    rc = subprocess.run(jobs[i], shell=True).returncode
    log.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} END   {i} rc={rc} {time.time()-t0:.0f}s {jobs[i][150:260]}\n")
    return rc
with ThreadPoolExecutor(max_workers=N) as ex:
    rcs = list(ex.map(run, range(len(jobs))))
log.write(f"ALL DONE: {sum(r == 0 for r in rcs)}/{len(rcs)} succeeded\n")
