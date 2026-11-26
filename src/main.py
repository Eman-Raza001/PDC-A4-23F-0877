"""
main.py - runs the whole project end to end:  python src/main.py

Workload (Option 3): Monte-Carlo estimation of pi with n random samples.
  * serial   : fastest single-process version (NumPy, vectorised, chunked)  -> T_S
  * parallel : same work split over p worker processes (multiprocessing)    -> T_P

Steps: 1) time serial + parallel for every (n, p), 5 runs each
       2) save every measurement to results/raw_timings.csv
       3) call metrics.py (results/metrics.csv + terminal table)
       4) call plot.py (figures/*.png)

Optional: python src/main.py --quick   (tiny test run: n up to 1e6, 1 run)
"""
import csv
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

import metrics
import plot

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

WORKLOAD_SERIAL = "pi_serial"
WORKLOAD_PARALLEL = "pi_parallel"
CHUNK = 1 << 20          # samples generated at a time (keeps arrays small / cache friendly)


def count_hits(n, seed):
    """Throw n random points in the unit square; return how many land inside the quarter circle."""
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    while done < n:
        m = min(CHUNK, n - done)
        x = rng.random(m)
        y = rng.random(m)
        hits += int(np.count_nonzero(x * x + y * y <= 1.0))
        done += m
    return hits


def serial_pi(n, seed=12345):
    """Serial version (T_S): one process does all n samples."""
    return 4.0 * count_hits(n, seed) / n


def _worker(args):
    n_local, seed = args
    return count_hits(n_local, seed)


def parallel_pi(n, p, seed=12345):
    """Parallel version (T_P): n samples split over p worker processes."""
    # independent random streams for each worker
    seeds = np.random.SeedSequence(seed).spawn(p)
    base, extra = divmod(n, p)
    jobs = [(base + (1 if i < extra else 0), seeds[i]) for i in range(p)]
    with mp.Pool(processes=p) as pool:          # pool start-up cost is part of the measured overhead
        hits = sum(pool.map(_worker, jobs))
    return 4.0 * hits / n


def write_system_info():
    """Save machine details (for the README) to results/system_info.txt."""
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info = [
        f"CPU model      : {cpu}",
        f"Logical CPUs   : {os.cpu_count()}",
        f"OS             : {platform.platform()}",
        f"Python         : {platform.python_version()}",
        f"NumPy          : {np.__version__}",
    ]
    (RESULTS / "system_info.txt").write_text("\n".join(info) + "\n")
    print("\n".join(info))


def run_benchmarks(sizes, procs, runs):
    rows = []
    for n in sizes:
        for run in range(1, runs + 1):
            t0 = time.perf_counter()
            est = serial_pi(n)
            t = time.perf_counter() - t0
            rows.append([WORKLOAD_SERIAL, n, 1, run, f"{t:.6f}"])
            print(f"serial   n={n:<10} p=1 run={run}  time={t:8.4f}s  pi~{est:.5f}", flush=True)
            for p in procs:
                t0 = time.perf_counter()
                est = parallel_pi(n, p)
                t = time.perf_counter() - t0
                rows.append([WORKLOAD_PARALLEL, n, p, run, f"{t:.6f}"])
                print(f"parallel n={n:<10} p={p} run={run}  time={t:8.4f}s  pi~{est:.5f}", flush=True)
    return rows


def main():
    quick = "--quick" in sys.argv
    sizes = [10**5, 10**6] if quick else [10**6, 10**7, 10**8]
    procs = [1, 2, 4, 8]
    runs = 1 if quick else 5

    RESULTS.mkdir(exist_ok=True)
    write_system_info()

    serial_pi(10**5)                    # warm-up (not recorded)
    rows = run_benchmarks(sizes, procs, runs)

    csv_path = RESULTS / "raw_timings.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "n", "p", "run", "time_sec"])
        w.writerows(rows)
    print(f"\nSaved {len(rows)} measurements to {csv_path}")

    metrics.main()                      # results/metrics.csv + terminal table
    plot.main()                         # figures/*.png


if __name__ == "__main__":              # required for multiprocessing on Windows
    main()