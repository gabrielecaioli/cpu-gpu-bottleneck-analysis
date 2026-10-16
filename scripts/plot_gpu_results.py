#!/usr/bin/env python3
"""
Plots the GPU benchmark results produced by scripts/benchmark_gpu.py
(results/gpu/res_<version>_w<workload>_TpB<threads per block>.json).

The time considered is the kernel time ("SEARCH" in the timing breakdown),
averaged over the seeds, with its 95% confidence interval.

Generated graphs (in results/gpu/graphs/ by default):
    <version>_performance.png / <version>_speedup.png     all the workloads of a version
    compare_performance_<workload>.png / compare_speedup_<workload>.png
    <version>_w<workload>_tpb_performance.png             if several threads-per-block were tested
It also prints the throughput (values searched per second) of every version.

Example:
    python3 scripts/plot_gpu_results.py
    python3 scripts/plot_gpu_results.py --tpb v1=32 v2_1=32 v2_2=32 v3=1024 --compare-workload 10000000
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VERSION_ORDER = ["v1", "v2_1", "v2_2", "v3"]
DEFAULT_TPB = {"v1": 32, "v2_1": 32, "v2_2": 32, "v3": 1024}
FILENAME_RE = re.compile(r"res_(v[\d_]+?)_w(\d+)_TpB(\d+)\.json$")


def printable_version(version):
    """'v2_1' -> '2.1'"""
    return version.lstrip("v").replace("_", ".")


def printable_workload(workload):
    for value, suffix in ((10**9, "G"), (10**6, "M"), (10**3, "K")):
        if workload >= value and workload % value == 0:
            return f"{workload // value}{suffix}"
    return str(workload)


def confidence_interval(data, confidence=0.95):
    """Half-width of the confidence interval of a list of values."""
    data = np.array(data)
    if len(data) < 2:
        return 0.0
    return stats.sem(data) * stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)


def load_files(results_dir):
    """
    :return: {(version, workload, threads_per_block): path}
    """
    files = {}
    for path in glob.glob(os.path.join(results_dir, "res_*.json")):
        match = FILENAME_RE.search(os.path.basename(path))
        if match:
            version, workload, tpb = match.group(1), int(match.group(2)), int(match.group(3))
            files[(version, workload, tpb)] = path
    return files


def search_times(path, min_threads=2**5, max_threads=2**17, only_powers_of_two=True):
    """
    Reads a results file and returns, for every total number of threads, the
    mean kernel ("SEARCH") time and its confidence interval.

    :return: (x, y, ci) - total threads, mean time (ms), confidence interval (ms)
    """
    with open(path) as f:
        data = json.load(f)

    by_threads = defaultdict(list)
    for run in data:
        for timing in run.get("timings", []):
            if timing.get("description") == "SEARCH":
                by_threads[run["Total_num_threads"]].append(timing["time"])

    x, y, ci = [], [], []
    for threads in sorted(by_threads):
        if not (min_threads <= threads <= max_threads):
            continue
        if only_powers_of_two and threads & (threads - 1):
            continue
        x.append(threads)
        y.append(float(np.mean(by_threads[threads])))
        ci.append(float(confidence_interval(by_threads[threads])))
    return x, y, ci


def finish(ax, title, ylabel, x_values, path, log_y=False):
    ax.set_xlabel("Total number of threads", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log", base=2)
    if x_values:
        ax.set_xticks(sorted(set(x_values)))
    ax.grid(True)
    ax.legend()
    ax.figure.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ax.figure.savefig(path, dpi=200)
    plt.close(ax.figure)
    print(f"Graph saved to {os.path.relpath(path, ROOT)}")


def plot_lines(series, title, path_prefix):
    """
    series: list of (label, x, y, ci). Saves a performance and a speedup graph.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    all_x = []
    for label, x, y, ci in series:
        ax.errorbar(x, y, yerr=ci, fmt='o-', capsize=5, label=label)
        all_x += x
    finish(ax, f"Performance - {title}", "Mean execution time (ms)", all_x, f"{path_prefix}_performance.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, x, y, ci in series:
        speedup = [y[0] / t for t in y]
        ax.plot(x, speedup, 'o-', label=label)
    finish(ax, f"Speedup - {title}", "Speedup", all_x, f"{path_prefix}_speedup.png", log_y=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results", "gpu"))
    parser.add_argument("--out", default=None, help="output directory (default: <results-dir>/graphs)")
    parser.add_argument("--tpb", nargs="+", default=None,
                        help="threads per block to use for each version, e.g. v1=32 v3=1024 (default: v1..v2_2=32, v3=1024)")
    parser.add_argument("--compare-workload", type=int, default=10_000_000)
    parser.add_argument("--throughput-threads", type=int, nargs="+", default=[2**15, 40960],
                        help="total numbers of threads at which the throughput is printed")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.results_dir, "graphs")
    tpb_of = dict(DEFAULT_TPB)
    for item in args.tpb or []:
        version, tpb = item.split("=")
        tpb_of[version] = int(tpb)

    files = load_files(args.results_dir)
    if not files:
        print(f"No results found in {args.results_dir}")
        return

    versions = sorted({v for v, _, _ in files}, key=lambda v: VERSION_ORDER.index(v) if v in VERSION_ORDER else 99)

    def file_for(version, workload):
        """File of the preferred threads-per-block, or any available one."""
        if (version, workload, tpb_of.get(version)) in files:
            return files[(version, workload, tpb_of.get(version))], tpb_of.get(version)
        for (v, w, tpb), path in sorted(files.items()):
            if v == version and w == workload:
                return path, tpb
        return None, None

    # 1 - Every workload of each version
    for version in versions:
        series = []
        for workload in sorted({w for v, w, _ in files if v == version}):
            path, _ = file_for(version, workload)
            x, y, ci = search_times(path)
            if x:
                series.append((f"Workload {printable_workload(workload)}", x, y, ci))
        if series:
            plot_lines(series, f"Version {printable_version(version)}", os.path.join(out_dir, version))

    # 2 - Version comparison on a fixed workload
    series = []
    for version in versions:
        path, tpb = file_for(version, args.compare_workload)
        if path:
            x, y, ci = search_times(path)
            series.append((f"Version {printable_version(version)} - {tpb} TpB", x, y, ci))
    if len(series) > 1:
        plot_lines(series, f"Version comparison - Workload {printable_workload(args.compare_workload)}",
                   os.path.join(out_dir, f"compare_{args.compare_workload}"))

    # 3 - Threads-per-block comparison, where several were tested
    groups = defaultdict(list)
    for (version, workload, tpb), path in files.items():
        groups[(version, workload)].append((tpb, path))
    for (version, workload), entries in sorted(groups.items()):
        if len(entries) < 2:
            continue
        series = []
        for tpb, path in sorted(entries):
            x, y, ci = search_times(path)
            series.append((f"{tpb} threads per block", x, y, ci))
        plot_lines(series, f"Version {printable_version(version)} - Workload {printable_workload(workload)}",
                   os.path.join(out_dir, f"{version}_w{workload}_tpb"))

    # 4 - Throughput
    print("\nThroughput (values searched per second, kernel time only):")
    for version in versions:
        path, tpb = file_for(version, args.compare_workload)
        if not path:
            continue
        x, y, _ = search_times(path, only_powers_of_two=False, max_threads=2**20)
        for threads in args.throughput_threads:
            if threads in x:
                mean_ms = y[x.index(threads)]
                throughput = args.compare_workload / (mean_ms / 1000)
                print(f"  Version {printable_version(version):<4} {threads:>7} threads ({tpb} TpB): "
                      f"{mean_ms:8.2f} ms -> {throughput:,.0f} values/s")


if __name__ == "__main__":
    main()
