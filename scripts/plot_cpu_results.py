#!/usr/bin/env python3
"""
Plots the CPU benchmark results (execution time and speedup with 95%
confidence intervals) from the JSON files in results/cpu/.

For every version vX it reads vX.json and, if present, the vX_workload_N.json
files, which add workloads or replace the matching one of vX.json (they
contain re-runs of single workloads).

Generated graphs (in results/cpu/graphs/ by default):
    vX/vX_execution_time_up_to_workload_K.png   one per K, workloads 1..K
    vX/vX_speedup_up_to_workload_K.png
    compare/compare_execution_time_N.png         all versions, workload with N values to search
    compare/compare_speedup_N.png

Example:
    python3 scripts/plot_cpu_results.py
    python3 scripts/plot_cpu_results.py --versions v2 v3 --compare-workloads 1000000 10000000
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = ['#1f77b4', '#32cd32', '#f4a300', '#d62728', '#9467bd']


def color(index):
    return COLORS[index % len(COLORS)]


def printable(version):
    """'v2_1' -> '2.1'"""
    return version.lstrip("v").replace("_", ".")


def confidence_interval(data, confidence=0.95):
    """
    Computes the confidence interval of an array of values.

    :param data: list of numeric values
    :param confidence: confidence level (default 0.95)
    :return: (mean, margin) - mean of the data and half-width of the confidence interval
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # standard error of the mean
    margin = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, margin


def load_results(results_dir, version):
    """
    Loads the results of a version: vX.json plus the optional vX_workload_N.json
    files, which replace the workload with the same number of values to search.

    :return: list of workloads, sorted by number of values to search
    """
    workloads = {}

    main_file = os.path.join(results_dir, f"{version}.json")
    if os.path.exists(main_file):
        with open(main_file) as f:
            for workload in json.load(f):
                workloads[int(workload["num_to_search"])] = workload

    for single_file in glob.glob(os.path.join(results_dir, f"{version}_workload_*.json")):
        with open(single_file) as f:
            workload = json.load(f)
        workloads[int(workload["num_to_search"])] = workload

    return [workloads[k] for k in sorted(workloads)]


def workload_stats(workload):
    """
    :return: (thread_counts, means, margins) of the execution times of a workload
    """
    thread_counts, means, margins = [], [], []
    for thread_config in workload["threads"]:
        execution_times = [run["execution_time"] for run in thread_config["runs"]]
        mean, margin = confidence_interval(execution_times)
        thread_counts.append(thread_config["num_threads"])
        means.append(mean)
        margins.append(margin)
    return thread_counts, means, margins


def speedup_stats(means, margins):
    """
    Speedup with respect to the single-thread execution, with the bounds
    derived from the confidence interval of the execution times.
    """
    baseline = means[0]  # time with 1 thread
    speedup = [baseline / m for m in means]
    lower_bound = [baseline / (m + err) for m, err in zip(means, margins)]
    upper_bound = [baseline / (m - err) if m > err else baseline / 0.0001 for m, err in zip(means, margins)]
    return speedup, lower_bound, upper_bound


def finish_plot(title, ylabel, thread_counts, path):
    plt.title(title)
    plt.xlabel("Number of Threads")
    plt.ylabel(ylabel)
    plt.xticks(thread_counts)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Graph saved to {os.path.relpath(path, ROOT)}")


def plot_version(version, results, out_dir):
    """
    Execution time and speedup graphs of a single version, one per increasing
    subset of the available workloads (workloads 1..K).
    """
    for index in range(len(results)):

        # Execution time
        plt.figure()
        for sub in range(index + 1):
            thread_counts, means, margins = workload_stats(results[sub])
            plt.errorbar(
                thread_counts, means, yerr=margins,
                fmt='o-', capsize=5,
                color=color(sub),
                linewidth=1.5 if sub == index else 1.0,
                label=f'{results[sub]["num_to_search"]} values'
            )
        finish_plot(
            f"Version {printable(version)} - Execution Time - Workloads 1 to {index + 1}",
            "Execution Time (s)", thread_counts,
            os.path.join(out_dir, version, f"{version}_execution_time_up_to_workload_{index + 1}.png"),
        )

        # Speedup
        plt.figure()
        for sub in range(index + 1):
            thread_counts, means, margins = workload_stats(results[sub])
            speedup, lower, upper = speedup_stats(means, margins)
            is_last = (sub == index)
            plt.plot(thread_counts, speedup, 'o-', color=color(sub),
                     linewidth=2.5 if is_last else 1.0,
                     label=f'{results[sub]["num_to_search"]} values')
            plt.fill_between(thread_counts, lower, upper, color=color(sub), alpha=0.3 if is_last else 0.1)
        finish_plot(
            f"Version {printable(version)} - Speedup - Workloads 1 to {index + 1}",
            "Speedup", thread_counts,
            os.path.join(out_dir, version, f"{version}_speedup_up_to_workload_{index + 1}.png"),
        )


def plot_comparison(all_results, num_to_search, out_dir):
    """
    Execution time and speedup of all the versions that have a workload with
    num_to_search values to search.
    """
    selected = []
    for version, results in all_results.items():
        for workload in results:
            if int(workload["num_to_search"]) == num_to_search:
                selected.append((version, workload))

    if len(selected) < 2:
        print(f"Skipping comparison for {num_to_search}: fewer than two versions have this workload")
        return

    plt.figure()
    for i, (version, workload) in enumerate(selected):
        thread_counts, means, margins = workload_stats(workload)
        plt.errorbar(thread_counts, means, yerr=margins, fmt='o-', capsize=5, color=color(i),
                     label=f'Version {printable(version)} - {workload["num_nodes"]} nodes')
    finish_plot(
        f"Comparing versions - Execution Time - {num_to_search} values",
        "Execution Time (s)", thread_counts,
        os.path.join(out_dir, "compare", f"compare_execution_time_{num_to_search}.png"),
    )

    plt.figure()
    for i, (version, workload) in enumerate(selected):
        thread_counts, means, margins = workload_stats(workload)
        speedup, lower, upper = speedup_stats(means, margins)
        plt.plot(thread_counts, speedup, 'o-', color=color(i), linewidth=1.5,
                 label=f'Version {printable(version)} - {workload["num_nodes"]} nodes')
        plt.fill_between(thread_counts, lower, upper, color=color(i), alpha=0.15)
    finish_plot(
        f"Comparing versions - Speedup - {num_to_search} values",
        "Speedup", thread_counts,
        os.path.join(out_dir, "compare", f"compare_speedup_{num_to_search}.png"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results", "cpu"))
    parser.add_argument("--out", default=None, help="output directory (default: <results-dir>/graphs)")
    parser.add_argument("--versions", nargs="+", default=None,
                        help="versions to plot (default: all the vX.json files found)")
    parser.add_argument("--compare-workloads", type=int, nargs="+", default=[1_000_000, 10_000_000],
                        help="numbers of values to search of the comparison graphs (default: 1000000 10000000)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.results_dir, "graphs")

    versions = args.versions
    if versions is None:
        names = [os.path.basename(p)[:-len(".json")] for p in glob.glob(os.path.join(args.results_dir, "v*.json"))]
        versions = sorted(n for n in names if re.fullmatch(r"v\d+(_\d+)?", n))

    all_results = {}
    for version in versions:
        results = load_results(args.results_dir, version)
        if not results:
            print(f"No results found for {version}")
            continue
        all_results[version] = results
        plot_version(version, results, out_dir)

    for num_to_search in args.compare_workloads:
        plot_comparison(all_results, num_to_search, out_dir)


if __name__ == "__main__":
    main()
