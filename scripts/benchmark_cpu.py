#!/usr/bin/env python3
"""
Benchmarks a CPU version of the parallel tree search.

For every workload, every thread count (1..MAX_THREADS) and every seed, the
executable is run once and its wall-clock time is measured. The results are
saved as JSON in the same format as the files in results/cpu/.

Versions:
    v1    thread dispatcher           (bin/cpu_v1, data generated at runtime from the seed)
    v2    range split                 (bin/cpu_v2_NN, one executable per dataset seed)
    v2_1  range split, thread count rounded down to a power of two
    v3    equal split                 (bin/cpu_v3_NN, one executable per dataset seed)

The default workloads are the ones used for the published results.

Examples:
    python3 scripts/benchmark_cpu.py v3
    python3 scripts/benchmark_cpu.py v3 --seeds 5 --max-threads 8 --nodes 10000 --queries 100000
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_WORKLOADS = {
    "v1":   {"nodes": [10_000] * 3, "queries": [10_000, 100_000, 1_000_000]},
    "v2":   {"nodes": [10_000] * 4, "queries": [10_000, 100_000, 1_000_000, 10_000_000]},
    "v2_1": {"nodes": [10_000] * 4, "queries": [10_000, 100_000, 1_000_000, 10_000_000]},
    "v3":   {"nodes": [10_000, 36_840, 135_720, 499_999], "queries": [10_000, 100_000, 1_000_000, 10_000_000]},
}

# Executable used by each version
EXECUTABLE = {"v1": "cpu_v1", "v2": "cpu_v2", "v2_1": "cpu_v2", "v3": "cpu_v3"}


def build(version, seeds, dataset_nodes, dataset_queries):
    """Builds the executables needed by the version (one per seed if it uses the pre-generated dataset)."""
    exe = EXECUTABLE[version]
    if version == "v1":
        subprocess.run(["make", f"bin/{exe}"], cwd=ROOT, check=True)
        return

    for seed in seeds:
        print(f"Building {exe} for dataset seed {seed}")
        subprocess.run(
            ["make", f"bin/{exe}_{seed:02d}", f"SEED={seed}", f"BIN_SUFFIX=_{seed:02d}",
             f"NODES={dataset_nodes}", f"QUERIES={dataset_queries}"],
            cwd=ROOT, check=True,
        )


def command(version, seed, num_threads, num_nodes, num_queries, min_value, max_value):
    """Command line of a single run."""
    exe = EXECUTABLE[version]
    if version == "v1":
        return [os.path.join(ROOT, "bin", exe), str(num_threads), str(num_nodes), str(num_queries),
                str(seed), str(min_value), str(max_value), "0", "0"]

    if version == "v2_1":
        # Main change from v2: highest power of two lower than or equal to num_threads
        num_threads = 2 ** (num_threads.bit_length() - 1)

    return [os.path.join(ROOT, "bin", f"{exe}_{seed:02d}"), str(num_threads), str(num_nodes), str(num_queries), "0", "0"]


def print_progress_bar(completed, total, prefix=""):
    bar = "=" * completed + " " * (total - completed)
    sys.stdout.write(f"\r{prefix}[{bar}]")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", choices=sorted(DEFAULT_WORKLOADS))
    parser.add_argument("--seeds", type=int, default=30, help="number of seeds / runs per configuration (default: 30)")
    parser.add_argument("--max-threads", type=int, default=24, help="threads go from 1 to MAX_THREADS (default: 24)")
    parser.add_argument("--nodes", type=int, nargs="+", help="number of nodes of each workload")
    parser.add_argument("--queries", type=int, nargs="+", help="number of values to search of each workload")
    parser.add_argument("--output", help="output JSON file (default: results/cpu/<version>.json)")
    parser.add_argument("--no-build", action="store_true", help="do not (re)build the executables")
    args = parser.parse_args()

    nodes = args.nodes or DEFAULT_WORKLOADS[args.version]["nodes"]
    queries = args.queries or DEFAULT_WORKLOADS[args.version]["queries"]
    if len(nodes) == 1 and len(queries) > 1:
        nodes = nodes * len(queries)
    if len(nodes) != len(queries):
        parser.error("--nodes and --queries must have the same length (or --nodes a single value)")

    seeds = list(range(1, args.seeds + 1))
    thread_counts = list(range(1, args.max_threads + 1))
    output = args.output or os.path.join(ROOT, "results", "cpu", f"{args.version}.json")

    if not args.no_build:
        # The dataset must be large enough for the largest workload
        build(args.version, seeds, max(nodes), max(queries))

    all_results = []
    for w, (num_nodes, num_queries) in enumerate(zip(nodes, queries)):
        # v1 generates its values at runtime in [num_nodes, 10 * num_nodes - 1]
        min_value, max_value = num_nodes, num_nodes * 10 - 1

        workload_results = {
            "workload_index": w,
            "num_nodes": num_nodes,
            "num_to_search": num_queries,
            "thread_count_min": 1,
            "thread_count_max": args.max_threads,
            "seeds": args.seeds,
            "threads": [],
        }
        if args.version == "v1":
            workload_results["min_value"] = min_value
            workload_results["max_value"] = max_value

        print(f"\nWORKLOAD {w}: {num_nodes} nodes, {num_queries} values to search")

        for num_threads in thread_counts:
            thread_results = {"num_threads": num_threads, "runs": []}

            for i, seed in enumerate(seeds):
                cmd = command(args.version, seed, num_threads, num_nodes, num_queries, min_value, max_value)

                start_time = time.perf_counter()
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                elapsed_time = time.perf_counter() - start_time

                thread_results["runs"].append({"seed": seed, "execution_time": elapsed_time})
                print_progress_bar(i + 1, len(seeds), prefix=f"Threads: {num_threads:03d} ")

            workload_results["threads"].append(thread_results)
            print()

            # Short pause between two thread configurations
            time.sleep(0.5)

        all_results.append(workload_results)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=6)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
