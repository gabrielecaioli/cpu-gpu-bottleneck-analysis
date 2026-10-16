#!/usr/bin/env python3
"""
Benchmarks the GPU versions of the parallel tree search (needs an NVIDIA GPU
and nvcc; it is meant to be run on Google Colab, see notebooks/gpu_colab.ipynb).

For every version, workload, total number of threads and seed, the executable
bin/gpu_<version>_<seed> is run once; the timing breakdown it prints is parsed
and saved in results/gpu/res_<version>_w<workload>_TpB<threads per block>.json.

If a workload is larger than the number of values available in the dataset,
the whole dataset is searched multiple times (REPETITIONS_OF_WORKLOAD).

Examples:
    python3 scripts/benchmark_gpu.py --versions v1 v2_1 v2_2 --queries 10000000
    python3 scripts/benchmark_gpu.py --versions v3 --threads-per-block 1024
    python3 scripts/benchmark_gpu.py --versions v3 --fixed-blocks 40   # 40 blocks = number of SMs of a Tesla T4
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VERSIONS = ["v1", "v2_1", "v2_2", "v3"]
DEFAULT_THREADS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 40960, 65536, 131072]
DEFAULT_THREADS_MUL_40 = [40 * 2**i for i in range(13)]  # 40 ... 163840


def build(versions, seeds, dataset_nodes, dataset_queries):
    """Builds one executable per version and per dataset seed."""
    for seed in seeds:
        targets = [f"bin/gpu_{v}_{seed:02d}" for v in versions]
        print(f"Building {', '.join(targets)}")
        subprocess.run(
            ["make", *targets, f"SEED={seed}", f"BIN_SUFFIX=_{seed:02d}",
             f"NODES={dataset_nodes}", f"QUERIES={dataset_queries}"],
            cwd=ROOT, check=True,
        )


def parse_gpu_output(output, version, threads_per_block, num_blocks, num_nodes, num_to_search):
    """
    Parses the timing breakdown printed by an executable.

    :return: dictionary with the results of the run
    """
    result = {
        "version": version,
        "workload_num_to_search": num_to_search,
        "num_nodes": num_nodes,
        "Threads_per_block": threads_per_block,
        "Num_blocks": num_blocks,
        "Total_num_threads": threads_per_block * num_blocks,
        "total_time": None,
        "timings": [],
    }

    match_total = re.search(r'\(tALL\).*TOTAL:\s+([\d.]+)', output)
    if match_total:
        result["total_time"] = float(match_total.group(1))

    for idx, desc, value in re.findall(r'\(t(\d+)\)\s+-\s+(.*?):\s+([\d.]+)', output):
        result["timings"].append({"index": int(idx), "description": desc.strip(), "time": float(value)})

    return result


def execute(executable, version, total_threads, threads_per_block, fixed_blocks, num_nodes, workload, dataset_queries):
    """Runs a single configuration and returns the parsed results."""
    if fixed_blocks:
        num_blocks = fixed_blocks
        tpb = total_threads // fixed_blocks
    elif total_threads <= threads_per_block:
        num_blocks = 1
        tpb = total_threads
    else:
        num_blocks = total_threads // threads_per_block
        tpb = threads_per_block

    # If the workload is larger than the dataset, search the whole dataset multiple times
    num_to_search, repetitions = workload, 1
    if workload > dataset_queries:
        repetitions = workload // dataset_queries
        num_to_search = dataset_queries

    cmd = [executable, str(tpb), str(num_blocks), str(num_nodes), str(num_to_search),
           "0",  # TO_EXTRACT
           "0",  # VERBOSE
           "1",  # VERBOSE_TIMES
           str(repetitions)]
    output = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return parse_gpu_output(output, version, tpb, num_blocks, num_nodes, num_to_search)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", nargs="+", default=VERSIONS, choices=VERSIONS)
    parser.add_argument("--seeds", type=int, default=30, help="number of dataset seeds / runs per configuration (default: 30)")
    parser.add_argument("--nodes", type=int, default=8190, help="number of nodes of the tree (default: 8190, fits in shared memory)")
    parser.add_argument("--queries", type=int, nargs="+", default=[10_000, 100_000, 1_000_000, 10_000_000],
                        help="workloads: numbers of values to search")
    parser.add_argument("--threads", type=int, nargs="+", default=None, help="total numbers of threads to test")
    parser.add_argument("--threads-per-block", type=int, default=32, help="threads per block (default: 32)")
    parser.add_argument("--fixed-blocks", type=int, default=None,
                        help="use this fixed number of blocks instead of a fixed number of threads per block")
    parser.add_argument("--dataset-nodes", type=int, default=500_000, help="size of the generated dataset (default: 500000)")
    parser.add_argument("--dataset-queries", type=int, default=10_000_000, help="size of the generated dataset (default: 10000000)")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "results", "gpu"))
    parser.add_argument("--bin-dir", default=os.path.join(ROOT, "bin"))
    parser.add_argument("--no-build", action="store_true", help="do not (re)build the executables")
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    threads = args.threads or (DEFAULT_THREADS_MUL_40 if args.fixed_blocks else DEFAULT_THREADS)

    if not args.no_build:
        build(args.versions, seeds, args.dataset_nodes, args.dataset_queries)

    os.makedirs(args.output_dir, exist_ok=True)

    for version in args.versions:
        print(f"\nVersion {version}")
        for workload in args.queries:
            print(f"Workload: {args.nodes} nodes, {workload} values to search")
            workload_results = []

            for total_threads in threads:
                start = time.time()
                for i, seed in enumerate(seeds):
                    executable = os.path.join(args.bin_dir, f"gpu_{version}_{seed:02d}")
                    try:
                        workload_results.append(execute(
                            executable, version, total_threads, args.threads_per_block, args.fixed_blocks,
                            args.nodes, workload, args.dataset_queries,
                        ))
                    except subprocess.CalledProcessError as e:
                        print(f"\nError running {version}, workload {workload}, {total_threads} threads, seed {seed}: {e}")
                    sys.stdout.write(f"\r  Threads: {total_threads:08d} [{i + 1}/{len(seeds)}]")
                    sys.stdout.flush()
                elapsed = time.time() - start
                print(f"\r  Threads: {total_threads:08d}: mean time {elapsed / len(seeds):08.5f}s | total time {elapsed:08.5f}s")

            suffix = f"Block{args.fixed_blocks}" if args.fixed_blocks else f"TpB{args.threads_per_block}"
            output = os.path.join(args.output_dir, f"res_{version}_w{workload}_{suffix}.json")
            with open(output, "w") as f:
                json.dump(workload_results, f, indent=2)
            print(f"Results saved to {os.path.relpath(output, ROOT)}")


if __name__ == "__main__":
    main()
