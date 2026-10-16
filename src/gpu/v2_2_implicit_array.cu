/*
 * GPU Version 2.2 - Tree as an implicit array (binary heap layout).
 *
 * The tree is first rebalanced on the CPU, then stored as a plain array of
 * values in breadth-first order: the children of the node at index i are at
 * 2*i + 1 (left) and 2*i + 2 (right). No pointers or indices are stored, so
 * every node takes 4 bytes instead of 24 (about -83% of memory).
 *
 * The array has enough room to reach the last level of the tree: with
 * MAX_LIV levels (so at most 1, 2, 4, 8, ... nodes per level) it needs at least
 *     1 + 2 + 4 + ... + 2^(MAX_LIV-1) = 2^MAX_LIV - 1 elements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "bintree.h"
#include "search_cuda.h"
#include "gpu_utils.cuh"


/**
 * Binary search on a tree stored as an implicit array in GPU memory.
 *
 * @param gpu_tree The tree array on the GPU
 * @param gpu_array_dim Size of the array
 * @param target The value to search
 * @return 1 if the value was found, 0 otherwise
 */
__device__ uint8_t search_tree(
    const uint32_t* gpu_tree,
    int gpu_array_dim,
    uint32_t target
) {
    int current = 0;

    // Loop while inside the tree, i.e. inside the array
    while (current < gpu_array_dim) {

        // Found
        if (gpu_tree[current] == target)
            return 1;

        // Otherwise move to the correct child
        current = ((target < gpu_tree[current]) ? (current * 2 + 1) : (current * 2 + 2));
    }
    return 0;
}


/**
 * CUDA kernel that performs parallel searches on a tree stored in GPU memory.
 * Every thread searches a subset of the values and stores the result in the
 * corresponding position of the output array.
 *
 * @param search_values           Array of the values to search, in GPU memory.
 * @param gpu_tree_array          Array representing the tree, in GPU memory.
 * @param results                 Output array in GPU memory: the result of each search.
 * @param root_index              Index of the root node in {@code gpu_tree_array}.
 * @param TOTAL_NUM_TO_SEARCH     Total number of values to search.
 * @param NUM_THREADS_TOTAL       Total number of CUDA threads launched.
 * @param DIM_TREE                Size of gpu_tree_array, in nodes.
 * @param REPETITIONS_OF_WORKLOAD How many times the whole input is searched.
 */
__global__ void search_kernel(
    const uint32_t* search_values,
    uint32_t* gpu_tree_array,
    uint8_t* results,
    int root_index,
    intmax_t TOTAL_NUM_TO_SEARCH,
    int NUM_THREADS_TOTAL,
    int DIM_TREE,
    intmax_t REPETITIONS_OF_WORKLOAD
) {
    // Initial index
    intmax_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    intmax_t idx_tot = idx;

    // Loop until all the values have been searched
    while (idx_tot < TOTAL_NUM_TO_SEARCH * REPETITIONS_OF_WORKLOAD) {
        idx = idx_tot % TOTAL_NUM_TO_SEARCH;

        // Value to search
        uint32_t value = search_values[idx];

        // Perform the search
        results[idx] = search_tree(gpu_tree_array, DIM_TREE, value);

        // Move to the next value
        idx_tot += NUM_THREADS_TOTAL;
    }
}


void SEARCH_GPU(
    Node* tree_root,
    uint32_t* values_to_search,
    uint8_t* return_values,
    intmax_t NUM_TO_SEARCH,
    int NUM_NODES,
    bool VERBOSE_INFO,
    bool VERBOSE_TIMES,
    int NUM_THREADS_PER_BLOCK,
    int NUM_BLOCKS,
    intmax_t REPETITIONS_OF_WORKLOAD
) {
    cudaError_t cudaStatus;

    // Check that a GPU is available
    if (!check_cuda_device()) return;

    // Step 1: Rebalance the tree (OVERWRITES THE ORIGINAL TREE)
double total_start = get_time_ms();
    tree_root = rebalance_tree(tree_root, NUM_NODES);


    // Step 2: Tree to array
double t1 = get_time_ms();

    // Number of levels of the tree
    int MAX_LIV = get_max_level(tree_root);

    // Size of the array needed to store the whole tree
    int dim_array_GPU_TREE_ROOT = (1 << MAX_LIV);

    // Tree on the CPU, as an array (zero-initialized)
    uint32_t* CPU_TREE_ROOT = (uint32_t*)malloc(dim_array_GPU_TREE_ROOT * sizeof(uint32_t));
    memset(CPU_TREE_ROOT, 0, dim_array_GPU_TREE_ROOT * sizeof(uint32_t));

    // Save the tree as an array
    cpu_tree_to_cpu_tree_array(tree_root, CPU_TREE_ROOT, 0);


    // Step 3: Tree array to the GPU
double t2 = get_time_ms();
    uint32_t* GPU_TREE_ROOT = NULL;
    cudaStatus = cudaMalloc((void**)&GPU_TREE_ROOT, dim_array_GPU_TREE_ROOT * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc on dim_array_GPU_TREE_ROOT failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    cudaStatus = cudaMemcpy(GPU_TREE_ROOT, CPU_TREE_ROOT, dim_array_GPU_TREE_ROOT * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy of CPU_TREE_ROOT to GPU_TREE_ROOT failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }


    // Step 4: Allocate GPU memory for the input/output arrays
double t3 = get_time_ms();
    uint32_t* d_values_to_search = NULL;
    uint8_t* d_results = NULL;
    cudaStatus = cudaMalloc((void**)&d_values_to_search, NUM_TO_SEARCH * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc on d_values_to_search failed!");
        return;
    }
    cudaStatus = cudaMalloc((void**)&d_results, NUM_TO_SEARCH * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc on d_results failed!");
        return;
    }

    // Step 5: Copy the input to the GPU
double t4 = get_time_ms();
    cudaStatus = cudaMemcpy(d_values_to_search, values_to_search, NUM_TO_SEARCH * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy of d_values_to_search failed!");
        return;
    }

    // Step 6: Launch the kernel
double t5 = get_time_ms();
    search_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(
        d_values_to_search,
        GPU_TREE_ROOT,
        d_results,
        0, // root_index
        NUM_TO_SEARCH,
        NUM_THREADS_PER_BLOCK * NUM_BLOCKS,
        dim_array_GPU_TREE_ROOT,
        REPETITIONS_OF_WORKLOAD
    );
    if (!check_kernel_error(NUM_BLOCKS, NUM_THREADS_PER_BLOCK)) return;

    // Step 7: Copy the results from the GPU to the CPU
double t6 = get_time_ms();
    cudaStatus = cudaMemcpy(return_values, d_results, NUM_TO_SEARCH * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy of d_results failed!");
        return;
    }

    // Step 8: Print the results
    if (VERBOSE_TIMES) {
        double total_end = get_time_ms();

        printf("[GPU v2.2] Timing breakdown (ms):\n");
        printf(" (info) > Workload:                %08jd\n", NUM_TO_SEARCH * REPETITIONS_OF_WORKLOAD);
        printf(" (info) > Number of nodes:         %08d\n", NUM_NODES);
        printf(" (info) > Threads per block:       %08d\n", NUM_THREADS_PER_BLOCK);
        printf(" (info) > Number of blocks:        %08d\n", NUM_BLOCKS);
        printf(" (info) > Number of threads:       %08d\n", NUM_THREADS_PER_BLOCK * NUM_BLOCKS);
        printf(" (t1)   - Rebalancing:             %07.3f ms\n", t1 - total_start);
        printf(" (t2)   - Tree to array:           %07.3f ms\n", t2 - t1);
        printf(" (t3)   - Tree to GPU:             %07.3f ms\n", t3 - t2);
        printf(" (t4)   - cudaMalloc of arrays:    %07.3f ms\n", t4 - t3);
        printf(" (t5)   - Input array copy:        %07.3f ms\n", t5 - t4);
        printf(" (t6)   - SEARCH:                  %07.3f ms\n", t6 - t5);
        printf(" (t7)   - Results copy:            %07.3f ms\n", total_end - t6);
        printf(" (tALL) - TOTAL:                   %07.3f ms\n", total_end - total_start);
    }

    // Step 9: Free the memory
    cudaFree(d_results);
    cudaFree(d_values_to_search);
    cudaFree(GPU_TREE_ROOT);
    free(CPU_TREE_ROOT);
}
