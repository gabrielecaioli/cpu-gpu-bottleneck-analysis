/*
 * GPU Version 3 - Implicit array in constant + shared memory.
 *
 * Starts from Version 2.2: the tree array is copied to the GPU into
 * __constant__ memory and, at the beginning of the kernel, every block copies
 * it into its shared memory, so that the walk down the tree no longer hits
 * global memory. The tree must fit in the shared memory of a block:
 * 8190 nodes (~2^13) of 4 bytes = 2^15 bytes (32 KB).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "bintree.h"
#include "search_cuda.h"
#include "gpu_utils.cuh"


// Maximum number of elements of the tree array (32 KB of constant memory)
#define MAX_TREE_ARRAY_SIZE 8192

__constant__ uint32_t GPU_TREE_ROOT_CONST[MAX_TREE_ARRAY_SIZE];

/**
 * CUDA kernel that performs parallel searches on a tree stored in constant memory.
 * Every block first copies the tree into its shared memory, then every thread
 * searches a subset of the values and stores the result in the corresponding
 * position of the output array.
 *
 * @param search_values           Array of the values to search, in GPU memory.
 * @param results                 Output array in GPU memory: the result of each search.
 * @param TOTAL_NUM_TO_SEARCH     Total number of values to search.
 * @param NUM_THREADS_TOTAL       Total number of CUDA threads launched.
 * @param DIM_TREE                Size of the tree array, in nodes.
 * @param REPETITIONS_OF_WORKLOAD How many times the whole input is searched.
 */
__global__ void search_kernel(
    const uint32_t* search_values,
    uint8_t* results,
    intmax_t TOTAL_NUM_TO_SEARCH,
    int NUM_THREADS_TOTAL,
    int DIM_TREE,
    intmax_t REPETITIONS_OF_WORKLOAD
) {
    // Buffer in shared memory
    extern __shared__ uint32_t shared_tree[];

    int tid = threadIdx.x;

    // Every thread copies a part of the tree into shared memory
    for (int i = tid; i < DIM_TREE; i += blockDim.x) {
        shared_tree[i] = GPU_TREE_ROOT_CONST[i];
    }

    // Synchronize the threads of the block before starting the search
    __syncthreads();

    // Initial index
    intmax_t idx = blockIdx.x * blockDim.x + tid;
    intmax_t idx_tot = idx;

    // Loop until all the values have been searched (possibly repeated)
    while (idx_tot < TOTAL_NUM_TO_SEARCH * REPETITIONS_OF_WORKLOAD) {

        // Map the index back onto the actual array
        idx = idx_tot % TOTAL_NUM_TO_SEARCH;

        // Value to search
        uint32_t target = search_values[idx];

        int current = 0;
        uint8_t found = 0;

        while (current < DIM_TREE) {

            // Found
            if (shared_tree[current] == target) {
                found = 1;
                break;
            }

            // Otherwise move to the correct child
            current = ((target < shared_tree[current]) ? (current * 2 + 1) : (current * 2 + 2));
        }

        // Store the result
        results[idx] = found;

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
    if (dim_array_GPU_TREE_ROOT > MAX_TREE_ARRAY_SIZE) {
        fprintf(stderr, "Error: the tree array (%d elements) does not fit in constant memory (max %d elements).\n",
                dim_array_GPU_TREE_ROOT, MAX_TREE_ARRAY_SIZE);
        return;
    }

    // Tree on the CPU, as an array (zero-initialized)
    uint32_t* CPU_TREE_ROOT = (uint32_t*)malloc(dim_array_GPU_TREE_ROOT * sizeof(uint32_t));
    memset(CPU_TREE_ROOT, 0, dim_array_GPU_TREE_ROOT * sizeof(uint32_t));

    // Save the tree as an array
    cpu_tree_to_cpu_tree_array(tree_root, CPU_TREE_ROOT, 0);


    // Step 3: Tree array to the GPU, in CONSTANT MEMORY
double t2 = get_time_ms();
    cudaStatus = cudaMemcpyToSymbol(GPU_TREE_ROOT_CONST, CPU_TREE_ROOT, dim_array_GPU_TREE_ROOT * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy of CPU_TREE_ROOT to GPU_TREE_ROOT_CONST failed! Error: %s\n", cudaGetErrorString(cudaStatus));
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

    // Step 6: Launch the kernel, with the whole tree as dynamic shared memory
double t5 = get_time_ms();
    size_t shared_mem_size = limit_shared_mem_parameter(dim_array_GPU_TREE_ROOT * sizeof(uint32_t));
    search_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, shared_mem_size>>>(
        d_values_to_search,
        d_results,
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

        printf("[GPU v3] Timing breakdown (ms):\n");
        printf(" (info) > Workload:                %08jd\n", NUM_TO_SEARCH * REPETITIONS_OF_WORKLOAD);
        printf(" (info) > Number of nodes:         %08d\n", NUM_NODES);
        printf(" (info) > Threads per block:       %08d\n", NUM_THREADS_PER_BLOCK);
        printf(" (info) > Number of blocks:        %08d\n", NUM_BLOCKS);
        printf(" (info) > Number of threads:       %08d\n", NUM_THREADS_PER_BLOCK * NUM_BLOCKS);
        printf(" (info) > Tree size:               %08d\n", dim_array_GPU_TREE_ROOT);
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
    free(CPU_TREE_ROOT);
}
