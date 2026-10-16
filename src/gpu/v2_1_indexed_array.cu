/*
 * GPU Version 2.1 - Tree as an array of structs.
 *
 * The tree is stored in a single contiguous array: every element is a struct
 * holding the value and the indices (instead of the pointers) of the two
 * children. This removes the scattered allocations of Version 1 and reduces
 * the size of each node from 24 to 12 bytes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bintree.h"
#include "search_cuda.h"
#include "gpu_utils.cuh"


// Node of the tree stored as an array: the children are indices in the array (-1 = no child)
typedef struct GpuNode {
    uint32_t value;
    int left_child_idx;
    int right_child_idx;
} GpuNode;


/**
 * Copies the subtree rooted at cpu_node into gpu_array in pre-order.
 * @return the index of cpu_node in the array (-1 if cpu_node is NULL)
 */
int copy_tree_recursive_indexed(const Node* cpu_node, GpuNode* gpu_array, int* current_index) {
    if (!cpu_node) return -1;

    int this_index = *current_index;
    gpu_array[this_index].value = cpu_node->value;

    (*current_index)++;
    int left_index = copy_tree_recursive_indexed(cpu_node->left, gpu_array, current_index);
    int right_index = copy_tree_recursive_indexed(cpu_node->right, gpu_array, current_index);

    gpu_array[this_index].left_child_idx = left_index;
    gpu_array[this_index].right_child_idx = right_index;

    return this_index;
}

/**
 * Converts the tree into a GpuNode array and copies it to the GPU.
 * @return the device pointer of the array (the root is at index 0)
 */
GpuNode* copy_tree_to_gpu_indexed(const Node* cpu_root, size_t* out_total_nodes) {
    if (!cpu_root) return NULL;

    size_t total_nodes = count_nodes(cpu_root);
    *out_total_nodes = total_nodes;

    GpuNode* gpu_array_host = (GpuNode*)malloc(sizeof(GpuNode) * total_nodes);

    int index = 0;
    copy_tree_recursive_indexed(cpu_root, gpu_array_host, &index);

    GpuNode* gpu_array_device = NULL;
    cudaMalloc((void**)&gpu_array_device, sizeof(GpuNode) * total_nodes);
    cudaMemcpy(gpu_array_device, gpu_array_host, sizeof(GpuNode) * total_nodes, cudaMemcpyHostToDevice);

    free(gpu_array_host);
    return gpu_array_device;
}


/**
 * Iterative search of a value in the GpuNode array.
 * @return 1 if found, 0 otherwise
 */
__device__ uint8_t search_tree(const GpuNode* nodes, int root_idx, uint32_t target) {
    int current = root_idx;
    while (current != -1) {
        GpuNode node = nodes[current];
        if (target == node.value)
            return 1;
        current = (target < node.value) ? node.left_child_idx : node.right_child_idx;
    }
    return 0;
}


/**
 * @brief CUDA kernel that performs parallel searches on a binary tree on the GPU.
 *
 * @param search_values             Array of the values to search.
 * @param gpu_tree_root             Array of the tree nodes (on the GPU).
 * @param results                   Output array (1 if found, 0 otherwise).
 * @param TOTAL_NUM_TO_SEARCH       Number of values to search.
 * @param root_index                Index of the root node in gpu_tree_root.
 * @param NUM_THREADS_TOTAL         Total number of CUDA threads launched.
 * @param REPETITIONS_OF_WORKLOAD   How many times the whole input is searched.
 */
__global__ void search_kernel(
    const uint32_t* search_values,
    const GpuNode* gpu_tree_root,
    uint8_t* results,
    intmax_t TOTAL_NUM_TO_SEARCH,
    int root_index,
    int NUM_THREADS_TOTAL,
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
        results[idx] = search_tree(gpu_tree_root, root_index, value);

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
    double total_start = get_time_ms();

    // Check that a GPU is available
    if (!check_cuda_device()) return;

    // Step 1: Copy the tree to the GPU
    double t1 = get_time_ms();
    GpuNode* gpu_tree_root = NULL;
    size_t total_nodes;
    gpu_tree_root = copy_tree_to_gpu_indexed(tree_root, &total_nodes);
    double t2 = get_time_ms();

    // Step 2: Allocate GPU memory for the input/output arrays
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
    double t4 = get_time_ms();

    // Step 3: Copy the input to the GPU
    double t5 = get_time_ms();
    cudaStatus = cudaMemcpy(d_values_to_search, values_to_search, NUM_TO_SEARCH * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy of d_values_to_search failed!");
        return;
    }
    double t6 = get_time_ms();

    // Step 4: Launch the kernel
    double t7 = get_time_ms();
    search_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(
        d_values_to_search,
        gpu_tree_root,
        d_results,
        NUM_TO_SEARCH,
        0, // root_index
        NUM_BLOCKS * NUM_THREADS_PER_BLOCK,
        REPETITIONS_OF_WORKLOAD
    );
    if (!check_kernel_error(NUM_BLOCKS, NUM_THREADS_PER_BLOCK)) return;
    double t8 = get_time_ms();

    // Step 5: Copy the results from the GPU to the CPU
    double t9 = get_time_ms();
    cudaStatus = cudaMemcpy(return_values, d_results, NUM_TO_SEARCH * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy of d_results failed!");
        return;
    }
    double t10 = get_time_ms();

    // Step 6: Print the results
    if (VERBOSE_TIMES) {
        double total_end = get_time_ms();

        printf("[GPU v2.1] Timing breakdown (ms):\n");
        printf(" (info) > Workload:                %08jd\n", NUM_TO_SEARCH * REPETITIONS_OF_WORKLOAD);
        printf(" (info) > Number of nodes:         %08d\n", NUM_NODES);
        printf(" (info) > Threads per block:       %08d\n", NUM_THREADS_PER_BLOCK);
        printf(" (info) > Number of blocks:        %08d\n", NUM_BLOCKS);
        printf(" (info) > Number of threads:       %08d\n", NUM_THREADS_PER_BLOCK * NUM_BLOCKS);
        printf(" (t1)   - Tree to GPU:             %07.3f ms\n", t2 - t1);
        printf(" (t2)   - cudaMalloc of arrays:    %07.3f ms\n", t4 - t3);
        printf(" (t3)   - Input array copy:        %07.3f ms\n", t6 - t5);
        printf(" (t4)   - SEARCH:                  %07.3f ms\n", t8 - t7);
        printf(" (t5)   - Results copy:            %07.3f ms\n", t10 - t9);
        printf(" (tALL) - TOTAL:                   %07.3f ms\n", total_end - total_start);
    }

    // Step 7: Free the GPU memory
    cudaFree(d_results);
    cudaFree(d_values_to_search);
    cudaFree(gpu_tree_root);
}
