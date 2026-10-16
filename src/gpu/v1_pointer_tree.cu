/*
 * GPU Version 1 - Pointer-based tree.
 *
 * Same algorithm as the final CPU version: the values to search are logically
 * split among the threads and every thread starts from the root. The tree is
 * copied to the GPU node by node, keeping the original pointer structure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bintree.h"
#include "search_cuda.h"
#include "gpu_utils.cuh"


/**
 * @brief Recursive function that copies a binary tree from the CPU to the GPU.
 *
 * @param cpu_node Pointer to the current node on the host (CPU).
 * @param gpu_node_out Pointer to the pointer that will contain the address of the node allocated on the GPU.
 * @return cudaError_t CUDA error code (cudaSuccess if everything went fine).
 */
cudaError_t copyTreeToGPU(const Node* cpu_node, Node** gpu_node_out) {
    if (cpu_node == NULL) {
        *gpu_node_out = NULL;
        return cudaSuccess;
    }

    Node* gpu_node;
    cudaError_t err;

    // Allocate memory for the node on the GPU
    err = cudaMalloc((void**)&gpu_node, sizeof(Node));
    if (err != cudaSuccess) return err;

    // Copy the value of the current node
    err = cudaMemcpy(&(gpu_node->value), &(cpu_node->value), sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    // Temporary pointers for the children
    Node* gpu_left = NULL;
    Node* gpu_right = NULL;

    // Recursive copy of the left child
    err = copyTreeToGPU(cpu_node->left, &gpu_left);
    if (err != cudaSuccess) return err;

    // Recursive copy of the right child
    err = copyTreeToGPU(cpu_node->right, &gpu_right);
    if (err != cudaSuccess) return err;

    // Copy the children pointers into the GPU node
    err = cudaMemcpy(&(gpu_node->left), &gpu_left, sizeof(Node*), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    err = cudaMemcpy(&(gpu_node->right), &gpu_right, sizeof(Node*), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    *gpu_node_out = gpu_node;
    return cudaSuccess;
}


/**
 * Iterative search of a value in the pointer-based tree.
 * @return 1 if found, 0 otherwise
 */
__device__ uint8_t search_tree(Node* root, uint32_t target) {
    while (root != NULL) {
        if (target == root->value) {
            return 1; // found
        }
        else if (target < root->value) {
            root = root->left;
        }
        else {
            root = root->right;
        }
    }
    return 0; // not found
}

/**
 * @brief CUDA kernel that performs parallel searches on a binary tree on the GPU.
 *
 * @param values_to_search        Array of the values to search.
 * @param tree_root               Pointer to the root of the tree (on the GPU).
 * @param results                 Output array (1 if found, 0 otherwise).
 * @param TOTAL_NUM_TO_SEARCH     Number of values to search.
 * @param NUM_THREADS_TOTAL       Total number of CUDA threads launched.
 * @param REPETITIONS_OF_WORKLOAD How many times the whole input is searched.
 */
__global__ void search_kernel(
    const uint32_t* values_to_search,
    Node* tree_root,
    uint8_t* results,
    intmax_t TOTAL_NUM_TO_SEARCH,
    int NUM_THREADS_TOTAL,
    intmax_t REPETITIONS_OF_WORKLOAD
) {
    intmax_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    intmax_t idx_tot = idx;

    // Loop until all the values have been searched
    while (idx_tot < TOTAL_NUM_TO_SEARCH * REPETITIONS_OF_WORKLOAD) {
        idx = idx_tot % TOTAL_NUM_TO_SEARCH;
        uint32_t value = values_to_search[idx];
        results[idx] = search_tree(tree_root, value);
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
    Node* d_tree_root = NULL;
    cudaError_t err = copyTreeToGPU(tree_root, &d_tree_root);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error while copying the tree to the GPU: %s\n", cudaGetErrorString(err));
        return;
    }
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
        fprintf(stderr, "cudaMemcpy on d_values_to_search failed!");
        return;
    }
    double t6 = get_time_ms();

    // Step 4: Launch the kernel
    double t7 = get_time_ms();
    search_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(
        d_values_to_search,
        d_tree_root,
        d_results,
        NUM_TO_SEARCH,
        NUM_BLOCKS * NUM_THREADS_PER_BLOCK,
        REPETITIONS_OF_WORKLOAD
    );
    if (!check_kernel_error(NUM_BLOCKS, NUM_THREADS_PER_BLOCK)) return;
    double t8 = get_time_ms();

    // Step 5: Copy the results from the GPU to the CPU
    double t9 = get_time_ms();
    cudaStatus = cudaMemcpy(return_values, d_results, NUM_TO_SEARCH * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy on return_values failed!");
        return;
    }
    double t10 = get_time_ms();

    // Step 6: Free the GPU memory
    double t11 = get_time_ms();
    cudaFree(d_values_to_search);
    cudaFree(d_results);
    // Note: the GPU tree (d_tree_root) is not freed, as it would require a recursive deallocation
    double t12 = get_time_ms();

    if (VERBOSE_TIMES) {
        double total_end = get_time_ms();

        printf("[GPU v1] Timing breakdown (ms):\n");
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
        printf(" (t6)   - Deallocation:            %07.3f ms\n", t12 - t11);
        printf(" (tALL) - TOTAL:                   %07.3f ms\n", total_end - total_start);
    }
}
