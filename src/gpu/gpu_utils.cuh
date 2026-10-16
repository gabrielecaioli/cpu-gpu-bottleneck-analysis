#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH

/*
 * Host-side helpers shared by the GPU versions: timing, error checking and
 * the CPU-side transformations of the tree (rebalancing, conversion to array).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "bintree.h"

// ---- Timing ---------------------------------------------------------------------

#ifdef _WIN32
#include <windows.h>
static inline double get_time_ms() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)(counter.QuadPart) * 1000.0 / freq.QuadPart;
}
#else
#include <time.h>
static inline double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

// ---- Error checking ---------------------------------------------------------------

/**
 * Checks that a CUDA device is available.
 * @return true if at least one device is available.
 */
static inline bool check_cuda_device() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        fprintf(stderr, "Error: no CUDA device available. Details: %s\n", cudaGetErrorString(status));
        return false;
    }
    return true;
}

/**
 * Waits for the kernel to finish and checks whether it failed.
 * @return true if the kernel completed successfully.
 */
static inline bool check_kernel_error(int num_blocks, int num_threads_per_block) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
        return true;

    if (err == cudaErrorInvalidConfiguration) {
        fprintf(stderr, "Kernel error: %s, configuration used: %d blocks of %d threads\n\n",
                cudaGetErrorString(err), num_blocks, num_threads_per_block);
    }
    else {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
    }
    return false;
}

// ---- Tree helpers (CPU side) ------------------------------------------------------

/**
 * Recursively counts the number of nodes in the tree.
 */
static inline size_t count_nodes(const Node* root) {
    if (root == NULL) return 0;
    return 1 + count_nodes(root->left) + count_nodes(root->right);
}

/**
 * Returns the number of levels of the tree.
 * @param root Root node of the tree.
 */
static inline int get_max_level(const Node* root) {
    if (root == NULL) return 0;
    int left_level = get_max_level(root->left) + 1;
    int right_level = get_max_level(root->right) + 1;
    return (left_level > right_level) ? left_level : right_level;
}

/**
 * Inserts the nodes of a binary tree into an array, following an in-order visit
 * (so the array ends up sorted by value).
 *
 * @param target_array Array where the nodes are inserted, already allocated with enough space
 * @param node_to_insert Root of the tree to insert
 * @param index_to_insert Index of the array where the next node is inserted
 */
static inline void insert_in_array_inOrder(Node** target_array, Node* node_to_insert, int* index_to_insert) {
    if (node_to_insert == NULL) return;

    // Recursive call on the left
    insert_in_array_inOrder(target_array, node_to_insert->left, index_to_insert);

    // Insert the current node and increment the index
    target_array[*index_to_insert] = node_to_insert;
    (*index_to_insert)++;

    // Recursive call on the right
    insert_in_array_inOrder(target_array, node_to_insert->right, index_to_insert);
}

/**
 * Builds a balanced tree from an array of nodes sorted by value.
 *
 * @param ordered_array Array of nodes sorted by value
 * @param start Initial index of the portion of {@code ordered_array} to consider
 * @param end Final index of the portion of {@code ordered_array} to consider
 * @returns Root node
 */
static inline Node* insert_from_ordered_array(Node** ordered_array, int start, int end) {
    if (start > end)
        return NULL;

    // The midpoint becomes the root of this subtree
    int midpoint = (start + end) / 2;
    Node* root = ordered_array[midpoint];

    // Recursive call on the two children
    root->left = insert_from_ordered_array(ordered_array, start, midpoint - 1);
    root->right = insert_from_ordered_array(ordered_array, midpoint + 1, end);

    return root;
}

/**
 * Rebalances a binary tree in place (the nodes are reused, only the links change).
 *
 * @param root Root of the tree
 * @param NUM_NODES Upper bound on the number of nodes of the tree
 * @returns The new root
 */
static inline Node* rebalance_tree(Node* root, const int NUM_NODES) {
    // Array of Node* large enough to contain all the nodes
    Node** tmp_array = (Node**)malloc(NUM_NODES * sizeof(Node*));

    // In-order visit: the array is sorted by value
    int index = 0;
    insert_in_array_inOrder(tmp_array, root, &index);

    // Build the balanced tree from the sorted array, overwriting the root
    root = insert_from_ordered_array(tmp_array, 0, index - 1);

    free(tmp_array);
    return root;
}

/**
 * Copies a {@code Node*} binary tree into a linear array using the binary heap layout:
 * every node is stored at index {@code i}, its left child at {@code 2*i + 1}
 * and its right child at {@code 2*i + 2}.
 *
 * @param cpu_root Root of the binary tree to copy.
 * @param cpu_array Destination array representing the tree in linear form.
 * @param index Current index where {@code cpu_root} is inserted; must initially be 0.
 */
static inline void cpu_tree_to_cpu_tree_array(const Node* cpu_root, uint32_t* cpu_array, int index) {
    if (!cpu_root) {
        return;
    }

    // Save the node
    cpu_array[index] = cpu_root->value;

    // Recursive calls on the left and right children
    cpu_tree_to_cpu_tree_array(cpu_root->left, cpu_array, 2 * index + 1);
    cpu_tree_to_cpu_tree_array(cpu_root->right, cpu_array, 2 * index + 2);
}

/**
 * Returns the actual size of the usable shared memory, comparing the desired
 * value with the maximum available per block.
 *
 * @param desired_size_in_bytes Desired size of the shared memory per block (in bytes).
 * @return Actual usable size (in bytes).
 */
static inline size_t limit_shared_mem_parameter(size_t desired_size_in_bytes) {
    int device;
    cudaGetDevice(&device);

    int max_shared_mem_per_block = 0;
    cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device);

    if (desired_size_in_bytes > (size_t)max_shared_mem_per_block) {
        fprintf(stderr,
            "Warning: requested shared memory (%zu bytes) is larger than the maximum per block (%d bytes). "
            "The maximum available will be used.\n",
            desired_size_in_bytes, max_shared_mem_per_block);
        return (size_t)max_shared_mem_per_block;
    }

    return desired_size_in_bytes;
}

#endif // GPU_UTILS_CUH
