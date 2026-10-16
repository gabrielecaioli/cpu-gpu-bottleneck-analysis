#ifndef SEARCH_CUDA_H
#define SEARCH_CUDA_H

#include "bintree.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Searches NUM_TO_SEARCH values in the tree on the GPU.
     * Every GPU version (src/gpu/v*.cu) provides its own implementation.
     *
     * @param tree_root               Root of the tree, built on the CPU
     * @param values_to_search        Values to search
     * @param results                 Output: results[i] = 1 if values_to_search[i] is in the tree, 0 otherwise
     * @param NUM_TO_SEARCH           Number of values to search
     * @param NUM_NODES               Number of values inserted in the tree
     * @param VERBOSE_INFO            Print debug information
     * @param VERBOSE_TIMES           Print the timing breakdown
     * @param NUM_THREADS_PER_BLOCK   CUDA threads per block
     * @param NUM_BLOCKS              CUDA blocks
     * @param REPETITIONS_OF_WORKLOAD How many times the whole input is searched
     *                                (to emulate workloads larger than the dataset)
     */
    void SEARCH_GPU(
        Node* tree_root,
        uint32_t* values_to_search,
        uint8_t* results,
        intmax_t NUM_TO_SEARCH,
        int NUM_NODES,
        bool VERBOSE_INFO,
        bool VERBOSE_TIMES,
        int NUM_THREADS_PER_BLOCK,
        int NUM_BLOCKS,
        intmax_t REPETITIONS_OF_WORKLOAD
    );

#ifdef __cplusplus
}
#endif

#endif // SEARCH_CUDA_H
