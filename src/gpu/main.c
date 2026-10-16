#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include "bintree.h"
#include "dataset.h"
#include "search_cuda.h"

/*
///////////////////////////////////////////
ARGUMENTS:
      <NUM_THREADS_PER_BLOCK> <NUM_BLOCKS> <NUM_NODES> <NUM_TO_SEARCH> <TO_EXTRACT>
      <VERBOSE (0 or 1)> <VERBOSE_TIMES (0 or 1)> <REPETITIONS_OF_WORKLOAD>
Builds the tree on the CPU from the pre-generated dataset, then searches the
values on the GPU with the SEARCH_GPU implementation the executable was
linked with.
///////////////////////////////////////////
*/

int main(int argc, char* argv[]) {

    // Default arguments, in the order they must be given on the command line
    int NUM_THREADS_PER_BLOCK = 32;
    int NUM_BLOCKS;
    int NUM_NODES = 8190;
    intmax_t NUM_TO_SEARCH = 10000;
    int TO_EXTRACT = 0;
    bool VERBOSE = false;
    bool VERBOSE_TIMES = true;
    NUM_BLOCKS = NUM_TO_SEARCH / NUM_THREADS_PER_BLOCK;
    intmax_t REPETITIONS_OF_WORKLOAD = 1;

    // Read the arguments from the command line
    if (argc > 1)
        NUM_THREADS_PER_BLOCK = atoi(argv[1]);
    if (argc > 2)
        NUM_BLOCKS = atoi(argv[2]);
    if (argc > 3)
        NUM_NODES = atoi(argv[3]);
    if (argc > 4)
        NUM_TO_SEARCH = atoi(argv[4]);
    if (argc > 5)
        TO_EXTRACT = atoi(argv[5]);
    if (argc > 6)
        VERBOSE = atoi(argv[6]);
    if (argc > 7)
        VERBOSE_TIMES = atoi(argv[7]);
    if (argc > 8)
        REPETITIONS_OF_WORKLOAD = atoi(argv[8]);

    // Check that the parameters are valid
    if (NUM_BLOCKS < 1 || NUM_THREADS_PER_BLOCK < 1 || NUM_NODES < 1 || NUM_TO_SEARCH < 1 || REPETITIONS_OF_WORKLOAD < 1) {
        printf("Error: Invalid arguments.\n");
        printf("Usage: %s <NUM_THREADS_PER_BLOCK> <NUM_BLOCKS> <NUM_NODES> <NUM_TO_SEARCH> <TO_EXTRACT> <VERBOSE (0 or 1)> <VERBOSE_TIMES (0 or 1)> <REPETITIONS_OF_WORKLOAD>\n", argv[0]);
        return EXIT_FAILURE;
    }
    // Check that the two sizes do not exceed the available dataset
    if (NUM_NODES > dataset_nodes_count) {
        printf("Error: NUM_NODES (%d) is greater than the maximum allowed (%d).\n", NUM_NODES, dataset_nodes_count);
        return EXIT_FAILURE;
    }
    if (NUM_TO_SEARCH > (intmax_t)dataset_queries_count) {
        printf("Error: NUM_TO_SEARCH (%jd) is greater than the maximum allowed (%d).\n", NUM_TO_SEARCH, dataset_queries_count);
        return EXIT_FAILURE;
    }


    if (VERBOSE) printf("\n REGION 1: ARRAY & BINTREE CREATION =================\n");
    BinTree tree;
    bintree_init(&tree);

    uint8_t* results = (uint8_t*)calloc(NUM_TO_SEARCH, sizeof(uint8_t));
    if (results == NULL) {
        perror("Failed to allocate memory for the results");
        return EXIT_FAILURE;
    }


    if (VERBOSE) printf("\n REGION 2: BINTREE INSERTIONS =======================\n");
    for (int i = 0; i < NUM_NODES; ++i) {
        if (VERBOSE) printf("%03d:%03u ", i, dataset_nodes[i]);
        insert(&tree, dataset_nodes[i]);
    }

    if (VERBOSE) printf("\nFinal tree structure, only top 4 layers\n");
    if (VERBOSE) print_tree_max_levels(tree.root, 0, 4);


    if (VERBOSE) printf("\n REGION 3: SEARCH (a lot of) VALUES =================\n");
    if (VERBOSE) {
        printf("Search values: ");
        for (int i = 0; i < NUM_TO_SEARCH; ++i) {
            printf("%03d:%03u ", i, dataset_queries[i]);
        }
        printf("\n");
    }

    // Note: the versions that rebalance the tree (v2.2 and later) relink the
    // nodes in place, so afterwards tree.root may point to an inner node and
    // the extraction/deallocation below only reach part of the tree. The rest
    // of the memory is released when the process exits.
    SEARCH_GPU(
        tree.root,
        dataset_queries,
        results,
        NUM_TO_SEARCH,
        NUM_NODES,
        VERBOSE,
        VERBOSE_TIMES,
        NUM_THREADS_PER_BLOCK,
        NUM_BLOCKS,
        REPETITIONS_OF_WORKLOAD
    );


    // Check the search results
    if (VERBOSE) {
        printf("Search results after search:\n");
        for (int i = 0; i < NUM_TO_SEARCH; i++) {
            printf("Value %04u: ", dataset_queries[i]);
            if (results[i] == 0)
                printf("NOT FOUND\n");
            else
                printf("FOUND\n");
        }
    }


    if (VERBOSE) printf("\n REGION 4: REMOVE SOME VALUES =======================\n");
    int deleted_nodes = 0;
    while ((deleted_nodes < TO_EXTRACT) && (tree.root != NULL)) {
        int i = rand() % NUM_NODES;
        if (extract(&tree, dataset_nodes[i]))
            deleted_nodes++;
    }


    if (VERBOSE) printf("\n REGION 5: DEALLOCATE ENTIRE TREE ===================\n");
    bintree_free(&tree);
    free(results);

    return 0;
}
