#include <stdio.h>
#include <stdlib.h>

#include "bintree.h"
#include "dataset.h"
#include "search_v3.h"

/*
///////////////////////////////////////////
ARGUMENTS:
      <NUM_THREADS> <NUM_NODES> <NUM_TO_SEARCH> <TO_EXTRACT> <VERBOSE (0 or 1)>
Builds the tree from the pre-generated dataset, assigns the searches to the
threads, removes some nodes.
///////////////////////////////////////////
*/

int main(int argc, char* argv[]) {

      // Default arguments, in the order they must be given on the command line
      int NUM_THREADS   = 8;
      int NUM_NODES     = 100000;
      int NUM_TO_SEARCH = 1000000;
      int TO_EXTRACT    = 0;
      bool VERBOSE      = false;

      // Read the arguments from the command line
      if (argc > 1)
            NUM_THREADS = atoi(argv[1]);
      if (argc > 2)
            NUM_NODES = atoi(argv[2]);
      if (argc > 3)
            NUM_TO_SEARCH = atoi(argv[3]);
      if (argc > 4)
            TO_EXTRACT = atoi(argv[4]);
      if (argc > 5)
            VERBOSE = atoi(argv[5]);

      // Check that the parameters are valid
      if (NUM_THREADS < 1 || NUM_NODES < 1 || NUM_TO_SEARCH < 1) {
            printf("Error: Invalid arguments.\n");
            printf("Usage: %s <NUM_THREADS> <NUM_NODES> <NUM_TO_SEARCH> <TO_EXTRACT> <VERBOSE (0 or 1)>\n", argv[0]);
            return EXIT_FAILURE;
      }
      // Check that the two sizes do not exceed the available dataset
      if (NUM_NODES > dataset_nodes_count) {
            printf("Error: NUM_NODES (%d) is greater than the maximum allowed (%d).\n", NUM_NODES, dataset_nodes_count);
            return EXIT_FAILURE;
      }
      if (NUM_TO_SEARCH > dataset_queries_count) {
            printf("Error: NUM_TO_SEARCH (%d) is greater than the maximum allowed (%d).\n", NUM_TO_SEARCH, dataset_queries_count);
            return EXIT_FAILURE;
      }


      if (VERBOSE) printf("\n REGION 1: ARRAY & BINTREE CREATION =================\n");
      BinTree tree;
      bintree_init(&tree);

      Node** results = (Node**)malloc(NUM_TO_SEARCH * sizeof(Node*));
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
      SEARCH_V3(
            &tree,
            dataset_queries,
            results,
            NUM_TO_SEARCH,
            NUM_THREADS,
            VERBOSE
      );

      // Check the search results
      if (VERBOSE) {
            printf("Search results after search:\n");
            for (int i = 0; i < NUM_TO_SEARCH; i++) {
                  printf("Value %04u: ", dataset_queries[i]);
                  if (results[i] == NULL)
                        printf("NOT FOUND\n");
                  else
                        printf(" %p (value=%04u)\n", (void*)results[i], results[i]->value);
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
