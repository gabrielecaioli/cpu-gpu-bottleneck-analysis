#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "bintree.h"
#include "search_v1.h"

/*
///////////////////////////////////////////
ARGUMENTS:
      <NUM_THREADS> <NUM_NODES> <NUM_TO_SEARCH> <SEED> <MIN> <MAX> <TO_EXTRACT> <VERBOSE (0 or 1)>
Generates the arrays, builds the tree from them, assigns the searches to the
threads, removes some nodes.
///////////////////////////////////////////
*/

int main(int argc, char* argv[]) {

      // Default arguments, in the order they must be given on the command line
      int NUM_THREADS   = 8;
      int NUM_NODES     = 100000;
      int NUM_TO_SEARCH = 1000000;
      int SEED          = time(NULL);
      int MIN           = 10;
      int MAX           = 10000;
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
            SEED = atoi(argv[4]);
      if (argc > 5)
            MIN = atoi(argv[5]);
      if (argc > 6)
            MAX = atoi(argv[6]);
      if (argc > 7)
            TO_EXTRACT = atoi(argv[7]);
      if (argc > 8)
            VERBOSE = atoi(argv[8]);

      // Check that the parameters are valid
      if (NUM_THREADS < 1 || NUM_NODES < 1 || NUM_TO_SEARCH < 1 || MAX < MIN) {
            printf("Error: Invalid arguments.\n");
            printf("Usage: %s <NUM_THREADS> <NUM_NODES> <NUM_TO_SEARCH> <SEED> <MIN> <MAX> <TO_EXTRACT> <VERBOSE (0 or 1)>\n", argv[0]);
            return EXIT_FAILURE;
      }


      if (VERBOSE) printf("\n REGION 1: THREADS CREATION =========================\n");
      initialize_global_vars(NUM_THREADS);
      initialize_threads();


      if (VERBOSE) printf("\n REGION 2: ARRAY & BINTREE CREATION =================\n");
      // Random seed
      srand(SEED);

      uint32_t* arr = (uint32_t*)malloc(NUM_NODES * sizeof(uint32_t));       // values to insert
      uint32_t* tos = (uint32_t*)malloc(NUM_TO_SEARCH * sizeof(uint32_t));   // values to search
      Node** res = (Node**)malloc(NUM_TO_SEARCH * sizeof(Node*));            // search results

      if (arr == NULL || tos == NULL || res == NULL) {
            perror("Failed to allocate memory for arrays");
            return EXIT_FAILURE;
      }

      // Generate the values
      for (int i = 0; i < NUM_NODES; ++i) {
            // Random values from MIN to MAX included
            arr[i] = (uint32_t)(rand() % (MAX - MIN + 1) + MIN);
      }
      for (int i = 0; i < NUM_TO_SEARCH; ++i) {
            tos[i] = (uint32_t)(rand() % (MAX - MIN + 1) + MIN);
            res[i] = NULL;
      }
      BinTree tree;
      bintree_init(&tree);


      if (VERBOSE) printf("\n REGION 3: BINTREE INSERTIONS =======================\n");
      for (int i = 0; i < NUM_NODES; ++i) {
            if (VERBOSE) printf("%03d:%03d ", i, arr[i]);
            insert(&tree, arr[i]);
      }

      if (VERBOSE) printf("\nFinal tree structure, only top 4 layers\n");
      if (VERBOSE) print_tree_max_levels(tree.root, 0, 4);


      if (VERBOSE) printf("\n REGION 4: ASSIGN SUBTREE ROOTS =====================\n");
      // Assign a subtree to each thread
      update_subroots(&tree);


      if (VERBOSE) printf("\n REGION 5: SEARCH (a lot of) VALUES =================\n");
      if (VERBOSE) {
            for (int i = 0; i < NUM_TO_SEARCH; ++i) {
                  printf("%03d:%03d ", i, tos[i]);
            }
            printf("\n");
      }
      SEARCH_THREAD(&tree, tos, res, NUM_TO_SEARCH);

      // Check the search results
      if (VERBOSE) {
            printf("Search results after search:\n");
            for (int i = 0; i < NUM_TO_SEARCH; i++) {
                  printf("Value %04u: ", tos[i]);
                  if (res[i] == NULL)
                        printf("NOT FOUND\n");
                  else
                        printf(" %p (value=%04u)\n", (void*)res[i], res[i]->value);
            }
      }


      if (VERBOSE) printf("\n REGION 6: REMOVE SOME VALUES =======================\n");
      int deleted_nodes = 0;
      while ((deleted_nodes < TO_EXTRACT) && (tree.root != NULL)) {
            int i = rand() % NUM_NODES;
            if (extract(&tree, arr[i]))
                  deleted_nodes++;
      }


      if (VERBOSE) printf("\n REGION 7: UNLOCK & JOIN THREADS ====================\n");
      terminate_threads();


      if (VERBOSE) printf("\n REGION 8: DEALLOCATE ENTIRE TREE ===================\n");
      bintree_free(&tree);
      free(arr);
      free(tos);
      free(res);

      return 0;
}
