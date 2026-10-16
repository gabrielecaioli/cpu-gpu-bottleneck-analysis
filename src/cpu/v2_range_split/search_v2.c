#include "search_v2.h"
#include "range_utils.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Arguments of each worker thread
typedef struct pthread_arg {
      // Subtree to examine
      Node* subtree_root;

      // Number of values to search
      int num_values_to_search;

      // Array of values to search
      uint32_t* values_to_search;

      // Array of the nodes found, in the same order as the values to search
      Node** results;

      // No KILL flag is needed: the thread terminates once its searches are done
} pthread_arg;


// Array of threads
static pthread_t* thread_array;

// Array of arguments passed to each thread
static pthread_arg* thread_args;

// Right path array (true => right, false => left)
static bool* right_path;


/**
 * Initializes the global variables
 * @param number_threads Number of threads to create
 */
static void initialize_global_vars(int number_threads) {

      thread_array = malloc(sizeof(pthread_t) * number_threads);

      thread_args = malloc(sizeof(pthread_arg) * number_threads);

      right_path = malloc(sizeof(bool) * number_threads);
}


/**
 * Computes the path to the node with the given index, where the index refers
 * to a breadth-first visit of the tree with the root at index 1.
 * @param index Index of the node to compute the path to
 * @param right_path Array of booleans representing the path to follow (true = right, false = left)
 * @returns the length of the computed path
 */
static int thread_index_to_right_path(int index, bool* right_path) {
      // Convert the index to binary and store it in the boolean array
      int path_len = floor(log2(index));

      for (int i = 0; i < path_len; i++) {
            right_path[i] = (index & (1 << (path_len - i - 1))) != 0 ? true : false;
      }

      return path_len; // Path length
}

/**
 * Finds a node given the path to reach it
 * @param tree Binary tree
 * @param right_path Array of booleans representing the path to follow (true = right, false = left)
 * @param path_len Length of the path
 * @returns The node found (NULL if the path leaves the tree)
 */
static Node* get_node_from_right_path(BinTree* tree, bool* right_path, int path_len) {
      Node* node = tree->root;
      for (int i = 0; i < path_len && node != NULL; i++) {
            if (right_path[i] == true) {
                  node = node->right;
            }
            else {
                  node = node->left;
            }
      }
      return node;
}

/**
 * Finds the root node of the subtree to assign to a thread
 * @param right_path Array of booleans representing the path to follow (true = right, false = left)
 * @param node Node from which the search starts
 * @param level Current level of the search
 * @param target_level Level to reach
 * @return Root node of the subtree to assign to the thread
 */
static Node* search_subroot_node(bool* right_path, Node* node, int level, int target_level) {
      if (node == NULL)
            return NULL;

      if (level == target_level)
            return node;

      if (right_path[level] == false) {
            return search_subroot_node(right_path, node->left, level + 1, target_level);
      }
      else {
            return search_subroot_node(right_path, node->right, level + 1, target_level);
      }
}

/**
 * Finds the search range of each thread
 * @param tree Binary tree
 * @param NUM_THREADS Number of threads, equal to the number of ranges
 * @param ranges Array of ranges to fill
 */
static void find_ranges(
      BinTree* tree,
      int NUM_THREADS,
      int* ranges
) {

      // Compute the range of each thread
      for (int i = 0; i < NUM_THREADS; i++) {

            // Index of the i-th node
            int index = i + NUM_THREADS;

            // Compute the path for the i-th thread
            int right_path_len = thread_index_to_right_path(index, right_path);

            // Find the root node of the subtree to examine
            Node* subroot = get_node_from_right_path(tree, right_path, right_path_len);

            // Compute the range for the i-th thread
            int retMin = computeRangeMin(tree, subroot, &ranges[i * 2]);
            int retMax = computeRangeMax(tree, subroot, &ranges[i * 2 + 1]);

            // In case of error, the range has min > max (empty)
            if (!retMin || !retMax) {
                  ranges[i * 2] = 1;
                  ranges[i * 2 + 1] = 0;
            }
      }

}


/**
 * Body of the thread that performs the searches
 * @param arg Thread arguments
 * @returns NULL
 */
static void* pthread_FULL_SEARCH(void* arg) {
      pthread_arg* args = (pthread_arg*)arg;

      // Search all the assigned values
      for (int i = 0; i < args->num_values_to_search; i++) {
            args->results[i] = search_rec(
                  args->subtree_root,
                  args->values_to_search[i]
            );
      }

      // Terminate the thread
      pthread_exit(NULL);
}

/**
 * Main function of the parallel search
 * @param tree Binary tree
 * @param values_to_search Array of values to search
 * @param return_values Array of the nodes found, in the same order as the values to search
 * @param NUM_TO_SEARCH Number of values to search
 * @param NUM_THREADS Number of threads to create
 * @param VERBOSE Flag to print debug messages
 */
void SEARCH_FULL_PARALLEL(
      BinTree* tree,
      uint32_t* values_to_search,
      Node** return_values,
      int NUM_TO_SEARCH,
      int NUM_THREADS,
      bool VERBOSE
) {
      bool with_extra_thread = true; // Whether to use an extra thread
      int NUM_THREADS_with_extra = NUM_THREADS + ((with_extra_thread == true) ? 1 : 0); // Includes the final vector

// STEP 0: Initialize the global variables
      initialize_global_vars(NUM_THREADS_with_extra);


// STEP 1: Find the search range of each thread

      // Search range of each thread (min, max), so two values per thread
      int* ranges;
      ranges = malloc(sizeof(int) * NUM_THREADS * 2);
      if (ranges == NULL) {
            perror("Failed to allocate memory for ranges\n");
            exit(EXIT_FAILURE);
      }
      find_ranges(tree, NUM_THREADS, ranges);
      if (VERBOSE) {
            printf("Ranges:\n");
            for (int i = 0; i < NUM_THREADS; i++) {
                  printf("Thread %d: [%u, %u]\n", i, (uint32_t)ranges[i * 2], (uint32_t)ranges[i * 2 + 1]);
            }
      }

// STEP 2: Create the separate vectors

      uint32_t** values_to_search_separated = malloc(sizeof(uint32_t*) * NUM_THREADS_with_extra);
      int** indexes = (int**)malloc(NUM_THREADS_with_extra * sizeof(int*));
      int* output_sizes = (int*)malloc(NUM_THREADS_with_extra * sizeof(int));
      if (values_to_search_separated == NULL || indexes == NULL || output_sizes == NULL) {
            perror("Failed to allocate memory for values to search separated, indexes or output sizes\n");
            exit(EXIT_FAILURE);
      }
      for (int i = 0; i < NUM_THREADS_with_extra; i++) {
            values_to_search_separated[i] = malloc(sizeof(uint32_t) * NUM_TO_SEARCH);
            indexes[i] = malloc(sizeof(int) * NUM_TO_SEARCH);
            if (values_to_search_separated[i] == NULL || indexes[i] == NULL) {
                  perror("Failed to allocate memory for values to search separated or indexes\n");
                  exit(EXIT_FAILURE);
            }
            output_sizes[i] = 0; // Initialize the sizes of the sub-vectors
      }

      if (with_extra_thread)
            separateVectorWithIndices_extraThread(
                  values_to_search,
                  NUM_TO_SEARCH,
                  NUM_THREADS, // one vector per thread
                  ranges,
                  values_to_search_separated,
                  indexes,
                  output_sizes
            );
      else
            separateVectorWithIndices(
                  values_to_search,
                  NUM_TO_SEARCH,
                  NUM_THREADS, // one vector per thread
                  ranges,
                  values_to_search_separated,
                  indexes,
                  output_sizes
            );

      if (VERBOSE) {
            printf("Values to search separated\n");
            for (int i = 0; i < NUM_THREADS_with_extra; i++) {
                  printf(">> Thread %d (size = %d): ", i, output_sizes[i]);
                  for (int j = 0; j < output_sizes[i]; j++) {
                        printf("%u ", values_to_search_separated[i][j]);
                  }
                  printf("\n");
            }
      }


// STEP 3: Start the searching threads
      int right_path_len;
      Node*** return_values_separated;

      return_values_separated = malloc(sizeof(Node**) * NUM_THREADS_with_extra);

      // NUM_THREADS threads are started normally
      for (int i = 0; i < NUM_THREADS; i++) {

            // Compute the path for the i-th thread
            right_path_len = thread_index_to_right_path(i + NUM_THREADS, right_path);

            // Find the root node of the subtree to examine
            thread_args[i].subtree_root = search_subroot_node(
                  right_path,
                  tree->root,
                  0,
                  right_path_len
            );

            // Assign the array of values to search
            thread_args[i].values_to_search = values_to_search_separated[i];

            // Assign the number of values to search
            thread_args[i].num_values_to_search = output_sizes[i];

            // Allocate the single return array
            return_values_separated[i] = malloc(sizeof(Node*) * output_sizes[i]);
            if (return_values_separated[i] == NULL) {
                  perror("Failed to allocate memory for return values separated\n");
                  exit(EXIT_FAILURE);
            }

            // Assign the array of return values
            thread_args[i].results = return_values_separated[i];

            // Create and start the thread
            pthread_create(
                  &thread_array[i],
                  NULL,
                  pthread_FULL_SEARCH,
                  &thread_args[i]
            );

            if (VERBOSE) {
                  printf("Thread %d started\n", i);
            }
      }

      // Last thread, which handles the values not assigned to any range
      if (with_extra_thread == true) {
            int i = NUM_THREADS; // Last thread

            // The last thread searches from the root of the tree
            thread_args[i].subtree_root = tree->root;

            // Assign the array of values to search
            thread_args[i].values_to_search = values_to_search_separated[i];

            // Assign the number of values to search
            thread_args[i].num_values_to_search = output_sizes[i];

            // Allocate the single return array
            return_values_separated[i] = malloc(sizeof(Node*) * output_sizes[i]);
            if (return_values_separated[i] == NULL) {
                  perror("Failed to allocate memory for return values separated\n");
                  exit(EXIT_FAILURE);
            }

            // Assign the array of return values
            thread_args[i].results = return_values_separated[i];

            // Create and start the thread
            pthread_create(
                  &thread_array[i],
                  NULL,
                  pthread_FULL_SEARCH,
                  &thread_args[i]
            );

            if (VERBOSE) {
                  printf("Thread %d (FINAL) started\n", i);
            }

      }


// STEP 4: Wait for all the threads to terminate
      for (int i = 0; i < NUM_THREADS_with_extra; i++) {
            pthread_join(thread_array[i], NULL);
      }

      if (VERBOSE) {
            printf("All threads terminated\n");
      }

// STEP 5: Merge the results of the threads
      recomposeNodeVectors(
            return_values_separated,
            indexes,
            output_sizes,
            return_values,
            NUM_THREADS_with_extra
      );

      if (VERBOSE) {
            printf("Results recomposed\n");
      }

// STEP 6: Free the memory
      free(thread_array);
      free(thread_args);
      free(right_path);
      free(ranges);
      for (int i = 0; i < NUM_THREADS_with_extra; i++) {
            free(values_to_search_separated[i]);
            free(return_values_separated[i]);
            free(indexes[i]);
      }
      free(values_to_search_separated);
      free(return_values_separated);
      free(indexes);
      free(output_sizes);

}
