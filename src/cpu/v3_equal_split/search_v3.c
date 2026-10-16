#include "search_v3.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Arguments of each worker thread
typedef struct pthread_arg {
      // Subtree to examine
      Node* subtree_root;

      // Total number of values to search
      int max_index_to_search;

      // Initial offset of the thread
      int initial_offset;

      // Number of threads
      int num_threads;

      // Array of values to search
      uint32_t* values_to_search;

      // Array of the nodes found, in the same order as the values to search
      Node** results;
} pthread_arg;


// Array of threads
static pthread_t* thread_array;

// Array of arguments passed to each thread
static pthread_arg* thread_args;


/**
 * Initializes the global variables
 * @param number_threads Number of threads to create
 */
static void initialize_global_vars(int number_threads) {

      thread_array = malloc(sizeof(pthread_t) * number_threads);

      thread_args = malloc(sizeof(pthread_arg) * number_threads);
}


/**
 * Body of the thread that performs the searches
 * @param arg Thread arguments
 * @returns NULL
 */
static void* pthread_SEARCH_V3(void* arg) {

      pthread_arg* args = (pthread_arg*)arg;

      // Search all the values assigned to this thread
      for (
            int i = args->initial_offset;
            i < args->max_index_to_search;
            i += args->num_threads
      ) {
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
void SEARCH_V3(
      BinTree* tree,
      uint32_t* values_to_search,
      Node** return_values,
      int NUM_TO_SEARCH,
      int NUM_THREADS,
      bool VERBOSE
) {

// STEP 0: Initialize the global variables
      initialize_global_vars(NUM_THREADS);
      if (VERBOSE) {
            printf("Global variables initialized\n");
      }


// STEP 1: Assign the arguments to the threads

      for (int i = 0; i < NUM_THREADS; i++) {

            // Every thread uses the whole tree as its root
            thread_args[i].subtree_root = tree->root;

            // Assign the array of values to search
            thread_args[i].values_to_search = values_to_search;

            // Assign the array of return values
            thread_args[i].results = return_values;

            // Assign the number of values to search
            thread_args[i].max_index_to_search = NUM_TO_SEARCH;

            // Assign the initial offset of the thread
            thread_args[i].initial_offset = i;

            // Assign the number of threads
            thread_args[i].num_threads = NUM_THREADS;

            if (VERBOSE) {
                  printf("Thread %d ready\n", i);
            }
      }


// STEP 2: Start the searching threads

      for (int i = 0; i < NUM_THREADS; i++) {

            // Create and start the thread
            pthread_create(
                  &thread_array[i],
                  NULL,
                  pthread_SEARCH_V3,
                  &thread_args[i]
            );

            if (VERBOSE) {
                  printf("Thread %d started\n", i);
            }
      }


// STEP 3: Wait for all the threads to terminate

      for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(thread_array[i], NULL);
      }

      if (VERBOSE) {
            printf("All threads terminated\n");
      }


// STEP 4: Free the memory
      free(thread_array);
      free(thread_args);
}
