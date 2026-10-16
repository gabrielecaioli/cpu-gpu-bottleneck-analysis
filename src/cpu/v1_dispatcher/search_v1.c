#include "search_v1.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/*
/////////////////////////////////////////////
MULTITHREADING FUNCTIONS
/////////////////////////////////////////////
*/

// Arguments of each worker thread
typedef struct pthread_arg {
      Node* subtree_root;                      // Subtree to examine
      uint32_t target_value;                   // Value to search for
      Node** result;                           // Where to write the search outcome (pointer to the node if found)
      pthread_mutex_t mutex;                   // Mutex for synchronization
      pthread_cond_t cond_start_search;        // Condition variable signalled to start a search
      pthread_cond_t cond_search_finished;     // Condition variable signalled when the search is finished
      bool start_search;                       // Flag: a new search has been assigned
      bool search_finished;                    // Flag: the last search has finished
      bool KILL;                               // Flag: the thread must terminate
} pthread_arg;


// Array of threads
static pthread_t* thread_array;

// Array of arguments passed to each thread
static pthread_arg* args;

// Right path array (true => right, false => left)
static bool* right_path;

// Number of threads
static int NUM_THREADS;

// Maximum tree level at which the threads' subtrees are located
static int MAX_PATH_LEVEL;

void initialize_global_vars(int number_threads) {
      NUM_THREADS = number_threads;
      MAX_PATH_LEVEL = (int)ceil(log2(NUM_THREADS)) + 1;  // ceil rounds up to the next integer
      thread_array = malloc(sizeof(pthread_t) * NUM_THREADS);
      args = malloc(sizeof(pthread_arg) * NUM_THREADS);
      right_path = malloc(sizeof(bool) * MAX_PATH_LEVEL);
}

// Body of the worker thread that performs the searches
static void* pthread_search(void* arg) {
      pthread_arg* args = (pthread_arg*)arg;
      Node* searched;

      // Stay alive until termination is requested
      while (1) {

            // Wait to be unblocked to start the search
            pthread_mutex_lock(&args->mutex);
            while (!args->start_search) {
                  // Wait until a search is assigned
                  pthread_cond_wait(&args->cond_start_search, &args->mutex);
            }
            args->start_search = false; // Reset the flag

            // If termination was requested, exit
            if (args->KILL == true) {
                  pthread_mutex_unlock(&args->mutex);
                  break;
            }

            pthread_mutex_unlock(&args->mutex);

            // Take the value to search for and search it
            searched = search_rec(
                  args->subtree_root,
                  args->target_value
            );

            // Write the result in the dedicated slot
            *(args->result) = searched;

            // Signal that the search is finished
            pthread_mutex_lock(&args->mutex);
            args->search_finished = true;
            pthread_cond_signal(&args->cond_search_finished);
            pthread_mutex_unlock(&args->mutex);
      }

      return NULL;
}

void initialize_threads(void) {

      for (int i = 0; i < NUM_THREADS; i++) {

            // Initialize the mutex
            pthread_mutex_init(&args[i].mutex, NULL);

            // Initialize the condition variables
            pthread_cond_init(&args[i].cond_start_search, NULL);
            pthread_cond_init(&args[i].cond_search_finished, NULL);

            // Define the thread arguments
            args[i].subtree_root = NULL; // This field is set later
            args[i].start_search = false;
            args[i].search_finished = true;
            args[i].KILL = false;

            // Create the thread
            pthread_create(
                  &thread_array[i],
                  NULL,
                  pthread_search,
                  &args[i]
            );
      }
}

// Blocks until the given worker thread has finished its current search
static void wait_search_finished(int thread_index) {
      pthread_mutex_lock(&args[thread_index].mutex);
      while (!args[thread_index].search_finished) {
            pthread_cond_wait(
                  &args[thread_index].cond_search_finished,
                  &args[thread_index].mutex
            );
      }
      pthread_mutex_unlock(&args[thread_index].mutex);
}

void terminate_threads(void) {

      for (int i = 0; i < NUM_THREADS; i++) {
            pthread_mutex_lock(&args[i].mutex);
            args[i].start_search = true;
            args[i].KILL = true;
            pthread_cond_signal(&args[i].cond_start_search);
            pthread_mutex_unlock(&args[i].mutex);
      }

      for (int i = 0; i < NUM_THREADS; i++) {
            // Wait for the termination of a specific thread
            pthread_join(thread_array[i], NULL);
      }

      free(thread_array);
      free(args);
      free(right_path);
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
 * Assigns the correct subtree to each thread, setting the subtree_root field
 * @param tree Binary tree from which the subtrees are taken
 */
void update_subroots(BinTree* tree) {
      int right_path_len;
      for (int i = 0; i < NUM_THREADS; i++) {
            // Compute the path for the i-th thread
            right_path_len = thread_index_to_right_path(i + NUM_THREADS, right_path);

            // Find the root node of the subtree to examine
            args[i].subtree_root = search_subroot_node(right_path, tree->root, 0, right_path_len);
      }
}

// Coordinates/assigns the searches to the threads
void SEARCH_THREAD(
      BinTree* tree,
      uint32_t* values_to_search,
      Node** return_values,
      int NUM_TO_SEARCH  // number of values to search
)
{
      uint32_t value_to_search;
      Node* n;

      for (int i = 0; i < NUM_TO_SEARCH; ++i) {
            value_to_search = values_to_search[i];
            n = tree->root;
            int node_index = 1;

            // Search the value in the top levels of the tree (above the threads' subtrees)
            while (node_index < NUM_THREADS && n != NULL && n->value != value_to_search) {
                  if (value_to_search < n->value) {
                        n = n->left;
                        node_index = node_index * 2;
                  }
                  else {
                        n = n->right;
                        node_index = node_index * 2 + 1;
                  }
            }

            // If the pointer is NULL there is no point in calling the threads
            if (n == NULL) {
                  continue;
            }

            // Value found by the main thread
            if (value_to_search == n->value) {
                  return_values[i] = n;
                  continue;
            }

            // Index of the thread to schedule (tts = Thread To Schedule)
            int tts = node_index - NUM_THREADS;

            // Wait until a possible previous search performed by the "chosen" thread has finished
            wait_search_finished(tts);

            // Schedule the search
            pthread_mutex_lock(&args[tts].mutex);
            args[tts].target_value = value_to_search;
            args[tts].search_finished = false;
            args[tts].start_search = true;
            args[tts].result = &return_values[i];
            pthread_cond_signal(&args[tts].cond_start_search);
            pthread_mutex_unlock(&args[tts].mutex);
      }

      // Wait for the last searches still running, so that every result is
      // written before the caller reads them or frees the tree
      for (int t = 0; t < NUM_THREADS; t++) {
            wait_search_finished(t);
      }
}
