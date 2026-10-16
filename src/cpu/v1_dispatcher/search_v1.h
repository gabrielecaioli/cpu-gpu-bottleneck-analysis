#ifndef SEARCH_V1_H
#define SEARCH_V1_H

#include "bintree.h"

/*
 * CPU Version 1 - Thread dispatcher.
 *
 * NUM_THREADS persistent worker threads are created, each one owning the
 * subtree rooted at one node of level log2(NUM_THREADS). The main thread walks
 * the top levels of the tree for every value to search and hands the search
 * over to the worker that owns the corresponding subtree, waiting if that
 * worker is still busy with its previous search.
 */

void initialize_global_vars(int number_threads);
void initialize_threads(void);
void update_subroots(BinTree* tree);
void SEARCH_THREAD(
      BinTree* tree,
      uint32_t* values_to_search,
      Node** return_values,
      int NUM_TO_SEARCH
);
void terminate_threads(void);

#endif // SEARCH_V1_H
