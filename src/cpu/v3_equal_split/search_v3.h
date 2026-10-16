#ifndef SEARCH_V3_H
#define SEARCH_V3_H

#include "bintree.h"

/*
 * CPU Version 3 - Equal split.
 *
 * The vector of values to search is logically divided into NUM_THREADS parts
 * of equal size (thread i handles the values at indices i, i + NUM_THREADS,
 * i + 2 * NUM_THREADS, ...). Every thread starts its searches from the root
 * of the tree, so no ranges or subtrees have to be computed and the load is
 * balanced even if the tree is not.
 */

void SEARCH_V3(
      BinTree* tree,
      uint32_t* values_to_search,
      Node** return_values,
      int NUM_TO_SEARCH,
      int NUM_THREADS,
      bool VERBOSE
);

#endif // SEARCH_V3_H
