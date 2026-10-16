#ifndef SEARCH_V2_H
#define SEARCH_V2_H

#include "bintree.h"

/*
 * CPU Version 2 - Range split.
 *
 * Before searching, the vector of values to search is split into one
 * sub-vector per thread, according to the range of values covered by the
 * subtree each thread owns. Every thread then searches its own sub-vector in
 * its own subtree, independently from the others. An extra thread searches,
 * starting from the root, the values that fall in the top levels of the tree.
 */

void SEARCH_FULL_PARALLEL(
      BinTree* tree,
      uint32_t* values_to_search,
      Node** return_values,
      int NUM_TO_SEARCH,
      int NUM_THREADS,
      bool VERBOSE
);

#endif // SEARCH_V2_H
