#ifndef RANGE_UTILS_H
#define RANGE_UTILS_H

#include "bintree.h"

// Minimum and maximum value a node can take (used for the extreme subtrees)
#define MINVAL 0
#define MAXVAL UINT32_MAX

bool computeRangeMin(BinTree* tree, Node* root, int* range_min);
bool computeRangeMax(BinTree* tree, Node* root, int* range_max);

void separateVectorWithIndices(
      uint32_t* input,
      int input_size,
      int NUM_THREADS,
      int* ranges,
      uint32_t** output,
      int** indices,
      int* output_sizes
);

void separateVectorWithIndices_extraThread(
      uint32_t* input,
      int input_size,
      int NUM_THREADS,
      int* ranges,
      uint32_t** output,
      int** indices,
      int* output_sizes
);

void recomposeNodeVectors(
      Node*** separated_vectors,
      int** indices,
      int* sizes,
      Node** output,
      int NUM_THREADS
);

#endif // RANGE_UTILS_H
