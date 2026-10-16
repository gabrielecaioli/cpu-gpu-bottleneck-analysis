#include "range_utils.h"

#include <stdio.h>
#include <stdlib.h>

/**
 * Computes the minimum of the range of values of the subtree rooted at the given node
 * @param tree Binary tree
 * @param root Root node of the subtree
 * @param range_min Pointer where the minimum of the search range is written
 * @returns true if the computation succeeded, false otherwise
 */
bool computeRangeMin(BinTree* tree, Node* root, int* range_min) {
      if (!tree || !root) {
            return false;
      }

      int min = MINVAL;
      Node* father = tree->root;
      while (father != root) {
            if (root->value < father->value) {
                  // left: the minimum is the father's minimum, set previously
                  father = father->left;
            } else {
                  // right: the minimum is the father's value + 1
                  min = (father->value + 1);
                  father = father->right;
            }
      }

      *range_min = min;
      return true;
}

/**
 * Computes the maximum of the range of values of the subtree rooted at the given node
 * @param tree Binary tree
 * @param root Root node of the subtree
 * @param range_max Pointer where the maximum of the search range is written
 * @returns true if the computation succeeded, false otherwise
 */
bool computeRangeMax(BinTree* tree, Node* root, int* range_max) {
      if (!tree || !root) {
            return false;
      }
      int max = MAXVAL;
      Node* father = tree->root;
      while (father != root) {
            if (root->value < father->value) {
                  // left: the maximum is the father's value - 1
                  max = (father->value - 1);
                  father = father->left;
            } else {
                  // right: the maximum is the father's maximum, set previously
                  father = father->right;
            }
      }
      *range_max = max;
      return true;
}


/**
 * Splits the input vector into sub-vectors according to the ranges, saving the original indices
 * @param input Input vector to split
 * @param input_size Size of the input vector
 * @param NUM_THREADS Number of sub-vectors to create
 * @param ranges Ranges of each sub-vector (min, max)
 * @param output Array of pointers to the sub-vectors
 * @param indices Array of pointers to the original indices
 * @param output_sizes Sizes of the sub-vectors
 * @note All the arrays must already be correctly allocated
 */
void separateVectorWithIndices(
      uint32_t* input,
      int input_size,
      int NUM_THREADS,
      int* ranges,
      uint32_t** output,
      int** indices,
      int* output_sizes
) {

      // Assign the values according to the ranges
      for (int i = 0; i < input_size; i++) {

            // Value to assign
            uint32_t value = input[i];

            bool found = false;

            // Check all the sub-vectors
            for (int j = 0; j < NUM_THREADS; j++) {

                  // Check whether the value belongs to the range of the j-th sub-vector
                  uint32_t min = ranges[j * 2];
                  uint32_t max = ranges[j * 2 + 1];

                  // If the value is within the range, assign it to the sub-vector
                  if (value >= min && value <= max) {
                        int pos = output_sizes[j];

                        // Save the value
                        output[j][pos] = value;

                        // Save the original index
                        indices[j][pos] = i;

                        // Increase the size of the sub-vector
                        output_sizes[j]++;

                        found = true;

                        // A value belongs to only one range
                        break;
                  }
            }

            if (!found)
                  printf("\tValue %u not assigned to any sub-vector\n", value);
      }
}


/**
 * Splits the input vector into sub-vectors according to the ranges, saving the original indices.
 * All the values that do not belong to any range are put in a final, extra sub-vector.
 * @param input Input vector to split
 * @param input_size Size of the input vector
 * @param NUM_THREADS Number of sub-vectors to create. It does NOT include the final sub-vector.
 * @param ranges Ranges of each sub-vector (min, max)
 * @param output Array of pointers to the sub-vectors
 * @param indices Array of pointers to the original indices
 * @param output_sizes Sizes of the sub-vectors
 * @note All the arrays must already be correctly allocated
 */
void separateVectorWithIndices_extraThread(
      uint32_t* input,
      int input_size,
      int NUM_THREADS,
      int* ranges,
      uint32_t** output,
      int** indices,
      int* output_sizes
) {

      // Assign the values according to the ranges
      for (int i = 0; i < input_size; i++) {

            // Value to assign
            uint32_t value = input[i];

            bool found = false;

            // Check all the sub-vectors
            for (int j = 0; j < NUM_THREADS; j++) {

                  // Check whether the value belongs to the range of the j-th sub-vector
                  uint32_t min = ranges[j * 2];
                  uint32_t max = ranges[j * 2 + 1];

                  // If the value is within the range, assign it to the sub-vector
                  if (value >= min && value <= max) {
                        int pos = output_sizes[j];

                        // Save the value
                        output[j][pos] = value;

                        // Save the original index
                        indices[j][pos] = i;

                        // Increase the size of the sub-vector
                        output_sizes[j]++;

                        found = true;

                        // A value belongs to only one range
                        break;
                  }
            }

            // If it does not belong to any range, put it in the final sub-vector (index NUM_THREADS)
            if (!found) {
                  int pos = output_sizes[NUM_THREADS];

                  // Save the value
                  output[NUM_THREADS][pos] = value;

                  // Save the original index
                  indices[NUM_THREADS][pos] = i;

                  // Increase the size of the sub-vector
                  output_sizes[NUM_THREADS]++;
            }
      }
}


/**
 * Rebuilds the original vector from the sub-vectors and the original indices
 * @param separated_vectors Array of pointers to the sub-vectors
 * @param indices Array of pointers to the original indices
 * @param sizes Sizes of the sub-vectors
 * @param output Rebuilt output vector
 * @param NUM_THREADS Number of sub-vectors
 */
void recomposeNodeVectors(
      Node*** separated_vectors,
      int** indices,
      int* sizes,
      Node** output,
      int NUM_THREADS
) {
      for (int i = 0; i < NUM_THREADS; i++) {
            for (int j = 0; j < sizes[i]; j++) {
                  int originalIndex = indices[i][j];
                  output[originalIndex] = separated_vectors[i][j];
            }
      }
}
