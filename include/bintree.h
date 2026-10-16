#ifndef BINTREE_H
#define BINTREE_H

#include <stdbool.h>
#include <stdint.h>

// Node structure
typedef struct Node {
    uint32_t value;
    struct Node* left;
    struct Node* right;
} Node;

// Binary search tree
typedef struct BinTree {
    Node* root;
} BinTree;

#ifdef __cplusplus
extern "C" {
#endif

void bintree_init(BinTree* tree);
void bintree_init_from_array(BinTree* tree, uint32_t* arr, uint32_t size);
Node* insert_rec(Node* node, uint32_t value);
void insert(BinTree* tree, uint32_t value);
Node* search_rec(Node* node, uint32_t value);
Node* search(BinTree* tree, uint32_t value);
Node* find_minimum(Node* node);
bool extract(BinTree* tree, uint32_t value);
void tree_destroy(Node* node);
void bintree_free(BinTree* tree);
void print_tree(Node* node, int level);
void print_tree_max_levels(Node* node, int level, int max_level);

#ifdef __cplusplus
}
#endif

#endif // BINTREE_H
