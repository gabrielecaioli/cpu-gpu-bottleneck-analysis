#include "bintree.h"
#include <stdlib.h>
#include <stdio.h>

// Leaf node creation
static Node* create_node(uint32_t value) {
      Node* new_node = (Node*)malloc(sizeof(Node));
      if (new_node == NULL) {
            perror("Failed to allocate memory for new node");
            exit(EXIT_FAILURE);
      }
      new_node->value = value;
      new_node->left = NULL;
      new_node->right = NULL;
      return new_node;
}

// BinTree initialization, empty constructor
void bintree_init(BinTree* tree) {
      tree->root = NULL;
}

// BinTree initialization, from array
void bintree_init_from_array(BinTree* tree, uint32_t* arr, uint32_t size) {
      // Initialize an empty tree
      bintree_init(tree);

      // Loop through the array and insert each value into the binary tree
      for (uint32_t i = 0; i < size; ++i) {
            insert(tree, arr[i]);
      }
}

// Recursive insert function
Node* insert_rec(Node* node, uint32_t value) {

      // If the node is NULL, instantiate it
      if (node == NULL) {
            return create_node(value);
      }

      if (node->value == value) {
            // Value already exists, do not insert duplicates
            return node;
      }

      // Otherwise insert in one of the subtrees
      if (value < node->value)
            node->left = insert_rec(node->left, value);
      else
            node->right = insert_rec(node->right, value);

      return node;
}

// Insertion
void insert(BinTree* tree, uint32_t value) {
      tree->root = insert_rec(tree->root, value);
}

// Recursive search function
Node* search_rec(Node* node, uint32_t value) {
      if (node == NULL)
            return NULL;

      if (node->value == value)
            return node;

      if (value < node->value)
            return search_rec(node->left, value);
      else
            return search_rec(node->right, value);
}

// Node search
Node* search(BinTree* tree, uint32_t value) {
      return search_rec(tree->root, value);
}

// Finds the minimum in a subtree
Node* find_minimum(Node* node) {
      while (node != NULL && node->left != NULL) {
            node = node->left;
      }
      return node;
}

// Recursive value extraction. Sets *removed to true if the value was found and deleted.
static Node* extract_rec(Node* node, uint32_t value, bool* removed) {

      if (node == NULL)
            return NULL;

      // Recursive search of the target node, updating the tree on the way back
      if (value < node->value) {
            node->left = extract_rec(node->left, value, removed);
            return node;
      }
      if (value > node->value) {
            node->right = extract_rec(node->right, value, removed);
            return node;
      }

      // Node found, delete it
      *removed = true;

      // If it has an empty subtree, just move the other subtree up one level
      if (node->left == NULL) { // only right subtree, or leaf node
            Node* tmp = node->right;
            free(node);
            return tmp;
      }
      if (node->right == NULL) { // only left subtree
            Node* tmp = node->left;
            free(node);
            return tmp;
      }

      // Both subtrees present, so:
      // 1) Find the minimum in the right subtree
      Node* min_right = find_minimum(node->right);

      // 2) Place the right subtree minimum in place of the node to be deleted
      node->value = min_right->value;

      // 3) Delete the (now duplicated) minimum from the right subtree
      bool ignored = false;
      node->right = extract_rec(node->right, node->value, &ignored);

      return node;
}

// Value extraction. Returns true if the value was found and removed.
bool extract(BinTree* tree, uint32_t value) {
      if (tree == NULL)
            return false;

      bool removed = false;
      tree->root = extract_rec(tree->root, value, &removed);
      return removed;
}

// Tree destruction
void tree_destroy(Node* node) {
      if (node != NULL) {
            tree_destroy(node->left);
            tree_destroy(node->right);
            free(node);
      }
}

// BinTree destructor
void bintree_free(BinTree* tree) {
      tree_destroy(tree->root);
      tree->root = NULL;
}

// Space between tree levels when printing
static const int SPACE = 3;

// Prints the tree rotated by 90 degrees (root on the left)
void print_tree(Node* node, int level) {
      if (node == NULL) {
            return;
      }

      print_tree(node->right, level + 1);

      for (int i = 0; i < level * SPACE; i++) {
            printf(" ");
      }
      printf("%u\n", node->value);

      print_tree(node->left, level + 1);
}

// Same as print_tree, but stops after max_level levels
void print_tree_max_levels(Node* node, int level, int max_level) {
      if (node == NULL) {
            return;
      }

      if (level >= max_level) {
            return;
      }

      print_tree_max_levels(node->right, level + 1, max_level);

      for (int i = 0; i < level * SPACE; i++) {
            printf(" ");
      }
      printf("%u\n", node->value);

      print_tree_max_levels(node->left, level + 1, max_level);
}
