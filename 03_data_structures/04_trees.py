# HOW TO RUN:
#   uv run python 03_data_structures/04_trees.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- BINARY TREES ---
# A tree is a data structure that looks like an upside-down tree:
# one "root" at the top, with branches going downward.
#
# A BINARY tree means each node has at most 2 children:
# a "left" child and a "right" child.
#
#         10           <-- root
#        /  \
#       5    15         <-- children of 10
#      / \     \
#     3   7     20      <-- leaves (no children)
#
# Why does this matter?
# - Decision trees in ML are literally trees!
# - File systems are organized as trees (folders inside folders)
# - HTML/XML documents are trees
# - Efficient searching and sorting uses trees
# - Understanding tree traversal helps with many algorithms


# === CONCEPT 1: Tree Node ===
# Each node in a binary tree has: a value, a left child, a right child.

print("=" * 50)
print("CONCEPT 1: Creating a Binary Tree")
print("=" * 50)

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None   # left child
        self.right = None  # right child

    def __str__(self):
        return f"TreeNode({self.value})"


# Build this tree:
#       10
#      /  \
#     5    15
#    / \     \
#   3   7    20

root = TreeNode(10)
root.left = TreeNode(5)
root.right = TreeNode(15)
root.left.left = TreeNode(3)
root.left.right = TreeNode(7)
root.right.right = TreeNode(20)

print(f"Root: {root}")
print(f"Root's left child: {root.left}")
print(f"Root's right child: {root.right}")
print(f"Leftmost leaf: {root.left.left}")
# Output:
#   Root: TreeNode(10)
#   Root's left child: TreeNode(5)
#   Root's right child: TreeNode(15)
#   Leftmost leaf: TreeNode(3)

print()


# === CONCEPT 2: Tree Traversals ===
# "Traversal" means visiting every node in the tree. There are
# three main ways to do it. The difference is WHEN you process
# the current node relative to its children.

print("=" * 50)
print("CONCEPT 2: Three ways to traverse a tree")
print("=" * 50)

# --- Inorder: Left, Current, Right ---
# Visit left subtree first, then current node, then right subtree.
# For a binary search tree, this gives values in sorted order!

def inorder(node):
    """Visit: left subtree -> current node -> right subtree."""
    if node is None:
        return []

    result = []
    result += inorder(node.left)     # visit left subtree
    result.append(node.value)        # visit current node
    result += inorder(node.right)    # visit right subtree
    return result


# --- Preorder: Current, Left, Right ---
# Visit the current node first, then left, then right.
# Useful for copying a tree or creating a "prefix" expression.

def preorder(node):
    """Visit: current node -> left subtree -> right subtree."""
    if node is None:
        return []

    result = []
    result.append(node.value)        # visit current node
    result += preorder(node.left)    # visit left subtree
    result += preorder(node.right)   # visit right subtree
    return result


# --- Postorder: Left, Right, Current ---
# Visit children first, then the current node.
# Useful for deleting a tree or evaluating expressions.

def postorder(node):
    """Visit: left subtree -> right subtree -> current node."""
    if node is None:
        return []

    result = []
    result += postorder(node.left)    # visit left subtree
    result += postorder(node.right)   # visit right subtree
    result.append(node.value)         # visit current node
    return result


# Using the tree from Concept 1:
#       10
#      /  \
#     5    15
#    / \     \
#   3   7    20

print("Inorder   (Left, Self, Right):", inorder(root))
# Output: Inorder   (Left, Self, Right): [3, 5, 7, 10, 15, 20]

print("Preorder  (Self, Left, Right):", preorder(root))
# Output: Preorder  (Self, Left, Right): [10, 5, 3, 7, 15, 20]

print("Postorder (Left, Right, Self):", postorder(root))
# Output: Postorder (Left, Right, Self): [3, 7, 5, 20, 15, 10]

print()

# A trick to remember:
print("Memory trick:")
print("  IN-order   = current node is IN the middle  (Left, ME, Right)")
print("  PRE-order  = current node comes FIRST/PRE    (ME, Left, Right)")
print("  POST-order = current node comes LAST/POST    (Left, Right, ME)")
print()


# === CONCEPT 3: Level-Order Traversal (BFS) ===
# Visit the tree level by level, from top to bottom.
# This uses a QUEUE (remember queues from the previous file!).

print("=" * 50)
print("CONCEPT 3: Level-order traversal (BFS)")
print("=" * 50)

from collections import deque

def level_order(node):
    """Visit the tree level by level using a queue."""
    if node is None:
        return []

    result = []
    queue = deque()
    queue.append(node)

    while queue:
        current = queue.popleft()  # dequeue
        result.append(current.value)

        if current.left:
            queue.append(current.left)   # enqueue left child
        if current.right:
            queue.append(current.right)  # enqueue right child

    return result


print("Level-order:", level_order(root))
# Output: Level-order: [10, 5, 15, 3, 7, 20]
# That's: level 0 = [10], level 1 = [5, 15], level 2 = [3, 7, 20]

print()


# === CONCEPT 4: Binary Tree class with useful methods ===

print("=" * 50)
print("CONCEPT 4: BinaryTree class")
print("=" * 50)

class BinaryTree:
    def __init__(self, root_value=None):
        if root_value is not None:
            self.root = TreeNode(root_value)
        else:
            self.root = None

    def height(self, node=None):
        """Find the height (number of levels) of the tree."""
        if node is None:
            return 0
        left_height = self.height(node.left)
        right_height = self.height(node.right)
        return 1 + max(left_height, right_height)

    def count_nodes(self, node=None):
        """Count total number of nodes."""
        if node is None:
            return 0
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)

    def search(self, node, target):
        """Search for a value in the tree."""
        if node is None:
            return False
        if node.value == target:
            return True
        return self.search(node.left, target) or self.search(node.right, target)

    def display(self, node=None, level=0, prefix="Root: "):
        """Print a visual representation of the tree."""
        if node is None and level == 0:
            node = self.root
        if node is not None:
            print(" " * (level * 4) + prefix + str(node.value))
            if node.left is not None or node.right is not None:
                if node.left:
                    self.display(node.left, level + 1, "L--- ")
                else:
                    print(" " * ((level + 1) * 4) + "L--- (empty)")
                if node.right:
                    self.display(node.right, level + 1, "R--- ")
                else:
                    print(" " * ((level + 1) * 4) + "R--- (empty)")


# Build the same tree using the class
tree = BinaryTree(10)
tree.root.left = TreeNode(5)
tree.root.right = TreeNode(15)
tree.root.left.left = TreeNode(3)
tree.root.left.right = TreeNode(7)
tree.root.right.right = TreeNode(20)

print("Tree structure:")
tree.display()
# Output:
#   Root: 10
#       L--- 5
#           L--- 3
#           R--- 7
#       R--- 15
#           L--- (empty)
#           R--- 20

print(f"\nHeight: {tree.height(tree.root)}")
# Output: Height: 3

print(f"Total nodes: {tree.count_nodes(tree.root)}")
# Output: Total nodes: 6

print(f"Search for 7: {tree.search(tree.root, 7)}")
# Output: Search for 7: True

print(f"Search for 99: {tree.search(tree.root, 99)}")
# Output: Search for 99: False

print()


# === CONCEPT 5: Why trees matter for ML ===

print("=" * 50)
print("CONCEPT 5: Trees in Machine Learning")
print("=" * 50)

print("""
  Trees are everywhere in ML:

  1. Decision Trees
     - A model that makes predictions by asking yes/no questions
     - Example: "Is age > 30?" -> Yes -> "Is income > 50k?" -> ...
     - Each question is a node, each answer is a branch

  2. Random Forests
     - A collection of many decision trees that "vote" on the answer

  3. Gradient Boosted Trees (XGBoost, LightGBM)
     - Some of the most powerful ML models for tabular data

  4. Parse Trees
     - Used in Natural Language Processing (NLP) to understand
       sentence structure

  5. Abstract Syntax Trees
     - How Python understands your code internally

  Understanding tree traversal now will make learning these
  ML concepts much easier later!
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# The tree used throughout this file:
#       10
#      /  \
#     5    15
#    / \     \
#   3   7    20

# Test inorder traversal (should be sorted for a BST)
assert inorder(root) == [3, 5, 7, 10, 15, 20]

# Test preorder traversal (root first)
assert preorder(root) == [10, 5, 3, 7, 15, 20]

# Test postorder traversal (root last)
assert postorder(root) == [3, 7, 5, 20, 15, 10]

# Test level_order traversal (level by level)
assert level_order(root) == [10, 5, 15, 3, 7, 20]

# Edge case: empty tree
assert inorder(None) == []
assert preorder(None) == []
assert postorder(None) == []
assert level_order(None) == []

# Test BinaryTree class methods
assert tree.height(tree.root) == 3
assert tree.count_nodes(tree.root) == 6
assert tree.search(tree.root, 7) == True
assert tree.search(tree.root, 99) == False
assert tree.search(tree.root, 10) == True   # root itself

# Edge case: single-node tree
_single = BinaryTree(42)
assert _single.height(_single.root) == 1
assert _single.count_nodes(_single.root) == 1
assert _single.search(_single.root, 42) == True
assert _single.search(_single.root, 0) == False

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Count leaves
#    Write a function count_leaves(node) that counts how many
#    leaf nodes (nodes with NO children) are in a tree.
#    For our example tree, the answer is 3 (nodes 3, 7, and 20).
#    Hint: a leaf is a node where both left and right are None.

# Your code here:


# 2. Find maximum value
#    Write a function find_max(node) that returns the largest
#    value in the tree.
#    For our tree, the answer is 20.
#    Hint: compare the current value with the max of left and right.

# Your code here:


# 3. Mirror a tree
#    Write a function mirror(node) that swaps left and right
#    children for every node, creating a mirror image.
#    Original:     10          Mirrored:     10
#                 /  \                      /  \
#                5    15                  15    5
#    Hint: swap left and right, then recursively mirror children.

# Your code here:
