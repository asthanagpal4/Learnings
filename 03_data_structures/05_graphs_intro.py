# HOW TO RUN:
#   uv run python 03_data_structures/05_graphs_intro.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- GRAPHS ---
# A graph is a collection of "nodes" (also called "vertices")
# connected by "edges" (lines between them).
#
# Unlike a tree, a graph can have cycles (loops) and any node
# can connect to any other node -- there's no single "root".
#
# Think of it like a map of cities connected by roads:
#   - Cities = nodes
#   - Roads = edges
#   - Some roads are one-way (directed graph)
#   - Some roads go both ways (undirected graph)
#
# Why does this matter?
# - Social networks (who is friends with whom)
# - Google Maps (finding shortest routes)
# - Recommendation systems (connecting users to products)
# - Neural networks in ML are essentially computation graphs
# - Knowledge graphs power modern AI systems


# === CONCEPT 1: Representing a graph with a dictionary ===
# The easiest way to represent a graph in Python is with a
# dictionary where each key is a node, and its value is a list
# of neighbors (called an "adjacency list").

print("=" * 50)
print("CONCEPT 1: Graph as a dictionary (adjacency list)")
print("=" * 50)

# Let's represent this friendship graph:
#
#   Astha --- Raj --- Sneha
#     |       |
#   Priya --- Amit
#
# Each connection goes both ways (undirected graph)

friends = {
    "Astha": ["Raj", "Priya"],
    "Raj":   ["Astha", "Sneha", "Amit"],
    "Priya": ["Astha", "Amit"],
    "Amit":  ["Raj", "Priya"],
    "Sneha": ["Raj"],
}

print("Friendship graph:")
for person, connections in friends.items():
    print(f"  {person} is friends with: {connections}")
# Output:
#   Astha is friends with: ['Raj', 'Priya']
#   Raj is friends with: ['Astha', 'Sneha', 'Amit']
#   ...

print()


# === CONCEPT 2: Graph class ===

print("=" * 50)
print("CONCEPT 2: Graph class with add/remove")
print("=" * 50)

class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_node(self, node):
        """Add a node to the graph."""
        if node not in self.adjacency_list:
            self.adjacency_list[node] = []

    def add_edge(self, node1, node2):
        """Add an undirected edge between two nodes."""
        # Make sure both nodes exist
        self.add_node(node1)
        self.add_node(node2)
        # Add connection both ways
        if node2 not in self.adjacency_list[node1]:
            self.adjacency_list[node1].append(node2)
        if node1 not in self.adjacency_list[node2]:
            self.adjacency_list[node2].append(node1)

    def remove_edge(self, node1, node2):
        """Remove the edge between two nodes."""
        if node1 in self.adjacency_list and node2 in self.adjacency_list[node1]:
            self.adjacency_list[node1].remove(node2)
        if node2 in self.adjacency_list and node1 in self.adjacency_list[node2]:
            self.adjacency_list[node2].remove(node1)

    def get_neighbors(self, node):
        """Get all nodes connected to the given node."""
        return self.adjacency_list.get(node, [])

    def display(self):
        """Print the graph."""
        for node, neighbors in self.adjacency_list.items():
            print(f"  {node} -> {neighbors}")


# Build the friendship graph using the class
g = Graph()
g.add_edge("Astha", "Raj")
g.add_edge("Astha", "Priya")
g.add_edge("Raj", "Sneha")
g.add_edge("Raj", "Amit")
g.add_edge("Priya", "Amit")

print("Graph:")
g.display()
print(f"\nAstha's friends: {g.get_neighbors('Astha')}")
# Output: Astha's friends: ['Raj', 'Priya']

print()


# === CONCEPT 3: BFS (Breadth-First Search) ===
# BFS explores the graph level by level, like ripples in water.
# It uses a QUEUE. It finds the SHORTEST path in unweighted graphs.
#
# Think of it like asking:
#   "Who are my direct friends?"
#   "Who are friends of my friends?"
#   "Who are friends of friends of friends?"
#   ...and so on.

print("=" * 50)
print("CONCEPT 3: BFS - Breadth-First Search")
print("=" * 50)

from collections import deque

def bfs(graph, start):
    """
    Visit all nodes reachable from 'start' using BFS.
    Returns the order in which nodes were visited.
    """
    visited = set()       # keep track of nodes we've already seen
    queue = deque()        # queue for nodes to visit next
    order = []             # the order we visit nodes

    visited.add(start)
    queue.append(start)

    while queue:
        current = queue.popleft()  # take from the front
        order.append(current)

        for neighbor in graph.adjacency_list[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


print("BFS starting from Astha:")
print(" ", bfs(g, "Astha"))
# Output: ['Astha', 'Raj', 'Priya', 'Sneha', 'Amit']
# Level 0: Astha
# Level 1: Raj, Priya (Astha's direct friends)
# Level 2: Sneha, Amit (friends of friends)

print()


# === CONCEPT 4: DFS (Depth-First Search) ===
# DFS explores as deep as possible before backtracking.
# It uses a STACK (or recursion, which uses the call stack).
#
# Think of it like exploring a maze: go as far as you can down
# one path, hit a dead end, backtrack, try the next path.

print("=" * 50)
print("CONCEPT 4: DFS - Depth-First Search")
print("=" * 50)

def dfs(graph, start):
    """
    Visit all nodes reachable from 'start' using DFS.
    Returns the order in which nodes were visited.
    """
    visited = set()
    stack = [start]        # use a list as a stack
    order = []

    while stack:
        current = stack.pop()  # take from the top (LIFO)

        if current not in visited:
            visited.add(current)
            order.append(current)

            # Add neighbors to stack (reversed so we visit in order)
            for neighbor in reversed(graph.adjacency_list[current]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order


# DFS using recursion (another way to write it)
def dfs_recursive(graph, node, visited=None):
    """DFS using recursion (the call stack IS our stack)."""
    if visited is None:
        visited = set()

    visited.add(node)
    result = [node]

    for neighbor in graph.adjacency_list[node]:
        if neighbor not in visited:
            result += dfs_recursive(graph, neighbor, visited)

    return result


print("DFS (iterative) from Astha:")
print(" ", dfs(g, "Astha"))
# Output: ['Astha', 'Raj', 'Sneha', 'Amit', 'Priya']

print("DFS (recursive) from Astha:")
print(" ", dfs_recursive(g, "Astha"))
# Output: ['Astha', 'Raj', 'Sneha', 'Amit', 'Priya']

print()


# === CONCEPT 5: BFS vs DFS comparison ===

print("=" * 50)
print("CONCEPT 5: BFS vs DFS")
print("=" * 50)

print("""
  BFS (Breadth-First)              DFS (Depth-First)
  ----------------------------     ----------------------------
  Uses a QUEUE                     Uses a STACK (or recursion)
  Explores level by level          Explores one path deeply
  Finds SHORTEST path              Does NOT guarantee shortest
  Uses more memory                 Uses less memory
  Good for: shortest path,         Good for: maze solving,
    closest friends,                 checking if path exists,
    level-by-level analysis          topological sorting

  In ML:
  - BFS-like: exploring hyperparameter space layer by layer
  - DFS-like: backpropagation through neural network layers
""")


# === CONCEPT 6: Finding a path between two nodes ===

print("=" * 50)
print("CONCEPT 6: Finding a path (BFS)")
print("=" * 50)

def find_path_bfs(graph, start, end):
    """Find the shortest path between start and end using BFS."""
    if start == end:
        return [start]

    visited = set()
    queue = deque()

    # Instead of just the node, store the entire path so far
    queue.append([start])
    visited.add(start)

    while queue:
        path = queue.popleft()
        current = path[-1]  # last node in the path

        for neighbor in graph.adjacency_list[current]:
            if neighbor not in visited:
                new_path = path + [neighbor]

                if neighbor == end:
                    return new_path  # found it!

                visited.add(neighbor)
                queue.append(new_path)

    return None  # no path exists


# Find path from Sneha to Priya
path = find_path_bfs(g, "Sneha", "Priya")
print(f"Shortest path from Sneha to Priya: {path}")
# Output: Shortest path from Sneha to Priya: ['Sneha', 'Raj', 'Astha', 'Priya']

# Another path
path2 = find_path_bfs(g, "Astha", "Sneha")
print(f"Shortest path from Astha to Sneha: {path2}")
# Output: Shortest path from Astha to Sneha: ['Astha', 'Raj', 'Sneha']

print()


# === CONCEPT 7: Checking if graph is connected ===

print("=" * 50)
print("CONCEPT 7: Is the graph connected?")
print("=" * 50)

def is_connected(graph):
    """Check if all nodes can reach each other."""
    if not graph.adjacency_list:
        return True

    start = next(iter(graph.adjacency_list))  # pick any node
    visited = bfs(graph, start)
    return len(visited) == len(graph.adjacency_list)


print(f"Is our friendship graph connected? {is_connected(g)}")
# Output: Is our friendship graph connected? True

# Add a lonely node
g.add_node("Lonely Lee")
print(f"After adding disconnected node: {is_connected(g)}")
# Output: After adding disconnected node: False

# Remove it so it doesn't affect exercises
del g.adjacency_list["Lonely Lee"]

print()


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# The graph used throughout this file:
#   Astha --- Raj --- Sneha
#     |       |
#   Priya --- Amit

# Test Graph: add_node / add_edge / get_neighbors
_g = Graph()
_g.add_edge("A", "B")
_g.add_edge("A", "C")
_g.add_edge("B", "D")
assert "B" in _g.get_neighbors("A")
assert "C" in _g.get_neighbors("A")
assert "A" in _g.get_neighbors("B")    # undirected: both ways
assert _g.get_neighbors("D") == ["B"]

# Test that add_edge doesn't add duplicate edges
_g.add_edge("A", "B")   # adding again
assert _g.get_neighbors("A").count("B") == 1   # still only one entry

# Test BFS: visits all reachable nodes, breadth-first order
_bfs_result = bfs(g, "Astha")
assert "Astha" in _bfs_result
assert "Raj" in _bfs_result
assert "Priya" in _bfs_result
assert "Sneha" in _bfs_result
assert "Amit" in _bfs_result
assert len(_bfs_result) == 5            # all 5 people visited
assert _bfs_result[0] == "Astha"        # start node is first
# Raj and Priya are direct friends (level 1) -- they come before Sneha/Amit
assert _bfs_result.index("Raj") < _bfs_result.index("Sneha")

# Test DFS: visits all reachable nodes
_dfs_result = dfs(g, "Astha")
assert set(_dfs_result) == {"Astha", "Raj", "Priya", "Sneha", "Amit"}
assert _dfs_result[0] == "Astha"

# Test find_path_bfs: shortest path
assert find_path_bfs(g, "Astha", "Astha") == ["Astha"]   # same node
_path = find_path_bfs(g, "Astha", "Sneha")
assert _path[0] == "Astha"
assert _path[-1] == "Sneha"
assert len(_path) == 3   # Astha -> Raj -> Sneha (shortest)

# Test is_connected
assert is_connected(g) == True
_isolated = Graph()
_isolated.add_edge("X", "Y")
_isolated.add_node("Z")     # Z is not connected to X or Y
assert is_connected(_isolated) == False

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Degree of separation
#    Write a function degrees_of_separation(graph, person1, person2)
#    that returns how many "hops" apart two people are.
#    Example: Astha -> Raj -> Sneha = 2 degrees of separation
#    Hint: find the shortest path and return len(path) - 1.

# Your code here:


# 2. Find all paths
#    Write a function find_all_paths(graph, start, end) that returns
#    ALL possible paths (not just the shortest) between two nodes.
#    Hint: use DFS with backtracking. Keep track of the current path
#    and "un-visit" nodes when you backtrack.

# Your code here:


# 3. Directed graph
#    Modify the Graph class to support directed edges (one-way roads).
#    add_edge should only add the connection in one direction.
#    Then build a small graph and test BFS/DFS on it.
#    Hint: in add_edge, only add node2 to node1's list (not both ways).

# Your code here:
