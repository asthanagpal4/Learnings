# HOW TO RUN:
#   uv run python 03_data_structures/project_maze_solver.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- MINI-PROJECT: MAZE SOLVER ---
# This program creates a maze, converts it to a graph, and then
# solves it using BFS and DFS. This brings together everything
# from this section: graphs, queues, stacks, and algorithmic thinking.
#
# The maze is a grid where:
#   '.' = open path (you can walk here)
#   '#' = wall (you can't walk here)
#   'S' = start
#   'E' = end
#
# Our job: find a path from S to E.


from collections import deque


# === STEP 1: Define the maze as a grid ===

MAZE = [
    "S . . # . . .",
    "# # . # . # .",
    ". . . . . # .",
    ". # # # . . .",
    ". . . # # # .",
    "# # . . . . .",
    ". . . # # . E",
]

def parse_maze(maze_strings):
    """Convert the maze from strings to a 2D list."""
    grid = []
    start = None
    end = None

    for row_index, row_string in enumerate(maze_strings):
        row = row_string.split()
        grid.append(row)
        for col_index, cell in enumerate(row):
            if cell == "S":
                start = (row_index, col_index)
            elif cell == "E":
                end = (row_index, col_index)

    return grid, start, end


def print_maze(grid, path=None):
    """Print the maze. If a path is given, mark it with '*'."""
    # Make a copy so we don't modify the original
    display = [row[:] for row in grid]

    if path:
        for row, col in path:
            if display[row][col] not in ("S", "E"):
                display[row][col] = "*"

    print()
    for row in display:
        print("  " + " ".join(row))
    print()


# Parse and display the maze
grid, start, end = parse_maze(MAZE)

print("=" * 50)
print("THE MAZE")
print("=" * 50)
print(f"Start: {start}")
print(f"End:   {end}")
print_maze(grid)


# === STEP 2: Convert the maze grid to a graph ===
# Each walkable cell becomes a node.
# Two adjacent walkable cells are connected by an edge.

def maze_to_graph(grid):
    """Convert a maze grid into a graph (adjacency list)."""
    rows = len(grid)
    cols = len(grid[0])
    graph = {}

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "#":
                continue  # skip walls

            node = (r, c)
            graph[node] = []

            # Check all 4 directions: up, down, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_r, new_c = r + dr, c + dc

                # Check bounds
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    # Check if it's not a wall
                    if grid[new_r][new_c] != "#":
                        graph[node].append((new_r, new_c))

    return graph


graph = maze_to_graph(grid)
print("=" * 50)
print("MAZE AS A GRAPH (first 5 nodes)")
print("=" * 50)
for i, (node, neighbors) in enumerate(graph.items()):
    if i >= 5:
        print("  ...")
        break
    print(f"  {node} -> {neighbors}")
print(f"  Total nodes: {len(graph)}")
print()


# === STEP 3: Solve with BFS (finds shortest path) ===

def solve_bfs(graph, start, end):
    """
    Find the shortest path from start to end using BFS.
    Returns the path as a list of (row, col) positions.
    """
    visited = set()
    queue = deque()

    # Each item in the queue is a path (list of positions)
    queue.append([start])
    visited.add(start)

    while queue:
        path = queue.popleft()
        current = path[-1]

        # Did we reach the end?
        if current == end:
            return path

        # Try all neighbors
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append(new_path)

    return None  # no path found


print("=" * 50)
print("SOLVING WITH BFS (Breadth-First Search)")
print("=" * 50)

bfs_path = solve_bfs(graph, start, end)

if bfs_path:
    print(f"Path found! Length: {len(bfs_path)} steps")
    print(f"Path: {bfs_path}")
    print("\nMaze with BFS path marked with '*':")
    print_maze(grid, bfs_path)
else:
    print("No path found!")

print()


# === STEP 4: Solve with DFS (finds A path, not necessarily shortest) ===

def solve_dfs(graph, start, end):
    """
    Find a path from start to end using DFS.
    Note: this may NOT be the shortest path!
    """
    visited = set()
    stack = [[start]]  # stack of paths

    while stack:
        path = stack.pop()
        current = path[-1]

        if current == end:
            return path

        if current not in visited:
            visited.add(current)

            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    stack.append(new_path)

    return None  # no path found


print("=" * 50)
print("SOLVING WITH DFS (Depth-First Search)")
print("=" * 50)

dfs_path = solve_dfs(graph, start, end)

if dfs_path:
    print(f"Path found! Length: {len(dfs_path)} steps")
    print(f"Path: {dfs_path}")
    print("\nMaze with DFS path marked with '*':")
    print_maze(grid, dfs_path)
else:
    print("No path found!")

print()


# === STEP 5: Compare BFS and DFS ===

print("=" * 50)
print("COMPARISON")
print("=" * 50)

print(f"BFS path length: {len(bfs_path) if bfs_path else 'No path'} steps")
print(f"DFS path length: {len(dfs_path) if dfs_path else 'No path'} steps")
print()
if bfs_path and dfs_path:
    if len(bfs_path) <= len(dfs_path):
        print("BFS found the shorter (or equal) path!")
        print("This is expected -- BFS always finds the shortest path.")
    else:
        print("DFS found a shorter path (unusual but possible with this setup).")
print()


# === STEP 6: A bigger maze to play with ===

print("=" * 50)
print("BONUS: A BIGGER MAZE")
print("=" * 50)

BIG_MAZE = [
    "S . # . . . . . . .",
    "# . # . # # # # . .",
    ". . . . . . . # . #",
    ". # # # # # . # . .",
    ". # . . . . . . . .",
    ". # . # # # # # # .",
    ". . . # . . . . . .",
    "# # . # . # # # . #",
    ". . . . . # . . . .",
    ". # # # . # . # # .",
    ". . . . . . . # . E",
]

big_grid, big_start, big_end = parse_maze(BIG_MAZE)
big_graph = maze_to_graph(big_grid)

print("Big maze:")
print_maze(big_grid)

big_bfs_path = solve_bfs(big_graph, big_start, big_end)
if big_bfs_path:
    print(f"BFS shortest path: {len(big_bfs_path)} steps")
    print_maze(big_grid, big_bfs_path)

big_dfs_path = solve_dfs(big_graph, big_start, big_end)
if big_dfs_path:
    print(f"DFS path: {len(big_dfs_path)} steps")
    print_maze(big_grid, big_dfs_path)

print()


# =========================================================
# TODO: Things for you (Astha) to try!
# =========================================================
#
# 1. Create your own maze!
#    - Change the MAZE variable at the top
#    - Use '#' for walls, '.' for paths, 'S' for start, 'E' for end
#    - Run the program to see if it can be solved
#
# 2. Add diagonal movement
#    - In maze_to_graph(), we only check 4 directions (up/down/left/right)
#    - Add 4 more: (-1,-1), (-1,1), (1,-1), (1,1) for diagonal moves
#    - How does this change the path?
#
# 3. Count steps explored
#    - Modify solve_bfs() and solve_dfs() to also return how many
#      nodes they explored (visited) before finding the answer
#    - Which one explores fewer nodes?
#    Hint: return (path, len(visited)) instead of just path
#
# 4. Dead end detector
#    - Write a function that finds all dead ends in the maze
#    - A dead end is a walkable cell with only 1 neighbor
#    Hint: check len(graph[node]) == 1 for each node
#
# 5. Generate a random maze (challenge!)
#    - Write a function that creates a random maze of size N x M
#    - Make sure there is always a valid path from S to E
#    Hint: start with all walls, then carve paths using DFS
#
# =========================================================

# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test parse_maze: returns correct start and end positions
assert start == (0, 0)
assert end == (6, 6)

# Test maze_to_graph: start node has neighbors, wall nodes are absent
assert start in graph
assert end in graph
# (0,1) should be a neighbor of start (0,0) since it's '.'
assert (0, 1) in graph[start]
# (1,0) is '#' so it should NOT be in the graph at all
assert (1, 0) not in graph

# Test solve_bfs: finds a path
assert bfs_path is not None
assert bfs_path[0] == start
assert bfs_path[-1] == end
# Every step in the path must be adjacent (distance 1 in row or col)
for _i in range(len(bfs_path) - 1):
    _r1, _c1 = bfs_path[_i]
    _r2, _c2 = bfs_path[_i + 1]
    assert abs(_r1 - _r2) + abs(_c1 - _c2) == 1

# Test solve_dfs: also finds a path
assert dfs_path is not None
assert dfs_path[0] == start
assert dfs_path[-1] == end
for _i in range(len(dfs_path) - 1):
    _r1, _c1 = dfs_path[_i]
    _r2, _c2 = dfs_path[_i + 1]
    assert abs(_r1 - _r2) + abs(_c1 - _c2) == 1

# BFS must give a path no longer than DFS (BFS = shortest)
assert len(bfs_path) <= len(dfs_path)

# Test big maze paths as well
assert big_bfs_path is not None
assert big_bfs_path[0] == big_start
assert big_bfs_path[-1] == big_end
assert big_dfs_path is not None
assert len(big_bfs_path) <= len(big_dfs_path)

print("\nAll tests passed!")

print("=" * 50)
print("ALL DONE!")
print("=" * 50)
print("You've seen how to:")
print("  1. Represent a maze as a grid")
print("  2. Convert a grid into a graph")
print("  3. Solve it with BFS (shortest path)")
print("  4. Solve it with DFS (a path, maybe not shortest)")
print()
print("Try the TODOs above to extend this project!")
