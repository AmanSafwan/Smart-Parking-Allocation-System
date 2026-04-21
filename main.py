from collections import deque
import heapq
import time

# =========================
# GRID CONFIGURATION
# =========================

# Define the search environment as a 2D grid
# This grid is used as the problem space for all search algorithms
grid = [
    ['S', '.', '.', 'X', '.', 'P', 'X'],
    ['.', 'X', 'X', 'X', '.', 'P', 'P'],
    ['.', '.', '.', '.', '.', '.', '.'],
    ['X', 'X', '.', 'X', '.', 'P', '.'],
    ['X', 'X', '.', 'X', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.'],
    ['.', 'X', 'X', 'X', 'X', 'P', 'X'],
]

# Get number of rows and columns dynamically from grid
# This is used to ensure movement checks stay within bounds
ROWS, COLS = len(grid), len(grid[0])

# Define starting position of the agent
# This is the initial node where all algorithms begin searching
start = (0, 0)

# Define multiple goal states (parking locations)
# These represent possible target destinations in the search space
goals = {(0, 5), (1, 5), (1, 6), (3, 5), (6, 5)}

# Define possible movement directions (right, left, down, up)
# This restricts movement to 4-directional grid traversal
moves = [(0,1), (0,-1), (1,0), (-1,0)]

# =========================
# HELPER FUNCTIONS
# =========================

def is_valid(x, y):
    # Check if the next position is inside grid boundaries
    # Prevents index error when moving outside grid
    if not (0 <= x < ROWS and 0 <= y < COLS):
        return False

    # Check if the cell is not an obstacle
    # 'X' represents blocked path that cannot be traversed
    if grid[x][y] == 'X':
        return False

    return True


def heuristic(node):
    # Heuristic function estimates distance from current node to goal
    # Used to guide informed search algorithms (Greedy & A*)
    
    # Compute Manhattan distance to all goals
    # Then choose the minimum (closest goal)
    return min(
        abs(node[0] - g[0]) + abs(node[1] - g[1])
        for g in goals
    )

# =========================
# VISUALIZATION FUNCTION
# =========================

def draw_path(path, title):
    # Convert path list into set for fast lookup
    # This is used to mark visited path on grid
    path_set = set(path)

    # Print header for visualization output
    print("\n" + "="*60)
    print(f"{title} - PATH VISUALIZATION")
    print("="*60)

    # Loop through each cell in grid
    for i in range(ROWS):
        row = []
        for j in range(COLS):

            # Mark start position clearly
            if (i, j) == start:
                row.append('S')

            # Mark goal positions clearly
            elif (i, j) in goals:
                row.append('P')

            # Mark path taken by algorithm
            elif (i, j) in path_set:
                row.append('*')

            # Otherwise show original grid value
            else:
                row.append(grid[i][j])

        print(" ".join(row))

# =========================
# BFS (UNINFORMED SEARCH)
# =========================

def bfs(start):
    # Breadth-First Search:
    # Explores all nodes level by level before going deeper
    # Guarantees shortest path in unweighted grid

    t0 = time.perf_counter()  # Start timer for performance measurement

    # Queue stores (current node, path taken so far)
    q = deque([(start, [start])])

    # Track visited nodes to avoid revisiting same state
    visited = {start}

    expanded = 0  # Count number of nodes processed

    while q:
        current, path = q.popleft()  # FIFO behavior (BFS logic)
        expanded += 1

        # If goal reached, return result immediately
        if current in goals:
            return path, expanded, time.perf_counter() - t0

        # Explore all possible movements
        for dx, dy in moves:
            nxt = (current[0] + dx, current[1] + dy)

            # Only process valid and unvisited nodes
            if is_valid(*nxt) and nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [nxt]))

    return None, expanded, time.perf_counter() - t0

# =========================
# DFS (UNINFORMED SEARCH)
# =========================

def dfs(start):
    # Depth-First Search:
    # Explores deep paths first before backtracking
    # Does NOT guarantee shortest path

    t0 = time.perf_counter()

    # Stack used for LIFO behavior (DFS logic)
    stack = [(start, [start])]

    visited = set()
    expanded = 0

    while stack:
        current, path = stack.pop()

        # Skip if already visited
        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        # Check if goal reached
        if current in goals:
            return path, expanded, time.perf_counter() - t0

        # Reverse moves to control exploration order
        for dx, dy in reversed(moves):
            nxt = (current[0] + dx, current[1] + dy)

            if is_valid(*nxt) and nxt not in visited:
                stack.append((nxt, path + [nxt]))

    return None, expanded, time.perf_counter() - t0

# =========================
# GREEDY BEST-FIRST SEARCH
# =========================

def greedy(start):
    # Greedy Search:
    # Chooses node that looks closest to goal (based on heuristic only)
    # Fast but may not produce optimal path

    t0 = time.perf_counter()

    # Priority queue sorted by heuristic value
    pq = [(heuristic(start), 0, start, [start])]

    visited = set()
    counter = 0
    expanded = 0

    while pq:
        _, _, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current in goals:
            return path, expanded, time.perf_counter() - t0

        for dx, dy in moves:
            nxt = (current[0] + dx, current[1] + dy)

            if is_valid(*nxt) and nxt not in visited:
                counter += 1
                heapq.heappush(
                    pq,
                    (heuristic(nxt), counter, nxt, path + [nxt])
                )

    return None, expanded, time.perf_counter() - t0

# =========================
# A* SEARCH (INFORMED SEARCH)
# =========================

def astar(start):
    # A* Search:
    # Uses both actual cost (g) + heuristic (h)
    # f(n) = g(n) + h(n)
    # Guarantees optimal solution if heuristic is admissible

    t0 = time.perf_counter()

    pq = [(0, 0, start, [start], 0)]  # (f, counter, node, path, g-cost)

    best = {start: 0}  # Store best cost found for each node
    counter = 0
    expanded = 0

    while pq:
        f, _, current, path, g = heapq.heappop(pq)

        # Skip if we already found better path before
        if g > best.get(current, float('inf')):
            continue

        expanded += 1

        if current in goals:
            return path, expanded, time.perf_counter() - t0

        for dx, dy in moves:
            nxt = (current[0] + dx, current[1] + dy)

            if is_valid(*nxt):
                new_g = g + 1  # Increment cost by 1 step

                # Only update if this path is better
                if new_g < best.get(nxt, float('inf')):
                    best[nxt] = new_g
                    counter += 1

                    heapq.heappush(
                        pq,
                        (new_g + heuristic(nxt), counter, nxt, path + [nxt], new_g)
                    )

    return None, expanded, time.perf_counter() - t0

# =========================
# RESULT HANDLER
# =========================

def run_algo(name, func):
    # Execute algorithm and collect performance metrics
    # This function standardizes output for all algorithms

    path, expanded, t = func(start)

    print("\n" + "="*60)
    print(name)
    print("="*60)
    print("Steps          :", len(path)-1 if path else "N/A")
    print("Nodes Expanded :", expanded)
    print("Time (s)       :", round(t, 6))
    print("Goal Reached   :", path[-1] if path else "None")

    draw_path(path, name)

    return path, expanded, t

# =========================
# EXECUTION
# =========================

print("\n🚗 AI SMART PARKING SYSTEM - FULL COMPARISON 🚗\n")

bfs_r = run_algo("BFS (Uninformed)", bfs)
dfs_r = run_algo("DFS (Uninformed)", dfs)
greedy_r = run_algo("Greedy Best-First", greedy)
astar_r = run_algo("A* Search", astar)

# =========================
# SUMMARY TABLE
# =========================

results = {
    "BFS": bfs_r,
    "DFS": dfs_r,
    "GREEDY": greedy_r,
    "A*": astar_r,
}

print("\n================ SUMMARY ================")
print("Algo     Steps   Expanded   Time(s)")
print("--------------------------------------")

for k, v in results.items():
    steps = len(v[0]) - 1 if v[0] else "N/A"
    print(f"{k:<8} {steps:<7} {v[1]:<9} {round(v[2],6)}")