from collections import deque
import heapq

def beam_search(graph, start, goal, bw, heuristic):
    visited = []

    queue = []
    queue.append([heuristic(start), [start]])
    
    while queue:
        _, path = heapq.heappop(queue)
        node = path[-1]
        if node==goal:
            return path
        visited.append(node)

        for v, cost in graph[node]:
            if v not in visited:
                path.append(v)
                value_heuristic = heuristic(v)
                heapq.heappush(queue, [value_heuristic, path])
        heapq.heapify(queue)
        if len(queue) > bw:
            heapq.nlargest(bw, queue) # Keep only the top k
            queue = queue[:bw]

    return None


# Define a sample graph
graph = {
    'A': [('B', 2), ('C', 4)],
    'B': [('D', 3), ('E', 1)],
    'C': [('F', 5)],
    'D': [('G', 2)],
    'E': [('G', 6)],
    'F': [('G', 1)],
    'G': []
}

# Define a heuristic function (e.g., distance to goal)
def heuristic(node):
    if node == 'G':
        return 0
    else:
        return 10 #Placeholder heuristic

# Set parameters
start_node = 'A'
goal_node = 'G'
beam_width = 2

# Perform beam search
path = beam_search(graph, start_node, goal_node, beam_width, heuristic)

# Print the path
if path:
    print("Path:", path)
else:
    print("No path found.")