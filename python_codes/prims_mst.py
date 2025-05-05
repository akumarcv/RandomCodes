"""
prims mst with an example, hashmap, and time complexity
Prim's algorithm is a greedy algorithm that finds a minimum 
spanning tree (MST) for a weighted undirected graph. 
The MST is a subset of the edges that connects all vertices 
with the minimum total edge weight.
"""

def prims_mst(graph):
    """
    Prim's algorithm to find the minimum spanning tree (MST) of a graph.

    Args:
        graph: Adjacency matrix representing the graph.

    Returns:
        list: Edges in the MST.
    """
    num_vertices = len(graph)
    selected = [False] * num_vertices  # Track selected vertices
    edges = []  # List to store edges in the MST
    min_edge = {i: float('inf') for i in range(num_vertices)}  # Initialize min edge weights
    min_edge[0] = 0  # Start from vertex 0
    parent = {i: None for i in range(num_vertices)}  # Track parent vertices

    for _ in range(num_vertices):
        # Find the vertex with the minimum edge weight
        u = min((v for v in range(num_vertices) if not selected[v]), key=lambda v: min_edge[v])
        selected[u] = True  # Mark vertex as selected

        # Update edges and parent for the MST
        if parent[u] is not None:
            edges.append((parent[u], u, graph[parent[u]][u]))  # Add edge to MST

        for v in range(num_vertices):
            if graph[u][v] and not selected[v] and graph[u][v] < min_edge[v]:
                min_edge[v] = graph[u][v]
                parent[v] = u

    return edges
# Example usage
if __name__ == "__main__":
    # Example graph represented as an adjacency matrix
    graph = [
        [0, 2, 0, 6, 0],
        [2, 0, 3, 8, 5],
        [0, 3, 0, 0, 7],
        [6, 8, 0, 0, 9],
        [0, 5, 7, 9, 0]
    ]

    mst_edges = prims_mst(graph)
    print("Edges in the Minimum Spanning Tree:")
    for edge in mst_edges:
        print(edge)
