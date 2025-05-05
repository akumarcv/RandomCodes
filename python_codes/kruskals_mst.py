"""
Kruskal's algorithm for finding the minimum spanning tree of a graph.
Exmaple usage with an adjacency matrix.
"""

def kruskals_mst(graph):
    """
    Kruskal's algorithm to find the minimum spanning tree (MST) of a graph.

    Args:
        graph: Adjacency matrix representing the graph.

    Returns:
        list: Edges in the MST.
    """
    num_vertices = len(graph)
    edges = []
    
    # Create a list of edges with weights
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if graph[i][j] != 0:
                edges.append((graph[i][j], i, j))
    
    # Sort edges based on weight
    edges.sort(key=lambda x: x[0])
    
    parent = list(range(num_vertices))  # Union-Find structure

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(u, v):
        parent[find(u)] = find(v)

    mst_edges = []
    
    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst_edges.append((u, v, weight))

    return mst_edges

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

    mst_edges = kruskals_mst(graph)
    print("Edges in the Minimum Spanning Tree:")
    for edge in mst_edges:
        print(edge)