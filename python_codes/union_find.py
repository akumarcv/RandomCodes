"""
Union-Find (Disjoint Set) data structure implementation with path compression and union by rank.
This module provides functionality to detect cycles in undirected graphs and find redundant connections.
"""

from typing import List


class UnionFind:
    """
    Union-Find (Disjoint Set) data structure implementation.
    
    This class implements the Union-Find data structure with path compression
    and union by rank optimizations for efficient operations.
    
    Attributes:
        parent (List[int]): List where parent[i] is the parent of element i.
        rank (List[int]): List where rank[i] is the approximate depth of the tree rooted at element i.
    """
    
    def __init__(self, n: int):
        """
        Initialize the Union-Find data structure with n elements.
        
        Args:
            n (int): Number of elements in the disjoint set.
        """
        # Initialize each element as its own parent
        self.parent = list(range(n))
        # Initialize rank of each element to 1
        self.rank = [1] * n
    
    def find(self, x: int) -> int:
        """
        Find the representative (root) of the set containing element x.
        Uses path compression to optimize future queries.
        
        Args:
            x (int): The element to find the representative for.
            
        Returns:
            int: The representative (root) of the set containing x.
        """
        # If x is not its own parent, recursively find the root
        if self.parent[x] != x:
            # Path compression: make the parent of x point directly to the root
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Merge the sets containing elements x and y.
        Uses union by rank to keep the tree balanced.
        
        Args:
            x (int): First element.
            y (int): Second element.
            
        Returns:
            bool: True if x and y were in different sets and were merged,
                  False if they were already in the same set.
        """
        # Find representatives (roots) of the sets containing x and y
        p1, p2 = self.find(x), self.find(y)
        
        # If x and y are already in the same set, return False
        if p1 == p2:
            return False
            
        # Union by rank: attach the smaller rank tree under the root of the higher rank tree
        if self.rank[p1] > self.rank[p2]:
            # Make p1 the parent of p2
            self.parent[p2] = p1
            # Update the rank of p1
            self.rank[p1] += self.rank[p2]
        else:
            # Make p2 the parent of p1
            self.parent[p1] = p2
            # Update the rank of p2
            self.rank[p2] += self.rank[p1]

        return True
    

def redundant_connection(edges: List[List[int]]) -> List[int]:
    """
    Find the redundant connection in an undirected graph.
    
    In a graph that started as a tree, exactly one edge is redundant.
    This function finds and returns that redundant edge.
    
    Args:
        edges (List[List[int]]): List of edges where each edge is represented as [u, v].
                                Node numbers are 1-indexed.
                                
    Returns:
        List[int]: The redundant edge [u, v] that forms a cycle.
                   Returns empty list if no redundant edge is found.
    """
    # Create a Union-Find data structure with size equal to number of nodes + 1
    # (since node numbers are 1-indexed)
    uf = UnionFind(len(edges) + 1)
    
    # Process each edge
    for u, v in edges:
        # If union operation returns False, it means u and v are already connected,
        # making this edge redundant (it forms a cycle)
        if not uf.union(u, v):
            return [u, v]
            
    # If no redundant edge is found (which shouldn't happen in a valid input)
    return []


def main():
    """
    Main function to test the redundant_connection function with sample inputs.
    """
    edges = [
        [[1, 2], [1, 3], [2, 3]], 
        [[1, 2], [2, 3], [1, 3]], 
        [[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]], 
        [[1, 2], [1, 3], [1, 4], [3, 4], [2, 4]], 
        [[1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]
    ]

    for i in range(len(edges)):
        print(i + 1, ".\tEdges: ", edges[i], sep="")
        print("\tThe redundant connection in the graph is: ", redundant_connection(edges[i]), sep="")
        print("-" * 100)


if __name__ == '__main__':
    main()