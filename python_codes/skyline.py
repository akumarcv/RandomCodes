from calendar import c


class UnionFind:
    """
    Union-Find (Disjoint Set) data structure for efficient merging of sets and finding set representatives.

    This implementation uses path compression in the find operation to optimize lookups.
    Used in the skyline problem to merge building segments with the same height.

    Attributes:
        root (list): Array where root[i] represents the parent of element i
    """

    def __init__(self, N):
        """
        Initialize the Union-Find data structure with N distinct elements.

        Args:
            N (int): Number of elements in the data structure
        """
        self.root = list(range(N))  # Initially, each element is its own root

    def find(self, x):
        """
        Find the root/representative of the set containing element x.
        Uses path compression to optimize future lookups.

        Args:
            x (int): The element to find the root for

        Returns:
            int: The root/representative of the set containing x
        """
        if self.root[x] != x:  # If x is not its own root
            self.root[x] = self.find(self.root[x])  # Path compression
        return self.root[x]

    def union(self, x, y):
        """
        Merge the sets containing elements x and y.

        Args:
            x (int): First element
            y (int): Second element
        """
        self.root[x] = self.root[y]  # Make y's root the parent of x


def get_skyline(buildings):
    """
    Compute the skyline formed by a list of buildings.

    The skyline is represented as a list of [x, height] points where each point marks
    either the start of a new height or the end of the current height in the skyline.

    Algorithm:
    1. Extract and sort unique x-coordinates from all buildings
    2. Process buildings in descending order of height
    3. Use Union-Find to efficiently track processed segments
    4. Build the skyline by identifying height changes

    Time Complexity:
    - O(B log B) for sorting B buildings by height
    - O(N log N) for sorting N unique coordinates
    - O(B * N * α(N)) where B is the number of buildings, N is the number of unique coordinates,
      and α(N) is the inverse Ackermann function (nearly constant) from Union-Find operations
    - Overall: O(B * N + N log N + B log B), which simplifies to O(B * N) in worst case

    Space Complexity:
    - O(N) for the coordinates list, heights array, index_map dictionary, and Union-Find data structure
    - O(B) for storing the sorted buildings (depending on the implementation of sort)
    - Overall: O(N + B)

    Where:
    - B = number of buildings
    - N = number of unique x-coordinates, which is at most 2*B (each building contributes at most 2 coordinates)

    Args:
        buildings (list): List of [left_x, right_x, height] representing buildings

    Returns:
        list: The skyline as a list of [x, height] points
    """
    # Extract all unique x-coordinates and sort them
    coordinates = sorted(list(set([x for building in buildings for x in building[:2]])))
    n = len(coordinates)

    # Initialize heights for each coordinate position
    heights = [0] * n

    # Create mapping from x-coordinate to its index in the sorted list
    index_map = {x: idx for idx, x in enumerate(coordinates)}

    # Sort buildings by height in descending order to process tallest buildings first
    buildings.sort(key=lambda x: -x[2])

    skyline = []
    # Initialize Union-Find structure for merging processed segments
    uf = UnionFind(n)

    for left_x, right_x, height in buildings:
        # Convert actual coordinates to their positions in the sorted list
        left, right = index_map[left_x], index_map[right_x]

        while left < right:
            # Find the root of the current position
            left = uf.find(left)

            if left < right:
                # Union current position with its representative
                uf.union(left, right)
                # Update height if this is the tallest building covering this position
                heights[left] = height
                left += 1

    # Build the skyline by identifying height changes
    for i in range(n):
        if i == 0 or heights[i] != heights[i - 1]:
            skyline.append([coordinates[i], heights[i]])

    return skyline


# Driver code
def main():
    """
    Test the get_skyline function with various building configurations.

    Prints the buildings and their corresponding skylines.
    """
    buildings = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 7, 9], [3, 4, 11], [5, 9, 10], [12, 16, 19]],
        [[1, 2, 1], [2, 4, 6], [8, 13, 18]],
        [[1, 4, 7]],
        [[2, 13, 5], [5, 8, 11], [8, 10, 1], [10, 11, 12]],
    ]

    for i, building in enumerate(buildings, 1):
        print(f"{i}.\tBuildings:", building)
        print("\n\tSkyline:", get_skyline(building))
        print("-" * 100)
        if i == 1:
            break


if __name__ == "__main__":
    main()
