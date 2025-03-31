from typing import List


def dfs(grid, i, j):
    """
    Depth-first search to mark all connected land cells as visited.
    Changes connected '1's to '0's to mark them as processed.

    Args:
        grid: 2D grid of '1's (land) and '0's (water)
        i: Current row index
        j: Current column index

    Returns:
        None: Modifies grid in-place

    Time Complexity: O(1) per call, but overall O(m*n) for entire grid traversal
    Space Complexity: O(m*n) worst case for recursion stack on a grid of all 1s
    """
    # Check bounds and if current cell is land
    if i < 0 or i > len(grid) - 1 or j < 0 or j > len(grid[0]) - 1 or grid[i][j] != "1":
        return  # Out of bounds or not land, return

    grid[i][j] = "0"  # Mark current land as visited by changing to water
    dfs(grid, i - 1, j)  # Check cell above
    dfs(grid, i + 1, j)  # Check cell below
    dfs(grid, i, j - 1)  # Check cell to left
    dfs(grid, i, j + 1)  # Check cell to right


class Solution:
    """
    Solution class for counting number of islands in a 2D grid.
    Uses DFS to identify and mark connected land cells.
    """

    def numIslands(self, grid: List[List[str]]) -> int:
        """
        Count number of islands in a 2D grid using DFS.
        An island is a group of '1's connected horizontally or vertically.

        Args:
            grid: 2D grid where '1' is land and '0' is water

        Returns:
            int: Number of islands found

        Time Complexity: O(m*n) where m,n are dimensions of grid
        Space Complexity: O(m*n) worst case for recursion stack

        Example:
            >>> Solution().numIslands([["1","1","0"],["0","1","0"],["0","0","1"]])
            2  # Two separate islands
        """
        if not grid:
            return 0  # Empty grid has no islands

        num_island = 0  # Counter for number of islands

        # Iterate through every cell in the grid
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":  # Found unvisited land
                    dfs(grid, i, j)  # Mark all connected land as visited
                    num_island += 1  # Count this connected group as one island

        return num_island


# Driver code to test the numIslands function
if __name__ == "__main__":
    """
    Test cases for island counting algorithm.
    Demonstrates:
    - Multiple islands
    - Single large island
    - Island at edges
    - No islands
    """
    grid = [
        ["1", "1", "0", "0", "0"],  # Island in top-left
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],  # Island in middle
        ["0", "0", "0", "1", "1"],  # Island in bottom-right
    ]

    solution = Solution()
    print("Number of islands:", solution.numIslands(grid))
