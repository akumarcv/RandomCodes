from typing import List


def dfs(grid, i, j):
    if i < 0 or i > len(grid)-1 or j < 0 or j > len(grid[0])-1 or grid[i][j] != '1':
        return

    grid[i][j] = '0'
    dfs(grid, i - 1, j)
    dfs(grid, i + 1, j)
    dfs(grid, i, j - 1)
    dfs(grid, i, j + 1)


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        num_island = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    dfs(grid, i, j)
                    num_island += 1
        return num_island


# Driver code to test the numIslands function
if __name__ == "__main__":
    grid = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]

    solution = Solution()
    print("Number of islands:", solution.numIslands(grid))