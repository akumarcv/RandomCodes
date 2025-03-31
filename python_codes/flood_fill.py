from typing import List


def helper(grid: List[List[int]], i: int, j: int, value: int, target: int) -> None:
    """
    Recursive helper function to perform flood fill operation.

    Args:
        grid: 2D grid to perform flood fill on
        i: Current row index
        j: Current column index
        value: Original color value to replace
        target: New color value to fill with

    Time Complexity: O(m*n) where m,n are grid dimensions
    Space Complexity: O(m*n) for recursion stack in worst case
    """
    # Check if current position is out of bounds
    if i < 0 or j < 0 or i > len(grid) - 1 or j > len(grid[0]) - 1:
        return

    # Check if current cell needs to be filled
    if grid[i][j] != value:
        return

    # Fill current cell and recursively fill neighbors
    grid[i][j] = target
    helper(grid, i + 1, j, value, target)  # Down
    helper(grid, i - 1, j, value, target)  # Up
    helper(grid, i, j + 1, value, target)  # Right
    helper(grid, i, j - 1, value, target)  # Left


def flood_fill(grid: List[List[int]], sr: int, sc: int, target: int) -> List[List[int]]:
    """
    Perform flood fill operation starting from given position.
    Similar to paint bucket tool in image editors.

    Args:
        grid: 2D grid to perform flood fill on
        sr: Starting row index
        sc: Starting column index
        target: New color value to fill with

    Returns:
        Modified grid after flood fill operation

    Time Complexity: O(m*n) where m,n are grid dimensions
    Space Complexity: O(m*n) for recursion stack

    Example:
        >>> grid = [[1,1,1],[1,1,0],[1,0,1]]
        >>> flood_fill(grid, 1, 1, 2)
        [[2,2,2],[2,2,0],[2,0,1]]
    """
    # No change needed if starting position already has target color
    if grid[sr][sc] == target:
        return grid

    # Start flood fill from given position
    helper(grid, sr, sc, grid[sr][sc], target)
    return grid


# Driver code


def main():
    grids = [
        [
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
        ],
        [[1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1]],
        [[9, 9, 6, 9], [6, 9, 9, 6], [6, 9, 9, 9], [9, 9, 9, 9]],
        [[1, 1, 0, 1], [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1]],
        [[1, 2, 0, 0], [3, 1, 3, 6], [7, 2, 1, 5], [1, 9, 2, 1]],
    ]

    starting_row = [4, 2, 2, 2, 1]
    starting_col = [3, 3, 1, 3, 1]
    new_target = [3, 2, 1, 0, 4]

    for i in range(len(grids)):
        print(i + 1, ".\t Grid before flood fill: ", grids[i], sep="")
        print(
            "\t Starting row and column are: (",
            starting_row[i],
            ", ",
            starting_col[i],
            ")",
            sep="",
        )
        print("\t Target value: ", new_target[i], sep="")
        print(
            "\t After perform flood fill: ",
            flood_fill(grids[i], starting_row[i], starting_col[i], new_target[i]),
            sep="",
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
