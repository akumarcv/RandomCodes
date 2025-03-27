from collections import deque


def rotting_oranges(grid):

    fresh = 0
    queue = deque()
    max_minutes = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2:
                queue.append((i, j, 0))
            elif grid[i][j] == 1:
                fresh = fresh + 1
    if fresh == 0:
        return 0

    while queue:
        r, c, minute = queue.popleft()
        max_minutes = max(max_minutes, minute)
        for di, dj in directions:
            new_row, new_col = r + di, c + dj
            if (
                new_row >= 0
                and new_col >= 0
                and new_row < len(grid)
                and new_col < len(grid[0])
                and grid[new_row][new_col] == 1
            ):
                grid[new_row][new_col] = 2
                queue.append((new_row, new_col, minute + 1))
                fresh -= 1

    return max_minutes if fresh == 0 else -1


# ...existing code...

# Driver code to test rotting_oranges function
if __name__ == "__main__":
    test_cases = [
        ([[2, 1, 1], [1, 1, 0], [0, 1, 1]], 4),  # Expected output
        (
            [[2, 1, 1], [0, 1, 1], [1, 0, 1]],
            -1,  # Expected output (impossible to rot all oranges)
        ),
        ([[0, 2]], 0),  # Expected output (no fresh oranges)
        ([[1, 2, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1]], -1),  # Expected output
    ]

    for i, (grid, expected) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print("Grid:")
        for row in grid:
            print(row)

        # Create a deep copy of grid to avoid modifying original
        import copy

        test_grid = copy.deepcopy(grid)

        result = rotting_oranges(test_grid)
        print(f"Expected Minutes: {expected}")
        print(f"Result Minutes: {result}")

        print("✓ Test passed") if expected == result else print("x Test failed")
