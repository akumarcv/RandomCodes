from collections import deque


def rotting_oranges(grid):
    """
    Calculates the minimum time to rot all fresh oranges.
    
    This function uses a breadth-first search (BFS) approach to simulate 
    the rotting process. Each minute, all the rotten oranges will spread 
    the rot to adjacent (up, down, left, right) fresh oranges.
    
    Parameters:
    -----------
    grid : List[List[int]]
        A grid where:
        - 0 represents an empty cell
        - 1 represents a fresh orange
        - 2 represents a rotten orange
    
    Returns:
    --------
    int
        The minimum number of minutes needed for all oranges to rot,
        or -1 if it's impossible to rot all oranges
    
    Time Complexity: O(m*n) where m and n are the dimensions of the grid
    Space Complexity: O(m*n) in the worst case for the queue
    """
    # Count fresh oranges and initialize BFS queue with rotten oranges
    fresh = 0
    queue = deque()
    max_minutes = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

    # Initial scan of the grid to count fresh oranges and add rotten ones to queue
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2:
                # Add rotten orange with its position and initial time (0 minutes)
                queue.append((i, j, 0))
            elif grid[i][j] == 1:
                fresh = fresh + 1
                
    # Early return if there are no fresh oranges to rot
    if fresh == 0:
        return 0

    # BFS to simulate the rotting process minute by minute
    while queue:
        # Get the next rotten orange from the queue
        r, c, minute = queue.popleft()
        max_minutes = max(max_minutes, minute)
        
        # Check all four adjacent positions
        for di, dj in directions:
            new_row, new_col = r + di, c + dj
            
            # Check if the adjacent cell is in bounds and has a fresh orange
            if (
                new_row >= 0
                and new_col >= 0
                and new_row < len(grid)
                and new_col < len(grid[0])
                and grid[new_row][new_col] == 1
            ):
                # Mark the fresh orange as rotten
                grid[new_row][new_col] = 2
                
                # Add the newly rotten orange to the queue with incremented time
                queue.append((new_row, new_col, minute + 1))
                
                # Decrement fresh orange count
                fresh -= 1

    # Return the result based on whether all oranges rotted
    return max_minutes if fresh == 0 else -1


# Driver code to test rotting_oranges function
if __name__ == "__main__":
    test_cases = [
        # Test Case 1: All oranges can be rotted in 4 minutes
        ([[2, 1, 1], [1, 1, 0], [0, 1, 1]], 4),
        
        # Test Case 2: Impossible case - some oranges can't be reached
        (
            [[2, 1, 1], [0, 1, 1], [1, 0, 1]],
            -1,
        ),
        
        # Test Case 3: No fresh oranges to rot
        ([[0, 2]], 0),
        
        # Test Case 4: Another impossible case with isolated orange
        ([[1, 2, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1]], -1),
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