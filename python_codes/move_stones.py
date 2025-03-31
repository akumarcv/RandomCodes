from collections import deque


def bfs(grid):
    """
    Perform breadth-first search to find minimum moves to reach target state.
    Target state is a 3x3 grid with exactly one stone in each cell.

    Args:
        grid: 3x3 grid representing current stone distribution

    Returns:
        int: Minimum moves required to reach target, -1 if impossible

    Time Complexity: O(9!) for possible grid states
    Space Complexity: O(9!) for storing visited states
    """
    target = [[1, 1, 1]] * 3  # Target state: one stone per cell
    queue = deque([(grid, 0)])  # Queue stores (current_grid, moves_count)

    visited = set()  # Track seen grid states to avoid cycles
    visited.add(tuple(tuple(row) for row in grid))  # Convert grid to hashable type

    while queue:
        current, moves = queue.popleft()  # Get next grid state to process
        if current == target:  # Check if target state reached
            return moves
        # Try moving stones from each cell
        for i in range(3):
            for j in range(3):
                if current[i][j] > 1:  # Can only move if cell has multiple stones
                    # Check all four adjacent cells
                    for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                        if 0 <= x < 3 and 0 <= y < 3:  # Stay within grid bounds
                            # Create new grid with stone moved
                            new_grid = [list(row) for row in current]
                            new_grid[i][j] -= 1  # Remove stone from current cell
                            new_grid[x][y] += 1  # Add stone to adjacent cell
                            new_tuple = tuple(tuple(row) for row in new_grid)
                            # Process new state if not seen before
                            if new_tuple not in visited:
                                queue.append((new_grid, moves + 1))
                                visited.add(new_tuple)
    return -1  # No solution found


def minimum_moves(grid):
    """
    Find minimum moves needed to distribute stones evenly.
    All cells should have exactly one stone after moves.

    Args:
        grid: 3x3 grid with initial stone distribution

    Returns:
        int: Minimum moves required, -1 if impossible

    Example:
        >>> minimum_moves([[1,1,1], [1,2,3], [0,0,0]])
        3  # Takes 3 moves to distribute stones evenly
    """
    # Check if total stones is correct (should be 9)
    stones = sum(sum(row) for row in grid)
    if stones != 9:  # Impossible if total stones != 9
        return -1
    return bfs(grid)  # Find minimum moves using BFS


def main():
    grids = [
        [
            [1, 1, 1],
            [1, 2, 3],
            [0, 0, 0],
        ],
        [
            [8, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [2, 2, 2],
            [1, 1, 1],
            [0, 0, 0],
        ],
        [
            [3, 0, 0],
            [3, 0, 0],
            [3, 0, 0],
        ],
        [
            [1, 0, 1],
            [3, 0, 0],
            [0, 4, 0],
        ],
    ]

    for i in range(len(grids)):
        print(i + 1, ".\t Input grid: ", sep="")
        print(grids[i])
        print("\n\t Minimum number of moves: ", minimum_moves(grids[i]), sep="")
        print("-" * 100)


if __name__ == "__main__":
    main()
