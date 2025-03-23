from collections import deque


def bfs(grid):
    target = [[1, 1, 1]] * 3
    queue = deque([(grid, 0)])

    visited = set()
    visited.add(tuple(tuple(row) for row in grid))

    while queue:
        current, moves = queue.popleft()
        if current == target:
            return moves
        for i in range(3):
            for j in range(3):
                if current[i][j] > 1:
                    for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                        if 0 <= x < 3 and 0 <= y < 3:
                            new_grid = [list(row) for row in current]
                            new_grid[i][j] -= 1
                            new_grid[x][y] += 1
                            new_tuple = tuple(tuple(row) for row in new_grid)
                            if new_tuple not in visited:
                                queue.append((new_grid, moves + 1))
                                visited.add(new_tuple)
    return -1


def minimum_moves(grid):
    stones = sum(sum(row) for row in grid)
    if stones != 9:
        return -1
    return bfs(grid)


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
