def helper(grid, i, j, value, target):
    if i < 0 or j < 0 or i > len(grid) - 1 or j > len(grid[0]) - 1:
        return

    if grid[i][j] != value:
        return

    grid[i][j] = target
    helper(grid, i + 1, j, value, target)
    helper(grid, i - 1, j, value, target)
    helper(grid, i, j + 1, value, target)
    helper(grid, i, j - 1, value, target)


def flood_fill(grid, sr, sc, target):
    if grid[sr][sc] == target:
        return grid
    helper(grid, sr, sc, grid[sr][sc], target)
    # Replace this placeholder return statement with your code

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
