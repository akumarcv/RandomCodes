tab = 2


def print_board_state(n, solution, indent=1):
    if len(solution) != n:
        return
    indent += tab
    for i in range(n):
        if i == 0:  # printing the top of the board
            print(" " * indent, " ", "_" * ((4 * n) - 1), sep="")

        print(" " * indent, "|", sep="", end="")

        # queen placed in this row in the solution
        if solution[n - (i + 1)] != -1:
            for j in range(n):
                print("_X_|" if solution[i][j] == 1 else "___|", end="")
        else:  # no queen in solution in this row, move not in this row
            print("___|" * n, end="")  # so print an empty row
        print("")


def check_correctness(board, i, j):
    for k in range(len(board)):
        if board[i][k] == 1 or board[k][j] == 1:
            return False
    for k in range(len(board)):
        for col in range(len(board)):
            if (k + col == i + j) or (k - col == i - j):
                if board[k][col] == 1:
                    return False
    return True


def helper(n, board, i, solution):
    if i == n:
        print_board_state(len(board), board)
        solution[0] += 1
        return
    for j in range(n):
        if check_correctness(board, i, j):
            board[i][j] = 1
            helper(n, board, i + 1, solution)
            board[i][j] = 0

    return


def solve_n_queens(n):

    i = 0
    solution = [0]
    board = [[0 for _ in range(n)] for _ in range(n)]
    helper(n, board, i, solution)

    # Replace this placeholder return statement with your code
    return solution[0]


def main():
    n = [4, 5, 6, 7, 8]
    for i in range(len(n)):
        print(
            i + 1,
            ". Queens: ",
            n[i],
            ", Chessboard: \
            (",
            n[i],
            "x",
            n[i],
            ")",
            sep="",
        )
        res = solve_n_queens(n[i])
        global tab
        tab = 2
        print(
            "\nTotal solutions count for ",
            n[i],
            " queens on a ",
            n[i],
            "x",
            n[i],
            " chessboard: ",
            res,
            sep="",
        )
        print("-" * 100, "\n", sep="")


if __name__ == "__main__":
    main()
