tab = 2  # Indentation constant for board visualization


def print_board_state(n, solution, indent=1):
    """
    Print the chessboard with queens placement visualization.
    
    Args:
        n: Size of the board (nxn)
        solution: 2D matrix with queens placement (1 for queen, 0 for empty)
        indent: Amount of indentation for board display
        
    Returns:
        None: Displays board directly to console
    """
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
    """
    Check if placing a queen at position (i,j) is valid.
    Verifies no queens attack each other (no shared row, column, diagonal).
    
    Args:
        board: Current board state with queens placed
        i: Row index for potential queen placement
        j: Column index for potential queen placement
        
    Returns:
        bool: True if placement is valid, False otherwise
        
    Time Complexity: O(n²) where n is board size
    """
    for k in range(len(board)):
        if board[i][k] == 1 or board[k][j] == 1:
            return False  # Check same row and same column
    for k in range(len(board)):
        for col in range(len(board)):
            if (k + col == i + j) or (k - col == i - j):
                if board[k][col] == 1:
                    return False  # Check diagonals
    return True


def helper(n, board, i, solution):
    """
    Recursive backtracking function to place queens on the board.
    
    Args:
        n: Size of the board (nxn)
        board: Current state of the board
        i: Current row being processed
        solution: List to track number of solutions found
        
    Returns:
        None: Updates solution count in-place
    """
    if i == n:
        print_board_state(len(board), board)  # Found a valid solution
        solution[0] += 1  # Increment solution count
        return
    for j in range(n):
        if check_correctness(board, i, j):
            board[i][j] = 1  # Place queen
            helper(n, board, i + 1, solution)  # Try next row
            board[i][j] = 0  # Backtrack (remove queen)

    return


def solve_n_queens(n):
    """
    Solve the N-Queens problem for a board of size nxn.
    
    Args:
        n: Size of the board (nxn)
        
    Returns:
        int: Number of distinct solutions found
        
    Time Complexity: O(n!) as we need to explore factorial possibilities
    Space Complexity: O(n²) for the board representation
    
    Example:
        >>> solve_n_queens(4)
        2  # There are exactly 2 distinct solutions for 4 queens
    """
    i = 0  # Start from first row
    solution = [0]  # Track solution count
    board = [[0 for _ in range(n)] for _ in range(n)]  # Initialize empty board
    helper(n, board, i, solution)

    # Return the total count of solutions found
    return solution[0]


def main():
    """
    Driver function to test N-Queens solutions for multiple board sizes.
    Tests board sizes from 4x4 to 8x8.
    Displays:
    - Board visualizations for each solution
    - Total solution count for each board size
    """
    n = [4, 5, 6, 7, 8]  # Different board sizes to test
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