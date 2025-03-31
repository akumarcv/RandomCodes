import numpy as np


def print_diag(arr):
    """
    Print all anti-diagonals of a 2D matrix from top-left to bottom-right.
    An anti-diagonal is a line of elements where row index increases and column index decreases.

    Args:
        arr: 2D matrix represented as a list of lists

    Returns:
        None: Prints diagonals directly to console

    Time Complexity: O(ROW*COL) where ROW,COL are dimensions of the matrix
    Space Complexity: O(1) as we only use constant extra space

    Example:
        >>> print_diag([[1,2,3],[4,5,6],[7,8,9]])
        1
        4 2
        7 5 3
        8 6
        9
    """
    ROW, COL = len(arr), len(arr[0])  # Get matrix dimensions

    # Start from each element of the first row
    for start_col in range(COL):
        i, j = 0, start_col  # Starting position (top row, varying column)
        while i < ROW and j >= 0:
            print(arr[i][j], end=" ")  # Print current element
            i += 1  # Move down one row
            j -= 1  # Move left one column
        print()  # New line after each diagonal

    # Start from each element of the last column except the first element
    for start_row in range(1, ROW):
        i, j = start_row, COL - 1  # Starting position (rightmost column, varying row)
        while i < ROW and j >= 0:
            print(arr[i][j], end=" ")  # Print current element
            i += 1  # Move down one row
            j -= 1  # Move left one column
        print()  # New line after each diagonal


if __name__ == "__main__":
    """
    Driver code to test the print_diag function.
    Creates a sample 2D matrix and prints all its anti-diagonals.
    """
    # Example matrix with 3 rows and 4 columns
    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    print("Input matrix:")
    for row in A:
        print(row)

    print("\nPrinting anti-diagonals:")
    print_diag(A)
