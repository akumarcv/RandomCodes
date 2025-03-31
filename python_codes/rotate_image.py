def rotate_image(matrix):
    """
    Rotate a square matrix 90 degrees clockwise in-place.
    Uses a layer-by-layer rotation approach to swap elements in groups of 4.

    Args:
        matrix: Square matrix represented as a list of lists

    Returns:
        list: The rotated matrix (same object, modified in-place)

    Time Complexity: O(n²) where n is the dimension of the matrix
    Space Complexity: O(1) as rotation is done in-place

    Example:
        >>> rotate_image([[1,2], [3,4]])
        [[3,1], [4,2]]  # 90 degrees clockwise rotation
    """
    n = len(matrix)  # Get matrix dimensions

    # Process each layer of the matrix from outermost to innermost
    for row in range(n // 2):
        # For each layer, process elements except the corners
        for col in range(row, n - row - 1):
            # Perform a 4-way swap to rotate elements 90° clockwise
            # Swap 1: top → right
            matrix[row][col], matrix[col][n - row - 1] = (
                matrix[col][n - row - 1],  # Right side element
                matrix[row][col],  # Top side element
            )
            # Swap 2: right → bottom (using new value at top position)
            matrix[row][col], matrix[n - row - 1][n - col - 1] = (
                matrix[n - row - 1][n - col - 1],  # Bottom side element
                matrix[row][col],  # Right side element (now at top)
            )
            # Swap 3: bottom → left (using new value at top position)
            matrix[row][col], matrix[n - col - 1][row] = (
                matrix[n - col - 1][row],  # Left side element
                matrix[row][col],  # Bottom side element (now at top)
            )
            # After these 3 swaps, the 4 elements have been rotated clockwise

    return matrix


def print_matrix(matrix):
    """
    Print a 2D matrix in a readable format
    Args:
        matrix: 2D list of integers
    """
    for row in matrix:
        print("\t", end="")
        for num in row:
            print(f"{num:4}", end="")  # Format each number with width 4
        print()  # New line after each row


# Driver code
def main():
    """
    Test the matrix rotation function with various examples.
    Tests different matrix sizes including:
    - 1x1 (trivial case)
    - 2x2 (simplest meaningful rotation)
    - 3x3 (odd dimensions)
    - 4x4 (even dimensions)
    - 5x5 (larger odd dimensions)

    For each matrix:
    1. Prints the original matrix
    2. Rotates it 90 degrees clockwise
    3. Prints the rotated result
    """
    inputs = [
        [[1]],  # 1x1 matrix
        [[6, 9], [2, 7]],  # 2x2 matrix
        [[2, 14, 8], [12, 7, 14], [3, 3, 7]],  # 3x3 matrix
        [[3, 1, 1, 7], [15, 12, 13, 13], [4, 14, 12, 4], [10, 5, 11, 12]],  # 4x4 matrix
        [
            [10, 1, 14, 11, 14],  # 5x5 matrix
            [13, 4, 8, 2, 13],
            [10, 19, 1, 6, 8],
            [20, 10, 8, 2, 12],
            [15, 6, 8, 8, 18],
        ],
    ]

    for i in range(len(inputs)):
        print(i + 1, ".\tMatrix:", sep="")
        print_matrix(inputs[i])

        print("\n\tRotated matrix:", sep="")
        print_matrix(rotate_image(inputs[i]))

        print("-" * 100)


if __name__ == "__main__":
    main()
