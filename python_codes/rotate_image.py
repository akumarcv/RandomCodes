def rotate_image(matrix):
    n = len(matrix)
    for row in range(n// 2):
        for col in range(row, n - row - 1):
            matrix[row][col], matrix[col][n-row-1] = matrix[col][n-row-1], matrix[row][col]
            matrix[row][col], matrix[n-row-1][n-col-1] = matrix[n-row-1][n-col-1] , matrix[row][col]
            matrix[row][col], matrix[n-col-1][row] = matrix[n-col-1][row] , matrix[row][col]


    # Replace this placeholder return statement with your code
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
            print(f"{num:4}", end="")
        print()

# Driver code
def main():

    inputs = [
        [[1]],
        [[6, 9], [2, 7]],
        [[2, 14, 8], [12, 7, 14], [3, 3, 7]],
        [[3, 1, 1, 7], [15, 12, 13, 13], [4, 14, 12, 4], [10, 5, 11, 12]],
        [
            [10, 1, 14, 11, 14],
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
