import numpy as np

def print_diag(arr):
    ROW, COL = len(arr), len(arr[0])
    
    # Start from each element of the first row
    for start_col in range(COL):
        i, j = 0, start_col
        while i < ROW and j >= 0:
            print(arr[i][j], end=' ')
            i += 1
            j -= 1
        print()
    
    # Start from each element of the last column except the first element
    for start_row in range(1, ROW):
        i, j = start_row, COL - 1
        while i < ROW and j >= 0:
            print(arr[i][j], end=' ')
            i += 1
            j -= 1
        print()

if __name__ == "__main__":
    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    print_diag(A)