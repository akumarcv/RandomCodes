"""
Russian Doll Envelopes Problem

You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi] represents the width and height of an envelope.
One envelope can fit into another if and only if both the width and height of one envelope is greater than the other.
Return the maximum number of envelopes that can be nested (Russian doll style).

This problem is a variation of the Longest Increasing Subsequence (LIS) problem in 2D.

Time Complexity:
- Sorting: O(n log n)
- Finding LIS: O(n log n)
- Overall: O(n log n)

Space Complexity: O(n) for the LIS array
"""

import time


def find_position(lis, val):
    """
    Binary search to find the position where val should be inserted in the LIS array
    
    Args:
        lis: List of increasing elements
        val: Value to find position for
    
    Returns:
        Position where val should be inserted to maintain the list order
    
    Time Complexity: O(log n)
    """
    low = 0
    high = len(lis) - 1

    while low <= high:
        mid = (low + high) // 2
        if lis[mid] == val:
            return mid
        elif lis[mid] < val:
            low = mid + 1
        else:
            high = mid - 1
    return low


def max_envelopes(envelopes):
    """
    Find the maximum number of Russian doll envelopes
    
    Args:
        envelopes: List of [width, height] pairs representing envelopes
    
    Returns:
        Maximum number of nested envelopes possible
    
    Time Complexity: O(n log n)
    """
    # Sort by increasing width and decreasing height when widths are equal
    # This trick ensures we only need to find LIS on heights
    envelopes.sort(key=lambda x: (x[0], -x[1]))

    lis = []
    for i in range(len(envelopes)):
        if not lis or envelopes[i][1] > lis[-1]:
            lis.append(envelopes[i][1])
        else:
            pos = find_position(lis, envelopes[i][1])
            lis[pos] = envelopes[i][1]
    return len(lis)


# Driver code
def main():
    """Test the max_envelopes function with various examples"""
    envelopes = [
        [[1, 4], [6, 4], [9, 5], [3, 3]],
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
        [[4, 4], [4, 4], [4, 4]],
        [[3, 1], [5, 8], [5, 9], [3, 1], [9, 1]],
        [[9, 8], [3, 1], [4, 5], [2, 1], [5, 7]],
    ]

    for i in range(len(envelopes)):
        print(f"Example {i + 1}:")
        print(f"  Envelopes: {envelopes[i]}")
        
        # Measure execution time
        start_time = time.time()
        result = max_envelopes(envelopes[i])
        end_time = time.time()
        
        print(f"  Maximum number of Russian-dolled envelopes: {result}")
        print(f"  Execution time: {(end_time - start_time)*1000:.4f} ms")
        print("-" * 80)


if __name__ == "__main__":
    main()
