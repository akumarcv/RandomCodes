from typing import List


def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    Merge overlapping intervals.

    Args:
        intervals: List of intervals where each interval is [start, end]

    Returns:
        List of merged non-overlapping intervals

    Time Complexity: O(n log n) where n is the number of intervals (due to sorting)
    Space Complexity: O(n) for storing the result
    """
    # Handle edge case of empty input
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])

    result = []
    i = 0
    while i < len(intervals):
        # Add first interval to result
        if i == 0:
            result.append(intervals[i])
        else:
            # If current interval overlaps with the last interval in result
            if intervals[i][0] <= result[-1][1]:
                # Merge by updating the end time to the maximum of both end times
                result[-1][1] = max(intervals[i][1], result[-1][1])
            else:
                # If no overlap, add the current interval to result
                result.append(intervals[i])
        i += 1
    return result


# Driver code
if __name__ == "__main__":
    # Test cases
    test_cases = [
        [[1, 3], [2, 6], [8, 10], [15, 18]],  # Basic overlapping intervals
        [[1, 4], [4, 5]],  # Intervals that touch
        [[1, 4], [2, 3]],  # Contained interval
        [[1, 4]],  # Single interval
        [],  # Empty input
        [[1, 10], [2, 3], [4, 7], [6, 8]],  # Multiple overlapping intervals
    ]

    for i, intervals in enumerate(test_cases):
        print(f"Test Case {i+1}:")
        print(f"Input: {intervals}")
        print(f"Output: {merge(intervals)}")
        print("-" * 40)
