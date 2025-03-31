import heapq


def find_right_interval(intervals):
    """
    Find the right interval for each interval in the input list.
    The right interval for an interval 'i' is the interval 'j' with the
    minimum start point such that j.start >= i.end.

    Args:
        intervals: List of intervals, each represented as [start, end]

    Returns:
        list: Array of indices where result[i] is the index of the right
              interval of intervals[i], or -1 if no such interval exists

    Time Complexity: O(n log n) where n is number of intervals
    Space Complexity: O(n) for storing the heaps

    Example:
        >>> find_right_interval([[1,2], [2,3], [3,4]])
        [1, 2, -1]  # For [1,2] right interval is [2,3] at index 1
    """
    result = [-1] * len(
        intervals
    )  # Initialize result with -1 (no right interval found)

    start_heap = []  # Min heap for interval start points
    end_heap = []  # Min heap for interval end points
    for i in range(len(intervals)):
        heapq.heappush(start_heap, (intervals[i][0], i))  # Push (start point, index)
        heapq.heappush(end_heap, (intervals[i][1], i))  # Push (end point, index)

    while end_heap:
        end, index = heapq.heappop(end_heap)  # Get interval with smallest end point
        while start_heap and start_heap[0][0] < end:
            heapq.heappop(start_heap)  # Remove intervals with start < current end
        if start_heap:
            # The minimum remaining start point is the right interval
            result[index] = start_heap[0][1]  # Store index of right interval

    return result


def main():
    """
    Driver code to test find_right_interval function with various test cases.
    Tests different interval configurations including:
    - Single interval case
    - Overlapping intervals
    - Non-overlapping intervals
    - Intervals in different orders
    """
    test_cases = [
        [[1, 2]],  # Single interval
        [[3, 4], [2, 3], [1, 2]],  # Descending start points
        [[1, 4], [2, 3], [3, 4]],  # Overlapping intervals
        [[5, 6], [1, 2], [3, 4]],  # Non-adjacent intervals
        [[1, 3], [2, 4], [3, 5], [4, 6]],  # Sequential overlapping intervals
    ]

    for i, test_case in enumerate(test_cases):
        print(i + 1, "\tintervals:", test_case)
        result = find_right_interval(test_case)
        print("\n\tOutput:", result)
        print("-" * 100)


if __name__ == "__main__":
    main()
