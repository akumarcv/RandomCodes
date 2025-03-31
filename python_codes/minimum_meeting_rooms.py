import heapq


def find_sets(intervals):
    """
    Find minimum number of meeting rooms required to schedule all meetings.
    Uses sorting by start time and min heap to track earliest ending meetings.

    Args:
        intervals: List of [start_time, end_time] for each meeting

    Returns:
        int: Minimum number of rooms needed to accommodate all meetings

    Time Complexity: O(n log n) where n is number of meetings
    Space Complexity: O(n) for storing meetings in heap

    Example:
        >>> find_sets([[2,8], [3,4], [3,9]])
        2  # Need 2 rooms for overlapping meetings
    """
    # Sort meetings by start time to process chronologically
    intervals.sort(key=lambda x: x[0])

    # Initialize min heap with first meeting's end time
    min_heap = []
    heapq.heappush(min_heap, intervals[0][1])

    # Process remaining meetings
    for i in range(1, len(intervals)):
        # If current meeting starts after earliest ending
        if intervals[i][0] >= min_heap[0]:
            heapq.heappop(min_heap)  # Reuse the room
        heapq.heappush(min_heap, intervals[i][1])  # Add current meeting end time

    # Heap size represents minimum rooms needed
    return len(min_heap)


def main():
    """
    Driver code to test minimum meeting rooms calculation.
    Tests various meeting schedules including:
    - Sequential meetings
    - Overlapping meetings
    - Nested meetings
    - Back-to-back meetings
    """
    inputs = [
        [[2, 8], [3, 4], [3, 9], [5, 11], [8, 20], [11, 15]],  # Mixed schedule
        [[1, 4], [2, 5], [3, 6]],  # Sequential overlap
        [[4, 5], [2, 3], [2, 4], [3, 5]],  # Dense scheduling
        [[1, 4], [4, 5], [5, 6]],  # Back-to-back
    ]

    for i, meetings in enumerate(inputs, 1):
        print(f"\nTest Case {i}:")
        print(f"Input meetings: {meetings}")
        rooms = find_sets(meetings)
        print(f"Minimum rooms needed: {rooms}")
        print("-" * 70)


if __name__ == "__main__":
    main()
