from typing import List

def insert_interval(existing_intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    """
    Insert a new interval into a sorted list of non-overlapping intervals.
    Merges overlapping intervals after insertion.
    
    Args:
        existing_intervals: List of sorted, non-overlapping intervals [start, end]
        new_interval: New interval to insert [start, end]
        
    Returns:
        List of merged intervals after inserting new interval
        
    Time Complexity: O(n) where n is number of existing intervals
    Space Complexity: O(n) for storing result
    
    Example:
        >>> insert_interval([[1,3], [6,9]], [2,5])
        [[1,5], [6,9]]  # [2,5] merges with [1,3]
    """
    new_result = []
    i = 0
    
    # Add all intervals that start before new interval
    while i < len(existing_intervals) and existing_intervals[i][0] < new_interval[0]:
        new_result.append(existing_intervals[i])
        i += 1

    # Insert new interval (either append or merge with last interval)
    if not new_result or new_result[-1][1] < new_interval[0]:
        new_result.append(new_interval)  # No overlap, just append
    else:
        new_result[-1][1] = max(new_result[-1][1], new_interval[1])  # Merge with last interval

    # Process remaining intervals
    while i < len(existing_intervals):
        if new_result[-1][1] < existing_intervals[i][0]:
            new_result.append(existing_intervals[i])  # No overlap, append as is
        else:
            new_result[-1][1] = max(new_result[-1][1], existing_intervals[i][1])  # Merge
        i += 1
        
    return new_result

def main():
    """
    Driver code to test interval insertion with various test cases.
    Tests different scenarios including:
    - Regular insertions with merging
    - Non-overlapping insertions
    - Complete overlap cases
    - Edge intervals
    """
    new_interval = [
        [2, 5],    # Regular case with merge
        [16, 18],  # Non-overlapping insertion
        [10, 12],  # Insertion between intervals
        [1, 3],    # Overlap at start
        [1, 10]    # Large overlap
    ]
    
    existing_intervals = [
        [[1, 2], [3, 4], [5, 8], [9, 15]],          # Multiple small intervals
        [[1, 3], [5, 7], [10, 12], [13, 15], [19, 21], [21, 25], [26, 27]],  # Many intervals
        [[8, 10], [12, 15]],                         # Few intervals
        [[5, 7], [8, 9]],                           # Non-overlapping
        [[3, 5]]                                     # Single interval
    ]

    for i in range(len(new_interval)):
        print(f"{i + 1}.\tExisting intervals: {existing_intervals[i]}")
        print(f"\tNew interval: {new_interval[i]}")
        output = insert_interval(existing_intervals[i], new_interval[i])
        print(f"\tUpdated intervals: {output}")
        print("-" * 100)

if __name__ == "__main__":
    main()