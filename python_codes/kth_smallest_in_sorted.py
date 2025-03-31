from heapq import *
from typing import List


def k_smallest_number(lists: List[List[int]], k: int) -> int:
    """
    Find kth smallest number among all sorted lists using min heap approach.
    
    Args:
        lists: List of sorted lists containing integers
        k: Position of number to find (1-based)
        
    Returns:
        int: kth smallest number across all lists
        
    Time Complexity: O(K * log(N)) where N is number of lists
    Space Complexity: O(N) for storing heap entries
    
    Example:
        >>> k_smallest_number([[2,6,8], [3,6,10]], 3)
        6  # Third smallest number among all lists
    """
    # Initialize min heap with first element from each list
    # Format: (value, list_index, element_index)
    min_heap = []
    for i in range(len(lists)):
        if len(lists[i]) > 0:
            heappush(min_heap, (lists[i][0], i, 0))

    pop_count = 0
    k_smallest_number = 0
    
    # Process elements until we find kth smallest
    while min_heap:
        # Get smallest current element and its position
        k_smallest_number, list_index, element_index = heappop(min_heap)
        pop_count += 1
        
        # Found kth element
        if pop_count == k:
            break

        # Add next element from same list if available
        if element_index < len(lists[list_index]) - 1:
            heappush(
                min_heap,
                (lists[list_index][element_index + 1], list_index, element_index + 1),
            )

    return k_smallest_number


def main():
    """
    Driver code to test k smallest number functionality.
    Tests various scenarios including:
    - Multiple sorted lists
    - Empty lists
    - Different k values
    - Lists with duplicates
    """
    lists = [
        [[2, 6, 8], [3, 6, 10], [5, 8, 11]],           # Regular case
        [[1, 2, 3], [4, 5], [6, 7, 8, 15], 
         [10, 11, 12, 13], [5, 10]],                    # Many lists
        [[], [], []],                                    # Empty lists
        [[1, 1, 3, 8], [5, 5, 7, 9], [3, 5, 8, 12]],   # Duplicates
        [[5, 8, 9, 17], [], [8, 17, 23, 24]],          # Some empty lists
    ]

    k = [5, 50, 7, 4, 8]  # Different k values

    # Process each test case
    for i in range(len(k)):
        print(
            f"{i + 1}.\tInput lists: {lists[i]}"
            f"\n\tK = {k[i]}"
            f"\n\t{k[i]}th smallest number from the given lists is: "
            f"{k_smallest_number(lists[i], k[i])}"
        )
        print("-" * 100)


if __name__ == "__main__":
    main()