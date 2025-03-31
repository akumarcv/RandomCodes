from heapq import *
from typing import List


def k_smallest_pairs(list1: List[int], list2: List[int], k: int) -> List[List[int]]:
    """
    Find k pairs with smallest sums where one element is from each list.
    Uses min heap to efficiently track and retrieve smallest sum pairs.

    Args:
        list1: First sorted list of integers
        list2: Second sorted list of integers
        k: Number of pairs to return

    Returns:
        List of k pairs [x,y] with smallest sums where x is from list1 and y from list2

    Time Complexity: O(k * log(min(k,m))) where m is length of list1
    Space Complexity: O(min(k,m)) for heap storage

    Example:
        >>> k_smallest_pairs([1,7,11], [2,4,6], 3)
        [[1,2], [1,4], [1,6]]  # Pairs with smallest sums
    """
    # Storing the length of lists to use it in a loop later
    list_length = len(list1)

    # Declaring a min-heap to keep track of the smallest sums
    # Format: (sum, index_list1, index_list2)
    min_heap = []

    # To store the pairs with smallest sums
    pairs = []

    # Initialize heap with pairs using first element from list2
    for i in range(min(k, list_length)):
        heappush(min_heap, (list1[i] + list2[0], i, 0))

    counter = 1

    # Process pairs until we have k pairs or heap is empty
    while min_heap and counter <= k:
        # Get pair with current smallest sum
        sum_of_pairs, i, j = heappop(min_heap)

        # Add current pair to result
        pairs.append([list1[i], list2[j]])

        # Move to next element in list2 for current list1 element
        next_element = j + 1

        # If we have more elements in list2, add new pair to heap
        if len(list2) > next_element:
            heappush(min_heap, (list1[i] + list2[next_element], i, next_element))

        counter += 1

    return pairs


def main():
    """
    Driver code to test k smallest pairs functionality.
    Tests various scenarios including:
    - Different list sizes
    - Different k values
    - Edge cases
    - Various number distributions
    """
    # Test cases with different configurations
    list1 = [
        [2, 8, 9],  # Regular case
        [1, 2, 300],  # Large number in list
        [1, 1, 2],  # Duplicates
        [4, 6],  # Small lists
        [4, 7, 9],  # Sorted numbers
        [1, 1, 2],  # More duplicates
    ]

    list2 = [
        [1, 3, 6],  # Regular case
        [1, 11, 20, 35, 300],  # More elements
        [1, 2, 3],  # Sequential
        [2, 3],  # Small list
        [4, 7, 9],  # Equal numbers
        [1],  # Single element
    ]

    k = [9, 30, 1, 2, 5, 4]  # Different k values

    # Process each test case
    for i in range(len(k)):
        print(f"{i + 1}.\tInput pairs: {list1[i]}, {list2[i]}")
        print(f"\tk = {k[i]}")
        result = k_smallest_pairs(list1[i], list2[i], k[i])
        print(f"\tPairs with the smallest sum are: {result}")
        print("-" * 100)


if __name__ == "__main__":
    main()
