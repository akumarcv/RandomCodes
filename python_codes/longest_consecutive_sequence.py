from typing import List, Set


def longest_consecutive_sequence(nums: List[int]) -> int:
    """
    Find length of longest consecutive sequence in an unsorted array.
    Uses set data structure for O(1) lookups and optimizes by only checking
    sequence starts (numbers that have no left neighbor).

    Args:
        nums: List of integers (can be unsorted, contain duplicates)

    Returns:
        int: Length of longest consecutive sequence

    Time Complexity: O(n) where n is length of input array
    Space Complexity: O(n) for storing numbers in set

    Example:
        >>> longest_consecutive_sequence([100,4,200,1,3,2])
        4  # Longest sequence is [1,2,3,4]
    """
    # Convert to set for O(1) lookups and remove duplicates
    num_set: Set[int] = set(nums)
    longest_streak: int = 0

    # Check each potential sequence start
    for n in num_set:
        # Only process numbers that could start a sequence
        # (numbers that have no left neighbor)
        if n - 1 not in num_set:
            current_num: int = n
            curr_streak: int = 1

            # Keep extending sequence as long as next number exists
            while current_num + 1 in num_set:
                current_num += 1
                curr_streak += 1

            # Update longest streak if current is longer
            longest_streak = max(curr_streak, longest_streak)

    return longest_streak


def test_longest_consecutive_sequence() -> None:
    """
    Test cases for finding longest consecutive sequence.
    Tests various scenarios including:
    - Basic consecutive sequences
    - Sequences with gaps
    - Empty array
    - Single element
    - Duplicates
    - Negative numbers
    - Already sorted sequences
    """
    test_cases = [
        ([100, 4, 200, 1, 3, 2], 4),  # Basic case [1,2,3,4]
        ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9),  # Longer sequence [0-8]
        ([], 0),  # Empty array
        ([1], 1),  # Single element
        ([1, 2, 0, 1], 3),  # With duplicates [0,1,2]
        ([1, 2, 3, 4, 5], 5),  # Already consecutive
        ([9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6], 7),  # With negatives
    ]

    for i, (nums, expected) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input array: {nums}")
        print(f"Expected longest sequence length: {expected}")

        result = longest_consecutive_sequence(nums)
        print(f"Got sequence length: {result}")

        assert (
            result == expected
        ), f"Test case {i} failed! Expected {expected}, got {result}"
        print("✓ Passed")

        # Print the actual sequence if array is not empty
        if nums:
            sequence = []
            start = min(nums)
            while len(sequence) < result and start <= max(nums):
                if start in set(nums):
                    sequence.append(start)
                start += 1
            print(f"Longest consecutive sequence: {sequence}")


if __name__ == "__main__":
    test_longest_consecutive_sequence()
    print("\nAll test cases passed!")
