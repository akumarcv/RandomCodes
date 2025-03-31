def smallest_missing_positive_integer(nums: list[int]) -> int:
    """
    Find the smallest missing positive integer in an array.
    Uses cyclic sort approach to place each number in its correct position.

    Args:
        nums: List of integers (can contain negatives, zero, and duplicates)

    Returns:
        int: Smallest positive integer that is missing from the array

    Time Complexity: O(n) where n is length of input array
    Space Complexity: O(1) as we modify array in-place

    Example:
        >>> smallest_missing_positive_integer([3, 4, -1, 1])
        2  # Array contains 1,3,4 but misses 2
    """
    i = 0
    # First pass: Place each positive number in its correct position
    while i < len(nums):
        correct_pos = nums[i] - 1  # Where current number should be

        # If number can be placed at correct position and isn't already there
        if 0 <= correct_pos < len(nums) and nums[i] != nums[correct_pos]:
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]  # Swap
        else:
            i += 1  # Move to next number if current can't be placed

    # Second pass: Find first position where number doesn't match index
    for i in range(len(nums)):
        if (nums[i] - 1) != i:
            return i + 1  # First missing positive integer

    # If all numbers from 1 to n exist, return n+1
    return len(nums) + 1


def test_smallest_missing_positive():
    """
    Test cases for finding smallest missing positive integer
    Tests various scenarios including edge cases
    """
    test_cases = [
        ([1, 2, 0], 3),  # Missing 3
        ([3, 4, -1, 1], 2),  # Missing 2 with negative number
        ([7, 8, 9, 11, 12], 1),  # Missing 1, all numbers > 1
        ([1, 2, 3, 4], 5),  # Missing 5, consecutive numbers
        ([1, 1], 2),  # Duplicate numbers
        ([], 1),  # Empty array
        ([-1, -2, -3], 1),  # All negative numbers
        ([2, 2, 2, 2], 1),  # All same numbers
    ]

    for i, (nums, expected) in enumerate(test_cases, 1):
        # Create copy of input to preserve original
        test_input = nums.copy()
        result = smallest_missing_positive_integer(test_input)

        print(f"\nTest Case {i}:")
        print(f"Input array: {nums}")
        print(f"Expected smallest missing positive: {expected}")
        print(f"Got: {result}")

        assert (
            result == expected
        ), f"Test case {i} failed! Expected {expected}, got {result}"
        print("✓ Passed")


if __name__ == "__main__":
    test_smallest_missing_positive()
    print("\nAll test cases passed!")
