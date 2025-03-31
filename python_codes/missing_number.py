def missing_number(nums: list[int]) -> int:
    """
    Find missing number in sequence 0 to n using cyclic sort approach.
    Places each number in its correct position and finds first mismatch.

    Args:
        nums: List of integers from 0 to n with one number missing

    Returns:
        int: The missing number in the sequence

    Time Complexity: O(n) where n is length of input array
    Space Complexity: O(1) as we sort in-place

    Example:
        >>> missing_number([3,0,1])
        2  # Number 2 is missing from sequence [0,1,3]
    """
    # Use cyclic sort to place numbers at correct indices
    index = 0
    while index < len(nums):
        correct_pos = nums[index]  # Get correct position for current number
        # If number can be placed at its correct position
        if correct_pos < len(nums) and correct_pos != index:
            # Swap number to its correct position
            nums[index], nums[correct_pos] = nums[correct_pos], nums[index]
        else:
            index += 1  # Move to next number if current is in position

    # Find first position where index doesn't match number
    for x in range(len(nums)):
        if x != nums[x]:
            return x  # Found missing number

    # If all numbers present, missing number is n
    return len(nums)


def test_missing_number():
    """
    Test cases for missing_number function
    Each test case contains:
    - Input array
    - Expected missing number

    Tests various scenarios including:
    - Missing number in middle
    - Missing last number
    - Different array orderings
    - Single element arrays
    - Larger sequences
    """
    # ...existing code...
