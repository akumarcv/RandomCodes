def find_corrupt_pair(nums: list[int]) -> list[int]:
    """
    Find missing and duplicate numbers in an array using cyclic sort.
    
    In a correct array of n numbers, each number from 1 to n should appear once.
    This function finds the missing number and the duplicate number.
    
    Args:
        nums: List of integers from 1 to n where one number is missing
             and one number is duplicated
             
    Returns:
        List containing [missing_number, duplicate_number], or empty list
        if no corrupt pair found
        
    Time Complexity: O(n) where n is length of input array
    Space Complexity: O(1) as we modify array in-place
    
    Example:
        >>> find_corrupt_pair([3, 1, 2, 3, 6, 4])
        [5, 3]  # 5 is missing, 3 appears twice
    """
    i = 0
    # First pass: Place each number in its correct position using cyclic sort
    while i < len(nums):
        correct_pos = nums[i] - 1  # Correct position for current number
        # If number isn't at its correct position and we can swap
        if correct_pos < len(nums) and nums[i] != nums[correct_pos]:
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
        else:
            i += 1

    # Second pass: Find number that doesn't match its position
    for i in range(len(nums)):
        if (nums[i] - 1) != i:
            return [i + 1, nums[i]]  # Missing number is i+1, duplicate is nums[i]
    return []  # No corrupt pair found

def test_find_corrupt_pair():
    """
    Test cases for finding corrupt pair in an array.
    Tests various scenarios including edge cases.
    
    Test cases cover:
    - Regular cases with missing and duplicate numbers
    - Small arrays with duplicates
    - Arrays with no corrupt pairs
    - Edge cases like all same numbers
    - Various positions of duplicates
    """
    test_cases = [
        ([3, 1, 2, 3, 6, 4], [5, 3]),  # Duplicate 3, Missing 5
        ([3, 1, 2, 5, 2], [4, 2]),     # Duplicate 2, Missing 4
        ([1, 2, 2, 4], [3, 2]),        # Duplicate 2, Missing 3
        ([1, 1], [2, 1]),              # Small array, Duplicate 1
        ([1, 2, 3], []),               # No corrupt pair
        ([2, 2], [1, 2]),              # All same numbers
        ([4, 3, 4, 1], [2, 4]),        # Duplicate 4, Missing 2
    ]

    for i, (nums, expected) in enumerate(test_cases, 1):
        # Create copy of input to preserve original
        test_input = nums.copy()
        result = find_corrupt_pair(test_input)

        print(f"\nTest Case {i}:")
        print(f"Input array: {nums}")
        print(f"Expected corrupt pair [missing, duplicate]: {expected}")
        print(f"Got: {result}")

        assert result == expected, \
            f"Test case {i} failed! Expected {expected}, got {result}"
        print("✓ Passed")

if __name__ == "__main__":
    test_find_corrupt_pair()
    print("\nAll test cases passed!")