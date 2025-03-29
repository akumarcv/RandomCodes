def find_corrupt_pair(nums):

    i = 0

    while i < len(nums):
        correct_pos = nums[i] - 1
        if correct_pos < len(nums) and nums[i] != nums[correct_pos]:
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]

        else:
            i += 1

    for i in range(len(nums)):
        if (nums[i] - 1) != i:
            return [i + 1, nums[i]]
    return []


# ...existing code...


def test_find_corrupt_pair():
    """
    Test cases for finding corrupt pair in an array
    Tests various scenarios including edge cases
    """
    test_cases = [
        ([3, 1, 2, 3, 6, 4], [5, 3]),  # Duplicate 3, Missing 5
        ([3, 1, 2, 5, 2], [4, 2]),  # Duplicate 2, Missing 4
        ([1, 2, 2, 4], [3, 2]),  # Duplicate 2, Missing 3
        ([1, 1], [2, 1]),  # Small array, Duplicate 1
        ([1, 2, 3], []),  # No corrupt pair
        ([2, 2], [1, 2]),  # All same numbers
        ([4, 3, 4, 1], [2, 4]),  # Duplicate 4, Missing 2
    ]

    for i, (nums, expected) in enumerate(test_cases, 1):
        # Create copy of input to preserve original
        test_input = nums.copy()
        result = find_corrupt_pair(test_input)

        print(f"\nTest Case {i}:")
        print(f"Input array: {nums}")
        print(f"Expected corrupt pair [missing, duplicate]: {expected}")
        print(f"Got: {result}")

        assert (
            result == expected
        ), f"Test case {i} failed! Expected {expected}, got {result}"
        print("✓ Passed")


if __name__ == "__main__":
    test_find_corrupt_pair()
    print("\nAll test cases passed!")
