def missing_number(nums):

    index = 0
    while index < len(nums):
        correct_pos = nums[index]
        if correct_pos < len(nums) and correct_pos != index:
            nums[index], nums[correct_pos] = nums[correct_pos], nums[index]
        else:
            index += 1
    for x in range(len(nums)):
        if x != nums[x]:
            return x
    return len(nums)


# ...existing code...


def test_missing_number():
    """
    Test cases for missing_number function
    Each test case contains:
    - Input array
    - Expected missing number
    """
    test_cases = [
        ([3, 0, 1], 2),  # Missing 2
        ([0, 1], 2),  # Missing last number
        ([1, 0], 2),  # Missing last number, different order
        ([9, 6, 4, 2, 3, 5, 7, 0, 1], 8),  # Missing 8
        ([0], 1),  # Single element
    ]

    for i, (nums, expected) in enumerate(test_cases, 1):
        # Create a copy of input to preserve original
        test_input = nums.copy()
        result = missing_number(test_input)

        print(f"\nTest Case {i}:")
        print(f"Input array: {nums}")
        print(f"Expected missing number: {expected}")
        print(f"Got: {result}")

        # Verify result
        assert (
            result == expected
        ), f"Test case {i} failed! Expected {expected}, got {result}"
        print("✓ Passed")


if __name__ == "__main__":
    test_missing_number()
    print("\nAll test cases passed!")
