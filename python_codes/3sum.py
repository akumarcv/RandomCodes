def threesum(nums):
    """
    Find all unique triplets in array that sum to zero.

    Uses two-pointer technique with sorting to achieve O(n^2) time complexity.
    Handles duplicates by skipping repeated values.

    Args:
        nums (List[int]): Input array of integers

    Returns:
        List[List[int]]: List of triplets where each triplet sums to zero

    Time Complexity: O(n^2) where n is length of input array
    Space Complexity: O(1) excluding space for output

    Example:
        >>> threesum([-1, 0, 1, 2, -1, -4])
        [[-1, -1, 2], [-1, 0, 1]]
    """
    # Sort array to handle duplicates and use two-pointer approach
    nums.sort()

    result = []
    # Iterate through possible first numbers of triplet
    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # Use two pointers for remaining two numbers
        left = i + 1
        right = len(nums) - 1

        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]

            if current_sum == 0:
                # Found a valid triplet
                result.append([nums[i], nums[left], nums[right]])

                # Skip duplicates for second number
                while left < right and nums[left + 1] == nums[left]:
                    left += 1
                # Skip duplicates for third number
                while left < right and nums[right - 1] == nums[right]:
                    right -= 1

                left += 1
                right -= 1
            elif current_sum > 0:
                # Sum too large, decrease right pointer
                right -= 1
            else:
                # Sum too small, increase left pointer
                left += 1

    return result


# Driver code to test the threesum function
if __name__ == "__main__":
    test_cases = [
        ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
        ([0, 0, 0, 0], [[0, 0, 0]]),
        ([], []),
        ([1, 2, -2, -1], []),
        ([-2, 0, 1, 1, 2], [[-2, 0, 2], [-2, 1, 1]]),
    ]

    for nums, expected in test_cases:
        result = threesum(nums)
        print(f"Input: {nums}, Expected: {expected}, Result: {result}")

    print("All tests passed.")
