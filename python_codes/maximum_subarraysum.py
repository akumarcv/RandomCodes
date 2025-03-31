from typing import List


def maxSubArray(nums: List[int]) -> int:
    """
    Find maximum sum of any contiguous subarray using dynamic programming (Kadane's Algorithm).

    Args:
        nums: List of integers to process

    Returns:
        int: Maximum sum of any contiguous subarray

    Time Complexity: O(n) where n is length of input array
    Space Complexity: O(n) for DP array

    Example:
        >>> maxSubArray([-2,1,-3,4,-1,2,1,-5,4])
        6  # Subarray [4,-1,2,1] has maximum sum
    """
    # Initialize DP array where dp[i] represents max sum ending at index i
    dp = [0 for _ in range(len(nums))]
    dp[0] = nums[0]  # Base case: single element is the sum

    # For each position, either:
    # 1. Start new subarray (take current element)
    # 2. Extend previous subarray (add current to previous sum)
    for i in range(1, len(nums)):
        dp[i] = max(nums[i] + dp[i - 1], nums[i])

    # Print DP array for visualization
    print(dp)

    # Return maximum value found in DP array
    return max(dp)


def test_max_subarray():
    """
    Test cases for finding maximum subarray sum.
    Tests various array configurations including:
    - Single element arrays
    - Arrays with all negative numbers
    - Mixed positive/negative arrays
    - Arrays with all zeros
    - Arrays with alternating values

    For each test case:
    1. Computes maximum sum
    2. Verifies against expected result
    3. Finds actual subarray producing maximum sum
    """
    # Test cases with arrays and their expected maximum sums
    test_cases = [
        ([1], 1),  # Single element
        ([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6),  # Basic case with positive sum
        ([-1], -1),  # Single negative
        ([-2, -1], -1),  # All negative
        ([5, 4, -1, 7, 8], 23),  # All positive except one
        ([1, -1, 1], 1),  # Alternating
        ([-2, -3, -1, -5], -1),  # All negative, return max
        ([0, 0, 0, 0], 0),  # All zeros
    ]

    # Process each test case
    for i, (nums, expected) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input array: {nums}")
        print(f"Expected max sum: {expected}")

        result = maxSubArray(nums)
        print(f"Got max sum: {result}")

        # Verify result matches expected
        assert (
            result == expected
        ), f"Test case {i} failed! Expected {expected}, got {result}"

        # Find and print the actual subarray that gives maximum sum
        if nums:
            max_sum = float("-inf")  # Track maximum sum found
            curr_sum = 0  # Track current sum
            start = 0  # Start index of max subarray
            end = 0  # End index of max subarray
            temp_start = 0  # Temporary start for current sum

            # Scan array to find actual subarray
            for i, num in enumerate(nums):
                curr_sum += num
                if curr_sum > max_sum:
                    max_sum = curr_sum
                    start = temp_start
                    end = i
                if curr_sum < 0:
                    curr_sum = 0
                    temp_start = i + 1

            print(f"Maximum subarray: {nums[start:end+1]}")
        print("✓ Passed")


if __name__ == "__main__":
    test_max_subarray()
    print("\nAll test cases passed!")
