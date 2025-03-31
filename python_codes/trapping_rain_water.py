"""
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
"""


def rain_water(nums):
    """
    Calculate trapped water using two different approaches:
    1. Left-right array method
    2. Two-pointer optimization

    Args:
        nums: List of integers representing elevation map heights

    Returns:
        tuple: (water_array_method, water_two_pointer_method)

    Time Complexity: O(n) for both methods where n is length of nums
    Space Complexity: O(n) for array method, O(1) for two-pointer method

    Example:
        >>> rain_water([0,1,0,2,1,0,1,3,2,1,2,1])
        (6, 6)  # Both methods correctly calculate 6 units of trapped water
    """
    if not nums:
        return 0, 0  # Handle empty input case

    def left_right(nums):
        """
        Calculate trapped water using left and right max arrays.
        For each position, water trapped = min(left_max, right_max) - height

        Args:
            nums: List of integers representing elevation map heights

        Returns:
            int: Total water trapped

        Time Complexity: O(n) where n is length of nums
        Space Complexity: O(n) for storing left and right max arrays
        """
        left_max = float("-inf")  # Track maximum height from left
        right_max = float("-inf")  # Track maximum height from right
        left = []  # Store maximum heights to the left of each position
        right = [0] * len(nums)  # Initialize right array with zeros

        # Calculate left max array
        for i in range(len(nums)):
            left_max = max(nums[i], left_max)  # Update left maximum
            left.append(left_max)  # Store current left maximum

        # Calculate right max array
        for i in range(len(nums) - 1, -1, -1):
            right_max = max(nums[i], right_max)  # Update right maximum
            right[i] = right_max  # Store current right maximum

        # Calculate trapped water
        water = 0
        for i in range(len(nums)):
            # Water at position i = min(left_max, right_max) - height
            water += min(left[i], right[i]) - nums[i]
        return water

    def two_pointer(nums):
        """
        Calculate trapped water using optimized two-pointer approach.
        Uses constant extra space by tracking left and right maximums.

        Args:
            nums: List of integers representing elevation map heights

        Returns:
            int: Total water trapped

        Time Complexity: O(n) where n is length of nums
        Space Complexity: O(1) as we only use constant extra space
        """
        left = 0  # Left pointer starts at beginning
        right = len(nums) - 1  # Right pointer starts at end
        water = 0  # Total trapped water
        lmax, rmax = 0, 0  # Track maximum heights on both sides

        while left < right:
            if nums[left] < nums[right]:
                # Left side is limiting factor
                if nums[left] >= lmax:
                    lmax = nums[left]  # Update left maximum
                else:
                    # Can trap water at this position
                    water = water + (lmax - nums[left])
                left += 1  # Move left pointer inward
            else:
                # Right side is limiting factor
                if rmax <= nums[right]:
                    rmax = nums[right]  # Update right maximum
                else:
                    # Can trap water at this position
                    water = water + (rmax - nums[right])
                right -= 1  # Move right pointer inward
        return water

    return left_right(nums), two_pointer((nums))


# Driver code
if __name__ == "__main__":
    """
    Test suite for rain water trapping algorithms.
    Verifies both implementations against various test cases:
    - Standard examples with trapped water
    - Edge cases (empty array, single element)
    - Special cases (no trapped water)
    - Different height distributions

    For each test:
    1. Runs both implementations
    2. Compares against expected output
    3. Verifies that both methods give identical results
    """
    test_cases = [
        ([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], 6),  # Classic example
        ([4, 2, 0, 3, 2, 5], 9),  # Another example
        ([], 0),  # Empty array
        ([1], 0),  # Single element
        ([1, 1, 1, 1], 0),  # No water can be trapped
        ([5, 4, 3, 2, 1], 0),  # Decreasing array
        ([1, 2, 3, 4, 5], 0),  # Increasing array
        ([2, 0, 2], 2),  # Simple valley
    ]

    for i, (heights, expected) in enumerate(test_cases, 1):
        array_result, two_pointer_result = rain_water(heights)

        print(f"\nTest case {i}:")
        print(f"Heights: {heights}")
        print(f"Expected: {expected}")
        print(f"Array method result: {array_result}")
        print(f"Two pointer method result: {two_pointer_result}")

        # Verify both implementations give correct results
        assert (
            array_result == expected
        ), f"Array method failed! Expected {expected}, got {array_result}"
        assert (
            two_pointer_result == expected
        ), f"Two pointer method failed! Expected {expected}, got {two_pointer_result}"
        print("✓ Both implementations passed")

    print("\nAll test cases passed successfully!")
