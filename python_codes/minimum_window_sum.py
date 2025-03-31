def min_sub_array_len(target: int, nums: list[int]) -> int:
    """
    Find length of shortest contiguous subarray with sum >= target.
    Uses sliding window approach with optimization to shrink window.

    Args:
        target: Target sum to reach or exceed
        nums: List of positive integers to search

    Returns:
        int: Length of shortest subarray with sum >= target, 0 if none exists

    Time Complexity: O(n) where n is length of nums
    Space Complexity: O(1) as we only use constant extra space

    Example:
        >>> min_sub_array_len(7, [2,3,1,2,4,3])
        2  # Subarray [4,3] has sum >= 7 with minimum length
    """
    min_length = float("inf")  # Track minimum valid window size
    left = 0  # Left pointer of window

    while left < len(nums):
        sum = 0  # Current window sum
        right = left  # Right pointer starts at left

        # Expand window until sum >= target or end reached
        while sum < target and right < len(nums):
            sum = sum + nums[right]
            right = right + 1

        # If we found a valid window
        if sum >= target:
            temp = left
            # Try to minimize window by removing elements from left
            while temp < right and sum >= target:
                sum = sum - nums[temp]
                temp = temp + 1

            # Update minimum length if current window is smaller
            min_length = min(min_length, right - temp + 1)

        left = left + 1  # Move left pointer to try next window

    return min_length if min_length != float("inf") else 0


def main():
    """
    Driver code to test minimum window sum functionality.
    Tests various scenarios including:
    - Basic cases with clear minimum window
    - Windows at array boundaries
    - No valid window exists
    - Single element windows
    - Entire array as window
    """
    target = [7, 4, 11, 10, 5, 15]  # Different target sums
    input_arr = [
        [2, 3, 1, 2, 4, 3],  # Multiple valid windows
        [1, 4, 4],  # Window at end
        [1, 1, 1, 1, 1, 1, 1, 1],  # Need multiple elements
        [1, 2, 3, 4],  # Sequential numbers
        [1, 2, 1, 3],  # Window in middle
        [5, 4, 9, 8, 11, 3, 7, 12, 15, 44],  # Larger numbers
    ]

    # Process each test case
    for i in range(len(input_arr)):
        window_size = min_sub_array_len(target[i], input_arr[i])
        print(
            i + 1,
            ".\t Input array: ",
            input_arr[i],
            "\n\t Target: ",
            target[i],
            "\n\t Minimum Length of Subarray: ",
            window_size,
            sep="",
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
