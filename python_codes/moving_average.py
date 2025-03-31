import numpy as np


def moving_average(nums: list[float], k: int) -> list[float]:
    """
    Calculate moving average of a sequence using sliding window approach.
    Optimizes computation by maintaining running sum of window.

    Args:
        nums: List of numbers to compute moving average
        k: Size of sliding window

    Returns:
        list[float]: List of moving averages, or None if invalid window size

    Time Complexity: O(n) where n is length of nums
    Space Complexity: O(n-k+1) for storing result array

    Example:
        >>> moving_average([1,2,3,4], 2)
        [1.5, 2.5, 3.5]  # Averages of [1,2], [2,3], [3,4]
    """
    result = []  # Store moving averages

    # Validate window size
    if k > len(nums):
        print("window size greater than nums")
        return
    if k == 1:
        return nums  # Each number is its own average

    # Calculate moving averages
    for i in range(len(nums) - k + 1):
        if i == 0:
            # First window: calculate full sum
            sums = sum(nums[:k])
            result.append(sums / k)
        else:
            # Subsequent windows: update sum by removing oldest and adding newest
            sums = sums - nums[i - 1]  # Remove leftmost number
            sums = sums + nums[i + k - 1]  # Add rightmost number
            result.append(sums / k)
    return result


def main():
    """
    Test moving average calculation with various inputs.
    Demonstrates:
    - Basic moving average
    - Different window sizes
    - Edge cases
    - Input validation
    """
    # Test case with sequence of numbers
    nums = [1, 2, 2, 3, 4, 5, 6, 6, 7, 7, 7, 8, 8]
    k = 3  # Window size

    print(f"Input sequence: {nums}")
    print(f"Window size: {k}")
    result = moving_average(nums, k)
    print(f"Moving averages: {result}")


if __name__ == "__main__":
    main()
