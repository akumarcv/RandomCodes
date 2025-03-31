def max_product(nums: list[int]) -> int:
    """
    Find maximum product of contiguous subarray within given array.
    Uses Kadane's algorithm variant tracking both max and min products
    to handle negative numbers.

    Args:
        nums: List of integers to process

    Returns:
        int: Maximum product of any contiguous subarray

    Time Complexity: O(n) where n is length of input array
    Space Complexity: O(1) as we only use constant extra space

    Example:
        >>> max_product([2,3,-2,4])
        6  # Subarray [2,3] has maximum product
    """
    # Initialize with first element for both max and min products
    max_so_far = nums[0]  # Track maximum product ending at current position
    min_so_far = nums[0]  # Track minimum product ending at current position
    result = max_so_far  # Store overall maximum product found

    # Process rest of array maintaining both max and min products
    for i in range(1, len(nums)):
        # Store previous max as it will be needed for min calculation
        previos_max = max_so_far

        # Update max product considering:
        # 1. Current number alone
        # 2. Product with previous max (for positive numbers)
        # 3. Product with previous min (for negative numbers)
        max_so_far = max(nums[i], max_so_far * nums[i], min_so_far * nums[i])

        # Update min product similarly
        min_so_far = min(nums[i], previos_max * nums[i], min_so_far * nums[i])

        # Update global maximum if needed
        result = max(result, max_so_far)
    return result


# Driver code
def main():
    """
    Test function for maximum product subarray.
    Tests various array configurations including:
    - Arrays with negative numbers
    - Arrays with zeros
    - All positive arrays
    - Mixed positive/negative arrays
    """
    input_bits = [
        [-2, 0, -1],  # Array with zero
        [2, 3, -2, 4],  # Basic case
        [2, -5, 3, 1, -4, 0, -10, 2],  # Complex case with negatives and zero
        [1, 2, 3, 0, 4],  # Array ending in zero
        [5, 4, 3, 10, 4, 1],  # All positive numbers
    ]

    for i in range(len(input_bits)):
        print(i + 1, ".\t Input array: ", input_bits[i], sep="")
        print("\n\t Maximum product: ", max_product(input_bits[i]), sep="")
        print("-" * 100)


if __name__ == "__main__":
    main()
