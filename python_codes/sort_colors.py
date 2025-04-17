def sort_colors(nums):
    """
    Sort an array containing only 0s, 1s, and 2s in-place.

    This is also known as the Dutch National Flag problem.
    The algorithm uses a three-pointer approach to sort the array in a single pass.

    Args:
        nums: List of integers (only 0s, 1s, and 2s)

    Returns:
        None: The array is sorted in-place

    Time Complexity: O(n) where n is the length of the array
    Space Complexity: O(1) as the sorting is done in-place

    Example:
        >>> nums = [2,0,2,1,1,0]
        >>> sort_colors(nums)
        >>> nums
        [0,0,1,1,2,2]
    """
    # Initialize three pointers
    low = 0  # points to the boundary of 0s (everything before low is 0)
    mid = 0  # current element being examined
    high = len(nums) - 1  # points to the boundary of 2s (everything after high is 2)

    # Continue until the middle pointer crosses the high pointer
    while mid <= high:
        if nums[mid] == 0:
            # Found a 0 - swap with the element at low pointer
            nums[low], nums[mid] = nums[mid], nums[low]
            # Move both pointers forward
            low += 1
            mid += 1
        elif nums[mid] == 1:
            # Found a 1 - it's already in the correct place
            mid += 1
        else:  # nums[mid] == 2
            # Found a 2 - swap with the element at high pointer
            nums[mid], nums[high] = nums[high], nums[mid]
            # Only decrement high pointer, as the swapped element needs to be examined
            high -= 1


def main():
    """
    Driver code to test the sort_colors function with various inputs.
    Tests different distributions of colors in arrays.
    """
    test_cases = [
        [2, 0, 2, 1, 1, 0],  # Mixed colors
        [0, 1, 2, 0, 1, 2],  # Repeating pattern
        [1, 1, 1, 1, 1, 1],  # All 1s
        [2, 2, 2, 0, 0, 0],  # Only 0s and 2s
        [2, 1, 0, 2, 1, 0, 2, 1],  # Longer sequence
    ]

    for i, nums in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Before sorting: {nums}")
        sort_colors(nums)
        print(f"After sorting:  {nums}")
        print("-" * 50)


if __name__ == "__main__":
    main()
