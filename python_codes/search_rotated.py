def search(nums, target):
    """
    Search for target value in a rotated sorted array using modified binary search.
    A rotated array is a sorted array that has been rotated at some pivot point.
    
    Args:
        nums: List of integers representing rotated sorted array
        target: Integer value to search for
        
    Returns:
        int: Index of target if found, -1 otherwise
        
    Time Complexity: O(log n) where n is length of array
    Space Complexity: O(1) using constant extra space
    
    Example:
        >>> search([4,5,6,7,0,1,2], 0)
        4  # Target 0 is at index 4
    """
    low = 0                # Initialize low pointer at start of array
    high = len(nums) - 1   # Initialize high pointer at end of array

    while low <= high:     # Continue search until pointers cross
        mid = low + (high - low) // 2  # Calculate middle index safely to avoid overflow
        
        if nums[mid] == target:
            return mid     # Found target at middle index
            
        elif nums[low] <= nums[mid]:
            # Left half is normally sorted (no rotation point in this half)
            if target < nums[mid] and target >= nums[low]:
                high = mid - 1  # Target is in left sorted half
            else:
                low = mid + 1   # Target is in right half (may be rotated)
        else:
            # Right half is normally sorted (rotation point is in left half)
            if target <= nums[high] and target > nums[mid]:
                low = mid + 1   # Target is in right sorted half
            else:
                high = mid - 1  # Target is in left half (may be rotated)

    return -1  # Target not found in array


# Driver code to test the search function in a rotated sorted array
if __name__ == "__main__":
    """
    Test search function with various rotated array configurations.
    Each test case includes:
    - A rotated sorted array
    - A target value to search for
    - Expected output (index or -1)
    
    Tests cover:
    - Standard rotated arrays
    - Targets present/not present
    - Edge cases (single element, small arrays)
    - Different rotation points
    """
    test_cases = [
        ([4, 5, 6, 7, 0, 1, 2], 0),  # Expected output: 4
        ([4, 5, 6, 7, 0, 1, 2], 3),  # Expected output: -1
        ([1], 0),  # Expected output: -1
        ([1, 3], 3),  # Expected output: 1
        ([5, 1, 3], 5),  # Expected output: 0
    ]

    for nums, target in test_cases:
        result = search(nums, target)
        print(f"Array: {nums}, Target: {target}, Result: {result}")