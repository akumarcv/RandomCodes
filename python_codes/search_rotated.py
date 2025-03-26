def search(nums, target):

    low = 0
    high = len(nums) - 1

    while low <= high:
        mid = low + (high - low) // 2
        if nums[mid] == target:
            return mid
        elif nums[low] <= nums[mid]:
            if target < nums[mid] and target >= nums[low]:
                high = mid - 1
            else:
                low = mid + 1
        else:
            if target <= nums[high] and target > nums[mid]:
                low = mid + 1
            else:
                high = mid - 1

    return -1


# driver code for search in rotated array.
# Driver code to test the search function in a rotated sorted array
if __name__ == "__main__":
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
