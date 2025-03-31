def merge_sorted(nums1: list[int], m: int, nums2: list[int], n: int) -> list[int]:
    """
    Merge two sorted arrays in-place into the first array.
    Uses two-pointer technique starting from end to avoid overwriting.

    Args:
        nums1: First sorted array with extra space at end
        m: Number of actual elements in nums1
        nums2: Second sorted array to merge
        n: Number of elements in nums2

    Returns:
        list[int]: Modified nums1 containing merged sorted arrays

    Time Complexity: O(m+n) where m,n are input array lengths
    Space Complexity: O(1) as we merge in-place

    Example:
        >>> merge_sorted([1,2,3,0,0], 3, [2,5], 2)
        [1,2,2,3,5]  # Merged array in sorted order
    """
    # Initialize pointers to last valid elements and last position
    p1 = m - 1  # Last element in nums1
    p2 = n - 1  # Last element in nums2
    p = len(nums1) - 1  # Last position in merged array

    # Merge arrays from end to avoid overwriting
    while p >= 0:
        if p2 >= 0:  # Still have elements in nums2 to process
            if p1 >= 0 and nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]  # Take larger element from nums1
                p1 -= 1
                p -= 1
            else:
                nums1[p] = nums2[p2]  # Take element from nums2
                p2 -= 1
                p -= 1
        else:
            break  # nums2 is fully processed

    return nums1


# Driver code
def main():
    """
    Test function for merge sorted arrays functionality.
    Tests various scenarios including:
    - Arrays of different sizes
    - Arrays with duplicates
    - Arrays with negative numbers
    - Edge cases with empty arrays
    """
    m = [9, 2, 3, 1, 8]  # Number of elements in nums1
    n = [6, 1, 4, 2, 1]  # Number of elements in nums2
    nums1 = [
        [23, 33, 35, 41, 44, 47, 56, 91, 105, 0, 0, 0, 0, 0, 0],  # Regular case
        [1, 2, 0],  # Small arrays
        [1, 1, 1, 0, 0, 0, 0],  # Duplicates
        [6, 0, 0],  # Single element
        [12, 34, 45, 56, 67, 78, 89, 99, 0],  # Sorted input
    ]
    nums2 = [
        [32, 49, 50, 51, 61, 99],  # Multiple elements
        [7],  # Single element
        [1, 2, 3, 4],  # Sequential numbers
        [-99, -45],  # Negative numbers
        [100],  # Single large number
    ]

    # Process each test case
    k = 1
    for i in range(len(m)):
        print(f"{k}.\tnums1: {nums1[i]}, m: {m[i]}")
        print(f"\tnums2: {nums2[i]}, n: {n[i]}")
        print(f"\n\tMerged list: {merge_sorted(nums1[i], m[i], nums2[i], n[i])}")
        print("-" * 100, "\n")
        k += 1


if __name__ == "__main__":
    main()
