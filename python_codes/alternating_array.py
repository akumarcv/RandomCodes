"""
You are given a 0-indexed array nums consisting of n positive integers.

The array nums is called alternating if:

nums[i - 2] == nums[i], where 2 <= i <= n - 1.
nums[i - 1] != nums[i], where 1 <= i <= n - 1.
In one operation, you can choose an index i and change nums[i] into any positive integer.

Return the minimum number of operations required to make the array alternating.
"""

"""
Detailed solution approach:

1. UNDERSTANDING THE PATTERN:
   - In an alternating array, all elements at even indices (0,2,4,...) must be equal
   - All elements at odd indices (1,3,5,...) must be equal
   - The values at even indices must be different from values at odd indices

2. KEY INSIGHT:
   - We need to find the most frequent number at even positions and most frequent at odd positions
   - If these two numbers are different, we can keep them and change all other numbers
   - If they're the same, we need to use the second most frequent number for one of the positions

3. ALGORITHM:
   - Count frequencies of numbers at even and odd positions separately
   - Find the most common number (and its count) for even positions
   - Find the most common number (and its count) for odd positions
   - If most common numbers at even and odd positions are different:
     * Change all other numbers at even positions to the most common even number
     * Change all other numbers at odd positions to the most common odd number
   - If most common numbers are the same:
     * Option 1: Keep most common at even positions, use second most common at odd positions
     * Option 2: Keep most common at odd positions, use second most common at even positions
     * Choose the option that requires fewer changes

4. EDGE CASES:
   - Arrays with 0 or 1 elements are already alternating (return 0)
   - For arrays with length 2, just check if elements are different
"""

from collections import Counter


def minimumOperations(nums):
    n = len(nums)

    # Handle edge cases
    if n <= 1:
        return 0
    if n == 2:
        return 1 if nums[0] == nums[1] else 0

    # Count frequencies at even and odd positions
    even_counts = Counter()
    odd_counts = Counter()

    for i, num in enumerate(nums):
        if i % 2 == 0:
            even_counts[num] += 1
        else:
            odd_counts[num] += 1

    # Find most common elements and their counts
    even_most_common = even_counts.most_common(2)
    odd_most_common = odd_counts.most_common(2)

    # Calculate number of elements at even and odd positions
    even_count = (n + 1) // 2  # Ceiling division for even positions (0,2,4...)
    odd_count = n // 2  # Floor division for odd positions (1,3,5...)

    # If there's no most common element, handle the empty case
    even_first = (0, 0) if not even_most_common else even_most_common[0]
    even_second = (0, 0) if len(even_most_common) < 2 else even_most_common[1]

    odd_first = (0, 0) if not odd_most_common else odd_most_common[0]
    odd_second = (0, 0) if len(odd_most_common) < 2 else odd_most_common[1]

    # If most common values are different, we can use them both
    if even_first[0] != odd_first[0]:
        return (even_count - even_first[1]) + (odd_count - odd_first[1])

    # Most common values are the same, try both alternatives
    # Option 1: Use most common at even, second most common at odd
    operations1 = (even_count - even_first[1]) + (odd_count - odd_second[1])

    # Option 2: Use second most common at even, most common at odd
    operations2 = (even_count - even_second[1]) + (odd_count - odd_first[1])

    return min(operations1, operations2)


# Driver code
if __name__ == "__main__":
    # Test cases
    test_cases = [
        [3, 1, 3, 2, 4, 3],  # Should return 3
        [1, 2, 3, 4, 5],  # Should return 3
        [1, 1, 1, 1, 1],  # Should return 2
        [1, 2, 1, 2, 1, 2],  # Should return 0 (already alternating)
        [1, 2, 3, 4],  # Should return 2
        [1, 2],  # Should return 0 (already alternating)
        [5, 5],  # Should return 1
        [7],  # Should return 0 (single element)
        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Should return 5
        [10, 20, 10, 20, 30, 40, 10, 20],  # Should return 4
    ]

    for i, test in enumerate(test_cases):
        result = minimumOperations(test)
        print(f"Test {i+1}: {test}")
        print(f"Minimum operations required: {result}")
        print("=" * 50)
