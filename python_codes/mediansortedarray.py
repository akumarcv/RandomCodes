import heapq


class Solution:
    """
    Solution class for finding median of two sorted arrays.
    Uses two heaps approach to find median efficiently.
    """

    def findMedianSortedArrays(self, nums1, nums2):
        """
        Find median of two sorted arrays using two heaps.

        Args:
            nums1: First sorted array of integers
            nums2: Second sorted array of integers

        Returns:
            float: Median value of merged arrays

        Time Complexity: O((m+n)log(m+n)) where m,n are array lengths
        Space Complexity: O(m+n) for storing elements in heaps

        Example:
            >>> Solution().findMedianSortedArrays([1,3], [2])
            2.0  # Median of merged array [1,2,3]
        """
        # Edge cases
        if not nums1:
            nums = nums2
        elif not nums2:
            nums = nums1
        else:
            nums = nums1 + nums2

        # Initialize heaps
        max_heap = []  # Max heap for smaller half (negate values to create max heap)
        min_heap = []  # Min heap for larger half

        # Process all numbers
        for num in nums:
            # First push to max heap
            heapq.heappush(max_heap, -num)

            # Balance: move largest element from max_heap to min_heap
            heapq.heappush(min_heap, -heapq.heappop(max_heap))

            # Ensure max_heap has equal or one more element than min_heap
            if len(min_heap) > len(max_heap):
                heapq.heappush(max_heap, -heapq.heappop(min_heap))

        # Calculate median
        if len(max_heap) > len(min_heap):
            # Odd number of elements
            return float(-max_heap[0])
        else:
            # Even number of elements
            return float((-max_heap[0] + min_heap[0]) / 2)


# Test code
arr1 = [2, 3, 4]
arr2 = [1]

obje = Solution()
median = obje.findMedianSortedArrays(arr1, arr2)
print(median)
