class Solution:
    """
    Solution class for finding median of two sorted arrays.
    Uses min heap approach to merge and find median efficiently.
    """

    def findMedianSortedArrays(self, nums1, nums2):
        """
        Find median of two sorted arrays using heap-based merge.
        
        Args:
            nums1: First sorted array of integers
            nums2: Second sorted array of integers
            
        Returns:
            float: Median value of merged arrays
            
        Time Complexity: O((m+n)log(m+n)) where m,n are array lengths
        Space Complexity: O(m+n) for storing merged array
        
        Example:
            >>> Solution().findMedianSortedArrays([1,3], [2])
            2.0  # Median of merged array [1,2,3]
        """
        # Handle case when first array is empty
        if not nums1:
            finarr = nums2
            median = (
                finarr[len(finarr) // 2]
                if len(finarr) % 2 != 0
                else (finarr[len(finarr) // 2 - 1] + finarr[len(finarr) // 2]) / 2
            )
            return median
            
        # Handle case when second array is empty
        if not nums2:
            finarr = nums1
            median = (
                finarr[len(finarr) // 2]
                if len(finarr) % 2 != 0
                else (finarr[len(finarr) // 2] + finarr[(len(finarr) // 2) - 1]) / 2
            )
            return median

        # Create and populate min heap
        arr = self.pusharray(nums1 + nums2)
        finarr = []
        
        # Extract elements in sorted order
        for j in range(1, len(arr)):
            val = self.minpop(arr)
            finarr.append(val)
            
        # Calculate median based on array length
        median = (
            finarr[len(finarr) // 2]
            if len(finarr) % 2 != 0
            else (finarr[len(finarr) // 2] + finarr[(len(finarr) // 2) - 1]) / 2
        )
        return float(median)

    def pusharray(self, nums):
        """
        Create min heap from input array.
        
        Args:
            nums: Array of numbers to heapify
            
        Returns:
            list: Min heap array with sentinel at index 0
        """
        arr = [0]  # Sentinel value for 1-based indexing
        for i in nums:
            arr.append(i)
            self.floatup(arr, len(arr) - 1)
        return arr

    def swap(self, arr, i, j):
        """
        Swap elements at given indices in array.
        
        Args:
            arr: Input array
            i, j: Indices of elements to swap
        """
        arr[i], arr[j] = arr[j], arr[i]

    def floatup(self, arr, index):
        """
        Float element up in min heap to maintain heap property.
        
        Args:
            arr: Heap array
            index: Index of element to float up
        """
        parent = index // 2

        if index < 1:
            return
        elif arr[parent] > arr[index]:
            self.swap(arr, parent, index)
            self.floatup(arr, parent)

    def minpop(self, arr):
        """
        Remove and return minimum element from heap.
        
        Args:
            arr: Heap array
            
        Returns:
            int: Minimum value in heap, or False if heap is empty
        """
        if len(arr) == 1:
            return False
        else:
            self.swap(arr, 1, len(arr) - 1)
            minval = arr.pop()
            self.floatdown(arr, 1)
            return minval

    def floatdown(self, arr, index):
        """
        Float element down in min heap to maintain heap property.
        
        Args:
            arr: Heap array
            index: Index of element to float down
        """
        left = index * 2
        right = index * 2 + 1
        smallest = index
        if len(arr) > left and arr[left] < arr[index]:
            smallest = left
        if len(arr) > right and arr[right] < arr[smallest]:
            smallest = right
        if smallest != index:
            self.swap(arr, smallest, index)
            self.floatdown(arr, smallest)


# Test code
arr1 = [2, 3, 4]
arr2 = [1]

obje = Solution()
median = obje.findMedianSortedArrays(arr1, arr2)
print(median)