from typing import List


def productExceptSelf(nums: List[int]) -> List[int]:
    """
    Calculate product of all elements in the array except self at each position.

    This function returns an array where each element at index i is the product of
    all elements in the input array except the element at i.

    Time Complexity: O(n) where n is the length of the input array
    Space Complexity: O(n) for the three arrays created

    Args:
        nums: List of integers

    Returns:
        List[int]: Array where each element is the product of all numbers except self
    """
    # Initialize arrays for left products, right products and final result
    leftproduct = [1] * len(nums)
    rightproduct = [1] * len(nums)
    result = [1] * len(nums)

    # Calculate products of all elements to the left of each position
    for i in range(1, len(nums)):
        leftproduct[i] = leftproduct[i - 1] * nums[i - 1]

    # Calculate products of all elements to the right of each position
    for j in range(len(nums) - 2, -1, -1):
        rightproduct[j] = rightproduct[j + 1] * nums[j + 1]

    # Calculate the final result by multiplying left and right products
    for i in range(len(nums)):
        result[i] = leftproduct[i] * rightproduct[i]

    return result


# Driver code
if __name__ == "__main__":
    # Test cases directly without creating a Solution class

    # Test case 1: Standard example
    nums1 = [1, 2, 3, 4]
    print(f"Input: {nums1}")
    print(f"Output: {productExceptSelf(nums1)}")
    # Expected output: [24, 12, 8, 6]

    # Test case 2: Array with zeros
    nums2 = [0, 1, 2, 3]
    print(f"Input: {nums2}")
    print(f"Output: {productExceptSelf(nums2)}")
    # Expected output: [6, 0, 0, 0]

    # Test case 3: Another example
    nums3 = [-1, 1, 0, -3, 3]
    print(f"Input: {nums3}")
    print(f"Output: {productExceptSelf(nums3)}")
    # Expected output: [0, 0, 9, 0, 0]
