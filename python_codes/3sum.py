def threesum(nums):
    """
    Return 3 distinct numbers whose sum is equal to 0
    """

    nums.sort()

    left = 0
    right = len(nums) - 1

    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left = i + 1
        right = len(nums) - 1

        while left < right:
            if nums[i] + nums[left] + nums[right] == 0:
                result.append([nums[left], nums[i], nums[right]])
                while left < right and nums[left + 1] == nums[left]:
                    left += 1
                while left < right and nums[right - 1] == nums[right]:
                    right -= 1
                left += 1
                right -= 1
            elif nums[i] + nums[left] + nums[right] > 0:
                right -= 1
            else:
                left += 1

    return result


# Driver code to test the threesum function
if __name__ == "__main__":
    test_cases = [
        ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
        ([0, 0, 0, 0], [[0, 0, 0]]),
        ([], []),
        ([1, 2, -2, -1], []),
        ([-2, 0, 1, 1, 2], [[-2, 0, 2], [-2, 1, 1]]),
    ]

    for nums, expected in test_cases:
        result = threesum(nums)
        print(f"Input: {nums}, Expected: {expected}, Result: {result}")

    print("All tests passed.")
