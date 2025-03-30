from typing import List


def maxSubArray(nums: List[int]) -> int:
    dp = [0 for _ in range(len(nums) )]
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        dp[i] = max(nums[i] + dp[i - 1], nums[i])
    print(dp)
    return max(dp)

# ...existing code...

def test_max_subarray():
    """
    Test cases for finding maximum subarray sum
    Each test case contains array and expected maximum sum
    """
    test_cases = [
        ([1], 1),                                # Single element
        ([-2,1,-3,4,-1,2,1,-5,4], 6),           # Basic case with positive sum
        ([-1], -1),                              # Single negative
        ([-2,-1], -1),                           # All negative
        ([5,4,-1,7,8], 23),                     # All positive except one
        ([1,-1,1], 1),                          # Alternating
        ([-2,-3,-1,-5], -1),                    # All negative, return max
        ([0,0,0,0], 0)                          # All zeros
    ]
    
    for i, (nums, expected) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input array: {nums}")
        print(f"Expected max sum: {expected}")
        
        result = maxSubArray(nums)
        print(f"Got max sum: {result}")
        
        assert result == expected, f"Test case {i} failed! Expected {expected}, got {result}"
        
        # Find and print the actual subarray
        if nums:
            max_sum = float('-inf')
            curr_sum = 0
            start = 0
            end = 0
            temp_start = 0
            
            for i, num in enumerate(nums):
                curr_sum += num
                if curr_sum > max_sum:
                    max_sum = curr_sum
                    start = temp_start
                    end = i
                if curr_sum < 0:
                    curr_sum = 0
                    temp_start = i + 1
                    
            print(f"Maximum subarray: {nums[start:end+1]}")
        print("✓ Passed")

if __name__ == "__main__":
    test_max_subarray()
    print("\nAll test cases passed!")