"""
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
"""

def rain_water(nums):
    if not nums:
        return 0, 0
        
    def left_right(nums):
        left_max = float("-inf")
        right_max = float("-inf")
        left = []
        right = [0] * len(nums)  # Initialize right array with zeros
        
        # Calculate left max array
        for i in range(len(nums)):
            left_max = max(nums[i], left_max)
            left.append(left_max)
            
        # Calculate right max array
        for i in range(len(nums)-1, -1, -1):
            right_max = max(nums[i], right_max)
            right[i] = right_max  # Store in correct position
            
        # Calculate trapped water
        water = 0
        for i in range(len(nums)):
            water += min(left[i], right[i]) - nums[i]
        return water
    
    def two_pointer(nums):
        left = 0
        right = len(nums)-1
        water = 0
        lmax, rmax = 0, 0
        while left<right:
            if nums[left]<nums[right]:
                if nums[left]>=lmax:
                    lmax = nums[left]
                else:
                    water = water + (lmax-nums[left])
                left+=1
            else:
                if rmax<=nums[right]:
                    rmax = nums[right]
                else:
                    water = water + (rmax-nums[right])
                right-=1
        return water
    return left_right(nums), two_pointer((nums))


# Driver code
if __name__ == "__main__":
    test_cases = [
        ([0,1,0,2,1,0,1,3,2,1,2,1], 6),  # Classic example
        ([4,2,0,3,2,5], 9),               # Another example
        ([], 0),                          # Empty array
        ([1], 0),                         # Single element
        ([1,1,1,1], 0),                  # No water can be trapped
        ([5,4,3,2,1], 0),                # Decreasing array
        ([1,2,3,4,5], 0),                # Increasing array
        ([2,0,2], 2)                     # Simple valley
    ]
    
    for i, (heights, expected) in enumerate(test_cases, 1):
        array_result, two_pointer_result = rain_water(heights)
        
        print(f"\nTest case {i}:")
        print(f"Heights: {heights}")
        print(f"Expected: {expected}")
        print(f"Array method result: {array_result}")
        print(f"Two pointer method result: {two_pointer_result}")
        
        # Verify both implementations give correct results
        assert array_result == expected, (
            f"Array method failed! Expected {expected}, got {array_result}"
        )
        assert two_pointer_result == expected, (
            f"Two pointer method failed! Expected {expected}, got {two_pointer_result}"
        )
        print("✓ Both implementations passed")
    
    print("\nAll test cases passed successfully!")