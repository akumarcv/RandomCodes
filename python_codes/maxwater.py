import os, sys, pdb


class Solution:
    """
    Solution class for container with most water problem.
    Uses two-pointer technique to find maximum area between heights.
    """
    
    def maxArea(self, height: list[int]) -> int:
        """
        Find container that can hold maximum amount of water between vertical lines.
        Uses two pointers moving inward from both ends to find optimal area.
        
        Args:
            height: List of integers where each value represents height of vertical line
            
        Returns:
            int: Maximum area that can be contained between any two lines
            
        Time Complexity: O(n) where n is length of height array
        Space Complexity: O(1) as we only use constant extra space
        
        Example:
            >>> Solution().maxArea([1,8,6,2,5,4,8,3,7])
            49  # Between heights 8 and 7 with width 7
        """
        max_area = 0        # Track maximum area found
        left = 0            # Left pointer starts at beginning
        right = len(height) - 1  # Right pointer starts at end
        
        while right > left:
            # Calculate area using shorter height (bottleneck for water)
            if height[left] <= height[right]:
                # Area = width * height = (right-left) * min(height[left], height[right])
                area = (right - left) * height[left]
                left += 1   # Move left pointer inward
                if area > max_area:
                    max_area = area
            else:
                area = (right - left) * height[right]
                right -= 1  # Move right pointer inward
                if area > max_area:
                    max_area = area

        return max_area


# Driver code to test maxArea function
if __name__ == "__main__":
    """
    Test cases for container with most water.
    Tests various height configurations including:
    - Basic cases with clear maximum
    - Equal heights
    - Increasing/decreasing heights
    - Single pair of heights
    """
    obj = Solution()
    input = [1, 8, 6, 2, 5, 4, 8, 3, 7]  # Example test case
    maxarea = obj.maxArea(input)
    print(f"Input heights: {input}")
    print(f"Maximum water area: {maxarea}")