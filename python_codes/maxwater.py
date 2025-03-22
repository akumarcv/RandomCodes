import os, sys, pdb


class Solution:
    def maxArea(self, height):
        max_area = 0
        left = 0
        right = len(height) - 1
        while right > left:
            if height[left] <= height[right]:
                area = (right - left) * height[left]
                left += 1
                if area > max_area:
                    max_area = area

            else:
                area = (right - left) * height[right]
                right -= 1
                if area > max_area:
                    max_area = area

        return max_area


obj = Solution()
input = [1, 8, 6, 2, 5, 4, 8, 3, 7]
maxarea = obj.maxArea(input)
print(maxarea)
