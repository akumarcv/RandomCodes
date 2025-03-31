import math
from typing import List


class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        """
        Find minimum eating speed k where Koko can eat all bananas within h hours.
        Uses binary search to efficiently find the minimum valid speed.

        Args:
            piles: List of integers where piles[i] is number of bananas in pile i
            h: Hours available to eat all bananas

        Returns:
            int: Minimum eating speed (bananas per hour) needed

        Time Complexity: O(n * log(m)) where n is number of piles and m is max pile size
        Space Complexity: O(1) as we only use constant extra space

        Example:
            >>> Solution().minEatingSpeed([3,6,7,11], 8)
            4  # Can eat all bananas in 8 hours at speed 4
        """
        # Initialize binary search bounds
        min_speed = 1  # Minimum possible speed
        max_speed = max(piles)  # Maximum needed speed

        # Binary search for minimum valid speed
        while min_speed < max_speed:
            # Try middle speed
            current_speed = (min_speed + max_speed) // 2
            hours = 0

            # Calculate hours needed at current speed
            for p in piles:
                hours += math.ceil(p / current_speed)  # Round up partial hours

            # Adjust search space based on hours needed
            if hours <= h:
                max_speed = current_speed  # Speed might be too high
            else:
                min_speed = current_speed + 1  # Speed too low

        return max_speed


def test_min_eating_speed():
    """
    Test cases for finding minimum eating speed for Koko.
    Tests various scenarios including:
    - Basic cases with multiple piles
    - Large numbers in piles
    - Different time constraints
    - Edge cases (single pile, uniform piles)
    """
    test_cases = [
        ([3, 6, 7, 11], 8, 4),  # Basic case
        ([30, 11, 23, 4, 20], 5, 30),  # Larger numbers
        ([30, 11, 23, 4, 20], 6, 23),  # Same piles, different hours
        ([1], 1, 1),  # Single pile
        ([312884470], 312884469, 2),  # Large pile
        ([1, 1, 1, 1], 4, 1),  # Uniform piles
    ]

    solution = Solution()
    for i, (piles, hours, expected) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Piles: {piles}")
        print(f"Hours available: {hours}")
        print(f"Expected minimum speed: {expected}")

        result = solution.minEatingSpeed(piles, hours)
        print(f"Got minimum speed: {result}")

        assert (
            result == expected
        ), f"Test case {i} failed! Expected {expected}, got {result}"
        print("✓ Passed")


if __name__ == "__main__":
    test_min_eating_speed()
    print("\nAll test cases passed!")
