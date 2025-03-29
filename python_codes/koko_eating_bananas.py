import math
from typing import List


class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        min_speed = 1
        max_speed = max(piles)

        while min_speed < max_speed:
            current_speed = (min_speed + max_speed) // 2
            hours = 0
            for p in piles:
                hours += math.ceil(p / current_speed)
            if hours <= h:
                max_speed = current_speed
            else:
                min_speed = current_speed + 1
        return max_speed


# ...existing code...


def test_min_eating_speed():
    """
    Test cases for finding minimum eating speed for Koko
    Each test case contains piles of bananas and hours available
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
