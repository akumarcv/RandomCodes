"""
There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.

You are giving candies to these children subjected to the following requirements:

Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
Return the minimum number of candies you need to have to distribute the candies to the children.


"""

from typing import List


def candy(ratings: List[int]) -> int:
    """
    Calculate the minimum number of candies required to distribute to children.

    Args:
        ratings (List[int]): List of ratings for each child

    Returns:
        int: Minimum number of candies needed

    Time Complexity: O(n) where n is the number of children
    Space Complexity: O(n) for the candies array

    Examples:
        >>> candy([1,0,2])
        5
        >>> candy([1,2,2])
        4
    """
    # Initialize with 1 candy per child
    candies = [1] * len(ratings)

    # First pass: left to right, ensure children with higher rating than left neighbor get more candies
    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    # Initialize sum with the last child's candies
    sum = candies[-1]

    # Second pass: right to left, ensure children with higher rating than right neighbor get more candies
    for i in range(len(ratings) - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
        sum += candies[i]

    return sum


# Driver code
if __name__ == "__main__":
    test_cases = [
        [1, 0, 2],
        [1, 2, 2],
        [1, 3, 4, 5, 2],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1],
    ]

    for i, ratings in enumerate(test_cases):
        result = candy(ratings)
        print(f"Test case {i+1}: {ratings}")
        print(f"Minimum candies needed: {result}")
        print("-" * 40)
