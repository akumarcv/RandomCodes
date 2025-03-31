from typing import List


def find_max_knapsack_profit(
    capacity: int, weights: List[int], values: List[int]
) -> int:
    """
    Solve 0/1 Knapsack problem using dynamic programming.
    For each item, we either include it or exclude it based on maximum profit.

    Args:
        capacity: Maximum weight capacity of knapsack
        weights: List of weights for each item
        values: List of values/profits for each item

    Returns:
        int: Maximum profit achievable within weight capacity

    Time Complexity: O(n*W) where n is number of items and W is capacity
    Space Complexity: O(n*W) for DP table

    Example:
        >>> find_max_knapsack_profit(6, [1,2,3,5], [1,5,4,8])
        13  # Select items with weights [1,2,3] and values [1,5,4]
    """
    # Initialize DP table with dimensions [items+1][capacity+1]
    dp = [[0 for _ in range(capacity + 1)] for _ in range(len(values) + 1)]

    # Fill DP table: For each item and each possible capacity
    for i in range(1, len(values) + 1):
        for j in range(1, capacity + 1):
            # If current item can fit in knapsack
            if weights[i - 1] <= j:
                # Max of: including current item or excluding it
                dp[i][j] = max(
                    values[i - 1] + dp[i - 1][j - weights[i - 1]],  # Include item
                    dp[i - 1][j],  # Exclude item
                )
            else:
                # Can't include current item, take previous best
                dp[i][j] = dp[i - 1][j]

    return dp[-1][-1]  # Return maximum profit


def main():
    """
    Driver code to test knapsack implementation with various inputs.
    Tests different scenarios including:
    - Multiple items with varying weights/values
    - Single item cases
    - Different capacities
    - Large scale problems (commented out)
    """
    # Test cases with different weights, values, and capacities
    weights = [
        [1, 2, 3, 5],  # Regular case
        [4],  # Single item
        [2],  # Small weight
        [3, 6, 10, 7, 2],  # Multiple items
        [3, 6, 10, 7, 2, 12, 15, 10, 13, 20],  # Many items
    ]
    values = [
        [1, 5, 4, 8],  # Regular case
        [2],  # Single value
        [3],  # Small value
        [12, 10, 15, 17, 13],  # Multiple values
        [12, 10, 15, 17, 13, 12, 30, 15, 18, 20],  # Many values
    ]
    capacity = [6, 3, 3, 10, 20]  # Different capacities

    # Process each test case
    for i in range(len(values)):
        print(
            f"{i + 1}. We have a knapsack of capacity {capacity[i]} "
            "and we are given the following list of item values and weights:"
        )
        print("-" * 30)
        print("{:<10}{:<5}{:<5}".format("Weights", "|", "Values"))
        print("-" * 30)

        # Print item details
        for j in range(len(values[i])):
            print("{:<10}{:<5}{:<5}".format(weights[i][j], "|", values[i][j]))

        # Calculate and print result
        result = find_max_knapsack_profit(capacity[i], weights[i], values[i])
        print(f"\nThe maximum profit achievable is: {result}")
        print("-" * 100, "\n")


if __name__ == "__main__":
    main()
