def house_robber(money: list[int]) -> int:
    """
    Find maximum amount that can be robbed from houses arranged in a circle.
    Uses dynamic programming but handles circular arrangement by:
    1. Trying houses from first to second-last
    2. Trying houses from second to last
    Then takes maximum of these two scenarios.

    Args:
        money: List of integers representing money in each house

    Returns:
        int: Maximum amount that can be robbed without alerting police

    Time Complexity: O(n) where n is number of houses
    Space Complexity: O(n) for DP array

    Example:
        >>> house_robber([2,3,2])
        3  # Rob house at index 1
    """
    if len(money) == 0:
        return 0
    if len(money) == 1:
        return money[0]

    return max(rob(money[:-1]), rob(money[1:]))


def rob(money: list[int]) -> int:
    """
    Helper function to find maximum amount that can be robbed from linear arrangement.
    Uses DP where dp[i] represents maximum amount that can be robbed from first i houses.

    Args:
        money: List of integers representing money in each house

    Returns:
        int: Maximum amount that can be robbed

    Time Complexity: O(n) where n is number of houses
    Space Complexity: O(n) for DP array
    """
    # Initialize DP array
    dp = [0] * (len(money) + 1)
    dp[0] = 0  # Base case: no houses
    dp[1] = money[0]  # Base case: one house

    # Fill DP array
    for i in range(2, len(money) + 1):
        # Either rob current house + money from i-2 houses
        # Or skip current house and keep money from i-1 houses
        dp[i] = max(money[i - 1] + dp[i - 2], dp[i - 1])

    return dp[-1]


def main():
    """
    Driver code to test house robber functionality with various inputs.
    Tests different house configurations including:
    - Circular arrangements
    - Different money distributions
    - Edge cases (empty, single house)
    """
    inputs = [
        [2, 3, 2],  # Simple circular case
        [1, 2, 3, 1],  # More houses
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Many houses
        [7, 4, 1, 9, 3],  # Random values
        [],  # Empty case
    ]

    for i, houses in enumerate(inputs, 1):
        print(f"{i}.\tHouses: {houses}")
        loot = house_robber(houses)
        print(f"\tMaximum loot: {loot}")

        # Show which houses could be robbed (for small arrays)
        if len(houses) <= 5 and houses:
            money1 = rob(houses[:-1])  # Try excluding last house
            money2 = rob(houses[1:])  # Try excluding first house
            print(
                f"\tStrategy: {'Rob houses 1 to n-1' if money1 > money2 else 'Rob houses 2 to n'}"
            )
        print("-" * 100)


if __name__ == "__main__":
    main()
