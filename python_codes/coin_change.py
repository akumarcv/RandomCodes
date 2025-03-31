def coin_change(coins: list[int], total: int) -> int:
    """
    Find minimum number of coins needed to make given total using dynamic programming.

    Uses bottom-up DP approach with a 2D table where:
    dp[i][j] = minimum coins needed using first i coins to make amount j

    Args:
        coins: List of coin denominations available
        total: Target amount to make

    Returns:
        int: Minimum number of coins needed, or -1 if impossible

    Time Complexity: O(n*m) where n is number of coins and m is total amount
    Space Complexity: O(n*m) for DP table

    Example:
        >>> coin_change([1,2,5], 11)
        3  # Use 5 + 5 + 1 = 11
    """
    # Base case: no coins needed for zero total
    if total == 0:
        return 0

    # Initialize DP table with infinity
    # dp[i][j] represents min coins needed using first i coins to make amount j
    dp = [[float("inf") for _ in range(total + 1)] for _ in range(len(coins) + 1)]

    # Initialize base case: 0 coins needed for amount 0
    for i in range(len(coins) + 1):
        dp[i][0] = 0

    # Fill DP table
    for i in range(1, len(coins) + 1):
        for j in range(1, total + 1):
            current_coin = coins[i - 1]

            # If current coin can be used for current amount
            if current_coin <= j:
                # Choose minimum between:
                # 1. Using current coin (1 + solution for remaining amount)
                # 2. Not using current coin (solution without current coin)
                dp[i][j] = min(1 + dp[i][j - current_coin], dp[i - 1][j])
            else:
                # Current coin too large, use solution without it
                dp[i][j] = dp[i - 1][j]

    # Return -1 if no solution found, otherwise return minimum coins needed
    return -1 if dp[-1][-1] == float("inf") else dp[-1][-1]


def main():
    """
    Driver code to test coin change algorithm with various inputs.
    Tests different combinations of coins and target amounts.
    """
    # Test cases: pairs of coin arrays and target amounts
    test_cases = [
        ([1, 2, 5], 11),  # Regular case
        ([2], 3),  # Impossible case
        ([1], 0),  # Zero amount
        ([1, 2, 5], 100),  # Larger amount
        ([2, 5, 10, 1], 27),  # Multiple coins
    ]

    for i, (coins, total) in enumerate(test_cases, 1):
        result = coin_change(coins, total)
        print(f"{i}. Coins available: {coins}")
        print(f"\tTarget amount: {total}")
        if result != -1:
            print(f"\tMinimum coins needed: {result}")
        else:
            print("\tImpossible to make this amount!")
        print("-" * 80)


if __name__ == "__main__":
    main()
