def coin_change(coins, total):
    if total == 0:
        return 0

    dp = [
        [float("inf") for _ in range(total + 1)]
        for _ in range(len(coins) + 1)
    ]

    for i in range(len(coins) + 1):
        dp[i][0] = 0

    for i in range(1, len(coins) + 1):
        for j in range(1, total + 1):

            if coins[i - 1] <= j:
                dp[i][j] = min(1 + dp[i][j - coins[i - 1]], dp[i - 1][j])
            else:
                dp[i][j] = dp[i - 1][j]

    if dp[-1][-1] == float("inf"):
        return -1
    return dp[-1][-1]


def main():
    coins = [
        [1, 2, 5],
        [2],
        [1],
        [1, 2, 5],
        [1, 2, 5],
    ]
    totals = [11, 3, 0, 100, 1000]

    for i in range(len(coins)):
        print(i + 1, ". Coins: ", coins[i])
        print("\tTotal: ", totals[i])
        result = coin_change(coins[i], totals[i])
        print("\tNumber of ways to make the total: ", result)
        print("-" * 100)


if __name__ == "__main__":
    main()
