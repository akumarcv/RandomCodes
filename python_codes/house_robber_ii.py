def house_robber(money):
    if len(money) == 0:
        return 0
    if len(money) == 1:
        return money[0]

    return max(rob(money[:-1]), rob(money[1:]))


def rob(money):

    dp = [0] * (len(money) + 1)
    dp[0] = 0
    dp[1] = money[0]
    for i in range(2, len(money) + 1):
        dp[i] = max(money[i - 1] + dp[i - 2], dp[i - 1])
    return dp[-1]


# Driver code


def main():

    inputs = [
        [2, 3, 2],
        [1, 2, 3, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [7, 4, 1, 9, 3],
        [],
    ]

    for i in range(len(inputs)):
        print(i + 1, ".\tHouses: ", inputs[i], sep="")
        print("\n\tMaximum loot:", house_robber(inputs[i]))
        print("-" * 100)


if __name__ == "__main__":
    main()
