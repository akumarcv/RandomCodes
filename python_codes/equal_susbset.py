def can_partition_array(nums):

    sum_nums = sum(nums)
    if sum_nums % 2 != 0:
        return False

    target = sum_nums // 2

    dp = [[False for _ in range(target + 1)] for _ in range(len(nums) + 1)]

    for i in range(len(nums) + 1):
        dp[i][0] = True

    for i in range(1, len(nums) + 1):
        for j in range(1, target + 1):
            if j >= nums[i - 1]:
                dp[i][j] = dp[i - 1][j] or dp[i-1][j - nums[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]
    return dp[-1][-1]


def main():

    nums = [
        [1, 5, 11, 5],
        [1, 2, 3, 5],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]

    for i in range(len(nums)):
        print(i + 1, ".\tArray:", nums[i])
        print("\tResult:", can_partition_array(nums[i]))
        print("-" * 100)


if __name__ == "__main__":
    main()
