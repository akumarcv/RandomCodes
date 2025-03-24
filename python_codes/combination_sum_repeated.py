def combination_sum(nums, target):

    dp = [[] for _ in range(target + 1)]
    dp[0].append([])

    for i in range(1, target + 1):
        for j in range(len(nums)):
            if nums[j] <= i:
                for p in dp[i - nums[j]]:
                    temp = p + [nums[j]]
                    temp.sort()
                    if temp not in dp[i]:
                        dp[i].append(temp)
    if dp[target]:
        return dp[target]
    else:
        return []


# Driver code
def main():
    nums = [
        [2, 3, 5],
        [3, 6, 7, 8],
        [4, 5, 6, 9],
        [20, 25, 30, 35, 40],
        [3, 5, 7]
    ]

    targets = [5, 15, 11, 40, 15]

    for i in range(len(nums)):
        print(i + 1, ".", "\tnums: ", nums[i], sep="")
        print("\tTarget: ", targets[i], sep="")
        combinations = combination_sum(nums[i], targets[i])
        print("\tNumber of Combinations: ", combinations, sep="")
        print("-" * 100)
        print()


if __name__ == "__main__":
    main()
