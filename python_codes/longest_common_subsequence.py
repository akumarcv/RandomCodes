def longest_common_subsequence(str1, str2):
    dp = [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def main():
    first_strings = ["qstw", "setter", "abcde", "partner", "freedom"]
    second_strings = ["gofvn", "bat", "apple", "park", "redeem"]

    for i in range(len(first_strings)):
        print(i + 1, ".\t str1: ", first_strings[i], sep="")
        print("\t str2: ", second_strings[i], sep="")
        print(
            "\n\t The length of the longest common subsequence is: ",
            longest_common_subsequence(first_strings[i], second_strings[i]),
            sep="",
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
