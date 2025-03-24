def num_of_decodings(decode_str):
    dp = [0] * (len(decode_str) + 1)
    dp[0] = 1
    dp[1] = 1 if decode_str[0] != "0" else 0

    for i in range(2, len(decode_str) + 1):
        if decode_str[i - 1] != '0':
            dp[i] += dp[i - 1]
        if decode_str[i - 2] == '1' or (decode_str[i - 2]
                                        == '2' and decode_str[i - 1] < '7'):
            dp[i] += dp[i - 2]
    return dp[-1]


def main():
    decode_str = [
        "124",
        "123456",
        "11223344",
        "0",
        "0911241",
        "10203",
        "999901"]

    for i in range(len(decode_str)):
        print(
            i + 1,
            f".\t There are {num_of_decodings(decode_str[i])} \
                ways in which we \
                can decode the string: '",
            decode_str[i],
            "'",
            sep="",
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
