def trib_helper(n, table):
    if table[n] != -1:
        return table[n]

    new_val = (
        trib_helper(n - 1, table)
        + trib_helper(n - 2, table)
        + trib_helper(n - 3, table)
    )

    table[n] = new_val

    return table[n]


def find_tribonacci(n):

    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 1

    table = [-1] * (n + 1)
    table[0] = 0
    table[1] = 1
    table[2] = 1

    val = trib_helper(n, table)
    # Replace this placeholder return statement with your code

    return val


# Driver code
def main():
    n_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for i in n_values:
        print(i, ".\t Tribonacci value: ", find_tribonacci(i), sep="")
        print("-" * 100)


if __name__ == "__main__":
    main()