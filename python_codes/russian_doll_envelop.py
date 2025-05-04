def find_position(lis, val):
    low = 0
    high = len(lis) - 1

    while low <= high:
        mid = (low + high) // 2
        if lis[mid] == val:
            return mid
        elif lis[mid] < val:
            low = mid + 1
        else:
            high = mid - 1
    return low


def max_envelopes(envelopes):
    envelopes.sort(key=lambda x: (x[0], -x[1]))

    lis = []
    for i in range(len(envelopes)):
        if not lis or envelopes[i][1] > lis[-1]:
            lis.append(envelopes[i][1])
        else:
            pos = find_position(lis, envelopes[i][1])
            lis[pos] = envelopes[i][1]
    return len(lis)


# Driver code
def main():
    envelopes = [
        [[1, 4], [6, 4], [9, 5], [3, 3]],
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
        [[4, 4], [4, 4], [4, 4]],
        [[3, 1], [5, 8], [5, 9], [3, 1], [9, 1]],
        [[9, 8], [3, 1], [4, 5], [2, 1], [5, 7]],
    ]

    for i in range(len(envelopes)):
        print(i + 1, ".\tEnvelopes:", envelopes[i])
        print(
            "\n\tMaximum number of envelopes which can be Russian-dolled:",
            max_envelopes(envelopes[i]),
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
