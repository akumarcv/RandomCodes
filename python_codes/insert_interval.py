def insert_interval(existing_intervals, new_interval):

    new_result = []
    i = 0
    while i < len(existing_intervals) and existing_intervals[i][0] < new_interval[0]:
        new_result.append(existing_intervals[i])
        i = i + 1

    print(new_result)
    if not new_result or new_result[-1][1] < new_interval[0]:
        new_result.append(new_interval)
    else:
        new_result[-1][1] = max(new_result[-1][1], new_interval[1])

    while i < len(existing_intervals):
        if new_result[-1][1] < existing_intervals[i][0]:
            new_result.append(existing_intervals[i])
        else:
            new_result[-1][1] = max(new_result[-1][1], existing_intervals[i][1])
        i = i + 1
    return new_result


def main():
    new_interval = [[2, 5], [16, 18], [10, 12], [1, 3], [1, 10]]
    existing_intervals = [
        [[1, 2], [3, 4], [5, 8], [9, 15]],
        [[1, 3], [5, 7], [10, 12], [13, 15], [19, 21], [21, 25], [26, 27]],
        [[8, 10], [12, 15]],
        [[5, 7], [8, 9]],
        [[3, 5]],
    ]

    for i in range(len(new_interval)):
        print(i + 1, ".\tExiting intervals: ", existing_intervals[i], sep="")
        print("\tNew interval: ", new_interval[i], sep="")
        output = insert_interval(existing_intervals[i], new_interval[i])
        print("\tUpdated intervals: ", output, sep="")
        print("-" * 100)


if __name__ == "__main__":
    main()
