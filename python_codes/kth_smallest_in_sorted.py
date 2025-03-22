from heapq import *


def k_smallest_number(lists, k):
    min_heap = []
    for i in range(len(lists)):
        if len(lists[i]) > 0:
            heappush(min_heap, (lists[i][0], i, 0))
        else:
            continue

    pop_count = 0
    k_smallest_number = 0
    while min_heap:
        k_smallest_number, list_index, element_index = heappop(min_heap)
        pop_count += 1
        if pop_count == k:
            break

        if element_index < len(lists[list_index]) - 1:
            heappush(
                min_heap,
                (lists[list_index][element_index + 1], list_index, element_index + 1),
            )
    # Replace this placeholder return statement with your code

    return k_smallest_number


def main():
    lists = [
        [[2, 6, 8], [3, 6, 10], [5, 8, 11]],
        [[1, 2, 3], [4, 5], [6, 7, 8, 15], [10, 11, 12, 13], [5, 10]],
        [[], [], []],
        [[1, 1, 3, 8], [5, 5, 7, 9], [3, 5, 8, 12]],
        [[5, 8, 9, 17], [], [8, 17, 23, 24]],
    ]

    k = [5, 50, 7, 4, 8]

    for i in range(len(k)):
        print(
            i + 1,
            ".\t Input lists: ",
            lists[i],
            f"\n\t K = {k[i]}",
            f"\n\t {k[i]}th smallest number from the given lists is: ",
            k_smallest_number(lists[i], k[i]),
            sep="",
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
