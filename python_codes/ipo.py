from heapq import *


def maximum_capital(c, k, capitals, profits):
    current_capital = c
    capital_min_heap = []
    profit_max_heap = []

    for i in range(len(capitals)):
        heappush(capital_min_heap, (capitals[i], i))

    for j in range(k):
        while capital_min_heap and capital_min_heap[0][0] <= current_capital:

            capital, i = heappop(capital_min_heap)
            heappush(profit_max_heap, -profits[i])

        if not profit_max_heap:
            break
        current_capital += -heappop(profit_max_heap)

    return current_capital


def main():
    input = (
        (0, 1, [1, 1, 2], [1, 2, 3]),
        (1, 2, [1, 2, 2, 3], [2, 4, 6, 8]),
        (2, 3, [1, 3, 4, 5, 6], [1, 2, 3, 4, 5]),
        (1, 3, [1, 2, 3, 4], [1, 3, 5, 7]),
        (7, 2, [6, 7, 8, 10], [4, 8, 12, 14]),
        (2, 4, [2, 3, 5, 6, 8, 12], [1, 2, 5, 6, 8, 9]),
    )
    num = 1
    for i in input:
        print(f"{num}.\tProject capital requirements:  {i[2]}")
        print(f"\tProject expected profits:      {i[3]}")
        print(f"\tNumber of projects:            {i[1]}")
        print(f"\tStart-up capital:              {i[0]}")
        print("\n\tMaximum capital earned: ", maximum_capital(i[0], i[1], i[2], i[3]))
        print("-" * 100, "\n")
        num += 1


if __name__ == "__main__":
    main()
