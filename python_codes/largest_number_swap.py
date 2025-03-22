import heapq


def largest_integer(num):
    digit_list = [int(d) for d in str(num)]
    odd_max_heap = []
    even_max_heap = []
    result = []

    for i in digit_list:
        if i % 2 == 0:
            heapq.heappush(even_max_heap, -i)
        else:
            heapq.heappush(odd_max_heap, -i)

    for i in digit_list:
        if i % 2 == 0:
            result.append(-heapq.heappop(even_max_heap))
        else:
            result.append(-heapq.heappop(odd_max_heap))
    return int("".join(map(str, result)))


def main():
    test_cases = [1234, 65875, 4321, 2468, 98123]
    for num in test_cases:
        print("\tInput number:", num)
        print("\n\tOutput number:", largest_integer(num))
        print("-" * 100)


if __name__ == "__main__":
    main()
