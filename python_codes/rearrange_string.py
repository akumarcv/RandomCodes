from collections import Counter
import heapq


def rearrange(s):
    if s == "":
        return

    char_counter = Counter(s)

    if len(char_counter) == 1:
        return ""

    max_heap = []
    for k, c in char_counter.items():
        heapq.heappush(max_heap, [-c, k])

    result = []
    i = 0

    even_hold_aside, odd_hold_aside = None, None
    for i in range(len(s)):
        if i % 2 == 0 and max_heap:
            count, c = heapq.heappop(max_heap)
            result.append(c)
            count = -count - 1
            if count > 0:
                even_hold_aside = [-count, c]
            else:
                even_hold_aside = None
            if odd_hold_aside is not None:
                heapq.heappush(max_heap, odd_hold_aside)

        elif i % 2 == 1 and max_heap:
            count, c = heapq.heappop(max_heap)
            result.append(c)
            count = -count - 1
            if count > 0:
                odd_hold_aside = [-count, c]
            else:
                odd_hold_aside = None
            if even_hold_aside is not None:
                heapq.heappush(max_heap, even_hold_aside)

    # Check if the rearranged string is valid
    for i in range(1, len(result)):
        if result[i] == result[i - 1]:
            return ""  # Not possible to rearrange
    if len(result) != len(s):
        return ""
    return "".join(result)


# Driver code to test the rearrange function
if __name__ == "__main__":
    test_cases = ["aabbcc", "aaabc", "aaabb", "aaa", "a", ""]

    for s in test_cases:
        result = rearrange(s)
        print(f"Input: {s}, Rearranged: {result}")
