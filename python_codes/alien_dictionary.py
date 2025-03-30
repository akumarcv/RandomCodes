from collections import defaultdict, Counter, deque


def alien_order(words):

    graph = defaultdict(set)
    in_degree = Counter({c: 0 for word in words for c in word})
    sorted_order = []

    for word1, word2 in zip(words[:-1], words[1:]):
        for c, d in zip(word1, word2):
            if c != d:
                if d not in graph[c]:
                    graph[c].add(d)
                    in_degree[d] += 1
                break
        else:
            if len(word2) < len(word1):
                return ""

    sources_queue = deque([c for c in in_degree if in_degree[c] == 0])

    while sources_queue:
        c = sources_queue.popleft()
        sorted_order.append(c)

        for d in graph[c]:
            in_degree[d] -= 1
            if in_degree[d] == 0:
                sources_queue.append(d)

    if len(sorted_order) < len(in_degree):
        return ""
    return "".join(sorted_order)


# Driver code
def main():
    words = [
        [
            "mzosr",
            "mqov",
            "xxsvq",
            "xazv",
            "xazau",
            "xaqu",
            "suvzu",
            "suvxq",
            "suam",
            "suax",
            "rom",
            "rwx",
            "rwv",
        ],
        ["vanilla", "alpine", "algor", "port", "norm", "nylon", "ophellia", "hidden"],
        ["passengers", "to", "the", "unknown"],
        ["alpha", "bravo", "charlie", "delta"],
        ["jupyter", "ascending"],
    ]

    for i in range(len(words)):
        print(i + 1, ".\twords = ", words[i], sep="")
        print('\tDictionary = "', alien_order(words[i]), '"', sep="")
        print("-" * 100)


if __name__ == "__main__":
    main()
