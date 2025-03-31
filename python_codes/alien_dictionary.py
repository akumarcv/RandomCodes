from collections import defaultdict, Counter, deque


def alien_order(words: list[str]) -> str:
    """
    Find the order of characters in an alien dictionary based on sorted words.
    Uses Kahn's algorithm for topological sort to determine character ordering.

    Args:
        words: List of strings sorted according to alien dictionary rules

    Returns:
        str: Characters in sorted order according to alien dictionary rules,
             or empty string if invalid order detected

    Time Complexity: O(C) where C is total length of all words
    Space Complexity: O(1) since there are at most 26 lowercase letters

    Example:
        >>> alien_order(["wrt", "wrf", "er", "ett", "rftt"])
        "wertf"
    """
    # Build character graph and initialize in-degree counter
    graph = defaultdict(set)
    in_degree = Counter({c: 0 for word in words for c in word})
    sorted_order = []

    # Compare adjacent words to build character dependencies
    for word1, word2 in zip(words[:-1], words[1:]):
        # Compare characters at same positions in both words
        for c, d in zip(word1, word2):
            if c != d:  # First different character determines order
                if d not in graph[c]:
                    graph[c].add(d)  # Add edge c -> d
                    in_degree[d] += 1  # Increment in-degree of d
                break
        else:  # No break occurred, check prefix condition
            if len(word2) < len(word1):
                return ""  # Invalid case: prefix word should be shorter

    # Initialize queue with characters having no dependencies
    sources_queue = deque([c for c in in_degree if in_degree[c] == 0])

    # Process queue to build topological sort
    while sources_queue:
        c = sources_queue.popleft()
        sorted_order.append(c)

        # Update in-degrees of neighbors
        for d in graph[c]:
            in_degree[d] -= 1
            if in_degree[d] == 0:  # If all dependencies processed
                sources_queue.append(d)

    # Check if all characters were included
    if len(sorted_order) < len(in_degree):
        return ""  # Cycle detected
    return "".join(sorted_order)


def main():
    """
    Driver code to test alien_order function with various test cases
    """
    words = [
        # Test case 1: Complex dictionary with multiple rules
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
        # Test case 2: Common English-like words
        ["vanilla", "alpine", "algor", "port", "norm", "nylon", "ophellia", "hidden"],
        # Test case 3: Short words with spaces
        ["passengers", "to", "the", "unknown"],
        # Test case 4: NATO phonetic alphabet subset
        ["alpha", "bravo", "charlie", "delta"],
        # Test case 5: Simple two-word case
        ["jupyter", "ascending"],
    ]

    for i, test_case in enumerate(words, 1):
        print(f"{i}.\twords = {test_case}")
        print(f'\tDictionary = "{alien_order(test_case)}"')
        print("-" * 100)


if __name__ == "__main__":
    main()
