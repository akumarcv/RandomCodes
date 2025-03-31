from collections import Counter, defaultdict
from typing import List


def combination_helper(w, pattern, result):
    """
    Recursively generate all 3-sequence patterns from a list of websites.
    Uses backtracking to build all possible combinations.

    Args:
        w: Remaining websites to choose from
        pattern: Current partial pattern being built
        result: List to store generated patterns

    Returns:
        None: Updates result list in-place

    Time Complexity: O(n³) where n is length of website list
    Space Complexity: O(n) for recursion stack and pattern storage
    """
    if len(pattern) == 3:
        result.append(tuple(pattern))  # Found complete 3-sequence pattern
        return
    if len(w) < 3 - len(pattern):  # Not enough elements left
        return

    for i in range(len(w)):
        # Add current website to pattern and continue with remaining websites
        combination_helper(w[i + 1 :], pattern + [w[i]], result)


def return_all_combinations(websites):
    """
    Generate all possible 3-sequence website patterns from a user's history.

    Args:
        websites: List of websites visited by a user in chronological order

    Returns:
        set: Set of tuples, each representing a unique 3-sequence pattern

    Time Complexity: O(n³) for generating all combinations
    Space Complexity: O(n³) for storing all unique combinations

    Example:
        >>> return_all_combinations(['home', 'about', 'career', 'home'])
        {('home', 'about', 'career'), ('home', 'about', 'home'), ('about', 'career', 'home')}
    """
    if len(websites) < 3:
        return []  # Cannot form 3-sequence with fewer than 3 websites
    result = []
    combination_helper(websites, [], result)
    return set(result)  # Convert to set to eliminate duplicates


def mostVisitedPattern(
    username: List[str], timestamp: List[int], website: List[str]
) -> List[str]:
    """
    Find the most common 3-sequence pattern of websites across all users.
    If multiple patterns have same frequency, return lexicographically smallest.

    Args:
        username: List of user identifiers for each visit
        timestamp: List of timestamps for each visit
        website: List of websites visited

    Returns:
        List[str]: Most frequent 3-sequence pattern

    Time Complexity: O(n³) where n is total number of website visits
    Space Complexity: O(n²) for storing user histories and patterns

    Example:
        >>> mostVisitedPattern(
        ...     ["joe", "joe", "joe", "james", "james"],
        ...     [1, 2, 3, 4, 5],
        ...     ["home", "about", "career", "home", "cart"]
        ... )
        ["home", "about", "career"]
    """
    # Build user history map: username -> list of websites in chronological order
    graph = defaultdict(list)
    for u, t, w in sorted(zip(username, timestamp, website)):
        graph[u].append(w)  # Group websites by user, sorted by timestamp

    # Count occurrences of each 3-sequence pattern across all users
    counter = Counter()
    for u, w in graph.items():
        # For each user, count each pattern only once (even if they visit it multiple times)
        for lists_3 in return_all_combinations(w):
            counter[tuple(lists_3)] += 1

    # Find pattern with highest count (or lexicographically smallest if tied)
    pattern, max_count = None, 0

    for pat, count in counter.items():
        if count > max_count:
            max_count = count
            pattern = pat
        elif count == max_count and pat < pattern:
            pattern = pat  # Lexicographically smaller pattern with same count

    return list(pattern)  # Convert tuple back to list for return


def test_most_visited_pattern():
    """
    Test cases for finding most visited 3-sequence pattern.
    Verifies algorithm works correctly for various input scenarios.

    Test cases include:
    - Basic pattern identification
    - Multiple users with same pattern
    - Different timestamp distributions
    - Complex overlapping patterns
    - Edge cases with unique patterns

    Each test displays:
    - Input data (users, timestamps, websites)
    - Expected output pattern
    - Actual output pattern
    - Test result (pass/fail)
    """
    test_cases = [
        # Test Case 1: Basic case with clear pattern
        (
            [
                "joe",
                "joe",
                "joe",
                "james",
                "james",
                "james",
                "james",
                "mary",
                "mary",
                "mary",
            ],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [
                "home",
                "about",
                "career",
                "home",
                "cart",
                "maps",
                "home",
                "home",
                "about",
                "career",
            ],
            ["home", "about", "career"],  # Expected output
        ),
        # Test Case 2: Multiple users with same pattern
        (
            ["ua", "ua", "ua", "ub", "ub", "ub"],
            [1, 2, 3, 4, 5, 6],
            ["a", "b", "c", "a", "b", "c"],
            ["a", "b", "c"],  # Expected output
        ),
        # Test Case 3: Pattern with different timestamps
        (
            ["dowg", "dowg", "dowg"],
            [10, 20, 30],
            ["home", "about", "career"],
            ["home", "about", "career"],  # Expected output
        ),
        # Test Case 4: Complex case with overlapping patterns
        (
            [
                "h",
                "eiy",
                "cq",
                "h",
                "cq",
                "txldsscx",
                "cq",
                "txldsscx",
                "h",
                "cq",
                "cq",
            ],
            [
                527896567,
                334462937,
                517687281,
                134127993,
                859112386,
                159548699,
                51100299,
                444082139,
                926837079,
                317455832,
                411747930,
            ],
            [
                "hibympufi",
                "hibympufi",
                "hibympufi",
                "hibympufi",
                "hibympufi",
                "hibympufi",
                "hibympufi",
                "hibympufi",
                "yljmntrclw",
                "hibympufi",
                "yljmntrclw",
            ],
            ["hibympufi", "hibympufi", "yljmntrclw"],  # Expected output
        ),
    ]

    for i, (users, times, sites, expected) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print("Input:")
        print(f"Users: {users}")
        print(f"Timestamps: {times}")
        print(f"Websites: {sites}")
        print(f"Expected Pattern: {expected}")

        result = mostVisitedPattern(users, times, sites)
        print(f"Got Pattern: {result}")

        assert (
            result == expected
        ), f"Test case {i} failed! Expected {expected}, got {result}"
        print("✓ Passed")


if __name__ == "__main__":
    test_most_visited_pattern()
    print("\nAll test cases passed!")
