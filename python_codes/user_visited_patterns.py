from collections import Counter, defaultdict
from typing import List

def combination_helper(w, pattern, result):
    if len(pattern)==3:
        result.append(tuple(pattern))
        return 
    if len(w) < 3 - len(pattern):  # Not enough elements left
        return
    
    for i in range(len(w)):
        combination_helper(w[i+1:], pattern+[w[i]], result)

def return_all_combinations(websites):
    if len(websites) < 3:
        return []
    result = []
    combination_helper(websites, [], result)
    return set(result)

def mostVisitedPattern(
    username: List[str], timestamp: List[int], website: List[str]
) -> List[str]:
    
    graph = defaultdict(list)
    for u, t, w in sorted(zip(username, timestamp, website)):
        graph[u].append(w)

    counter = Counter()
    for u, w in graph.items():
        for lists_3 in return_all_combinations(w):
            counter[tuple(lists_3)]+=1
    
    pattern, max_count = None, 0

    for pat, count in counter.items():
        if count>max_count:
            max_count = count
            pattern = pat
        elif count==max_count and pat<pattern:
            pattern = pat

    return list(pattern)


def test_most_visited_pattern():
    """
    Test cases for finding most visited 3-sequence pattern
    Each test case contains usernames, timestamps, and websites visited
    """
    test_cases = [
        # Test Case 1: Basic case with clear pattern
        (
            ["joe","joe","joe","james","james","james","james","mary","mary","mary"],
            [1,2,3,4,5,6,7,8,9,10],
            ["home","about","career","home","cart","maps","home","home","about","career"],
            ["home","about","career"]  # Expected output
        ),
        
        # Test Case 2: Multiple users with same pattern
        (
            ["ua","ua","ua","ub","ub","ub"],
            [1,2,3,4,5,6],
            ["a","b","c","a","b","c"],
            ["a","b","c"]  # Expected output
        ),
        
        # Test Case 3: Pattern with different timestamps
        (
            ["dowg","dowg","dowg"],
            [10,20,30],
            ["home","about","career"],
            ["home","about","career"]  # Expected output
        ),
        
        # Test Case 4: Complex case with overlapping patterns
        (
            ["h","eiy","cq","h","cq","txldsscx","cq","txldsscx","h","cq","cq"],
            [527896567,334462937,517687281,134127993,859112386,159548699,51100299,444082139,926837079,317455832,411747930],
            ["hibympufi","hibympufi","hibympufi","hibympufi","hibympufi","hibympufi","hibympufi","hibympufi","yljmntrclw","hibympufi","yljmntrclw"],
            ["hibympufi","hibympufi","yljmntrclw"]  # Expected output
        )
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
        
        assert result == expected, f"Test case {i} failed! Expected {expected}, got {result}"
        print("✓ Passed")

if __name__ == "__main__":
    test_most_visited_pattern()
    print("\nAll test cases passed!")