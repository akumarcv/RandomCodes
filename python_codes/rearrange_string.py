from collections import Counter
import heapq


def rearrange(s):
    """
    Rearrange string so that no adjacent characters are the same.
    Uses max heap to prioritize characters with highest frequency.

    Args:
        s: Input string to rearrange

    Returns:
        str: Rearranged string with no adjacent duplicates
        empty string: If rearrangement is not possible
        None: If input is empty

    Time Complexity: O(n log k) where n is string length and k is unique characters
    Space Complexity: O(k) for heap and counter storage

    Example:
        >>> rearrange("aabbcc")
        "abcabc"  # Characters distributed evenly with no adjacents
    """
    # Handle empty string case
    if s == "":
        return

    # Count frequency of each character
    char_counter = Counter(s)

    # If only one character type exists, can't rearrange to avoid adjacency
    if len(char_counter) == 1:
        return ""

    # Create max heap (using negative counts for max behavior in min heap)
    max_heap = []
    for k, c in char_counter.items():
        heapq.heappush(max_heap, [-c, k])  # Store as [negative count, character]

    result = []  # Store rearranged characters
    i = 0  # Position tracker

    # Variables to hold characters that need to wait a turn before being reused
    even_hold_aside, odd_hold_aside = None, None

    for i in range(len(s)):
        if i % 2 == 0 and max_heap:
            # Handle even positions with most frequent character
            count, c = heapq.heappop(max_heap)  # Get most frequent char
            result.append(c)  # Add to result
            count = -count - 1  # Decrease frequency (working with negatives)

            if count > 0:
                # If character still has occurrences, hold for later
                even_hold_aside = [-count, c]
            else:
                even_hold_aside = None

            if odd_hold_aside is not None:
                # Add back previous odd position's held character
                heapq.heappush(max_heap, odd_hold_aside)

        elif i % 2 == 1 and max_heap:
            # Handle odd positions with most frequent character
            count, c = heapq.heappop(max_heap)  # Get most frequent char
            result.append(c)  # Add to result
            count = -count - 1  # Decrease frequency

            if count > 0:
                # If character still has occurrences, hold for later
                odd_hold_aside = [-count, c]
            else:
                odd_hold_aside = None

            if even_hold_aside is not None:
                # Add back previous even position's held character
                heapq.heappush(max_heap, even_hold_aside)

    # Check if the rearranged string is valid
    for i in range(1, len(result)):
        if result[i] == result[i - 1]:
            return ""  # Not possible to rearrange (adjacent duplicates exist)

    if len(result) != len(s):
        return ""  # Not all characters were used

    return "".join(result)  # Return rearranged string


# Driver code to test the rearrange function
if __name__ == "__main__":
    """
    Test the rearrange function with various string patterns.
    Tests include:
    - Evenly distributed characters
    - Unevenly distributed characters
    - Impossible rearrangements
    - Edge cases (empty string, single character)
    """
    test_cases = ["aabbcc", "aaabc", "aaabb", "aaa", "a", ""]

    for s in test_cases:
        result = rearrange(s)
        print(f"Input: {s}, Rearranged: {result}")
