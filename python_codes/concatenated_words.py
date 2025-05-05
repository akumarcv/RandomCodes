"""
Given an array of strings words (without duplicates), return all the
concatenated words in the given list of words.

A concatenated word is defined as a string that is
comprised entirely of at least two shorter words (not necessarily distinct)
in the given array.
"""

from typing import List, Set


def dfs(word: str, word_set: Set[str], memo: dict = None) -> bool:
    """
    Determine if a word can be formed by concatenating other words in the word_set.

    Args:
        word: String to check if it can be formed by concatenating other words
        word_set: Set of words to use for concatenation
        memo: Dictionary for memoization to avoid redundant calculations

    Returns:
        True if word can be formed by concatenating other words, False otherwise

    Time Complexity: O(n*m²) where n is number of words and m is the max word length
    Space Complexity: O(m) for recursion stack and memoization
    """
    if memo is None:
        memo = {}

    # Return memoized result if available
    if word in memo:
        return memo[word]

    for i in range(1, len(word)):
        prefix = word[:i]
        suffix = word[i:]

        # Case 1: Both prefix and suffix are in word_set
        if prefix in word_set and suffix in word_set:
            memo[word] = True
            return True

        # Case 2: Prefix is in word_set and suffix can be further split
        if prefix in word_set and dfs(suffix, word_set, memo):
            memo[word] = True
            return True

        # Case 3: Suffix is in word_set and prefix can be further split
        if suffix in word_set and dfs(prefix, word_set, memo):
            memo[word] = True
            return True

    memo[word] = False
    return False


def findAllConcatenatedWordsInADict(words: List[str]) -> List[str]:
    """
    Find all concatenated words in the given list.

    Args:
        words: List of words to search for concatenated words

    Returns:
        List of all concatenated words found

    Time Complexity: O(n*m²) where n is number of words and m is the max word length
    Space Complexity: O(n) for storing the word set and result
    """
    # Create a set for O(1) lookups
    word_set = set(words)
    concatenated_words = []

    # Check each word if it can be formed by concatenating other words
    for word in words:
        # Skip empty strings or single-character words as they can't be concatenated
        if len(word) <= 1:
            continue

        # Remove the current word from set to avoid using itself as a whole
        word_set.remove(word)

        if dfs(word, word_set):
            concatenated_words.append(word)

        # Add the word back to the set
        word_set.add(word)

    return concatenated_words


# Driver code
if __name__ == "__main__":
    # Test case 1: Example from the problem statement
    words1 = [
        "cat",
        "cats",
        "catsdogcats",
        "dog",
        "dogcatsdog",
        "hippopotamuses",
        "rat",
        "ratcatdogcat",
    ]
    result1 = findAllConcatenatedWordsInADict(words1)
    print("Test Case 1:")
    print(f"Input: {words1}")
    print(
        f"Output: {result1}"
    )  # Expected: ['catsdogcats', 'dogcatsdog', 'ratcatdogcat']
    print()

    # Test case 2: Empty list
    words2 = []
    result2 = findAllConcatenatedWordsInADict(words2)
    print("Test Case 2:")
    print(f"Input: {words2}")
    print(f"Output: {result2}")  # Expected: []
    print()

    # Test case 3: List with no concatenated words
    words3 = ["apple", "banana", "orange", "kiwi"]
    result3 = findAllConcatenatedWordsInADict(words3)
    print("Test Case 3:")
    print(f"Input: {words3}")
    print(f"Output: {result3}")  # Expected: []
    print()

    # Test case 4: List with repeated words that form concatenations
    words4 = ["a", "b", "ab", "abc", "bc"]
    result4 = findAllConcatenatedWordsInADict(words4)
    print("Test Case 4:")
    print(f"Input: {words4}")
    print(f"Output: {result4}")  # Expected: ['ab', 'abc', 'bc']
