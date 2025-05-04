def word_break(s, word_dict):
    """
    Break a string into a space-separated sequence of dictionary words.

    This function uses dynamic programming to find all possible ways to
    segment the input string using words from the provided dictionary.

    Args:
        s (str): Input string to be segmented
        word_dict (set or list): Dictionary of valid words

    Returns:
        list: All possible space-separated sentences that can be formed

    Time Complexity:
        O(n^3): Where n is the length of the string.
        - We have n positions to check (outer loop)
        - For each position, we check j from 0 to i (inner loop)
        - For each j, we may need to copy and append O(n) strings

    Space Complexity:
        O(n^2) in the average case, but could be O(2^n) in the worst case
        if there are exponentially many valid segmentations.
        - dp array stores all valid segmentations
        - Each segmentation could be O(n) in length
        - In worst case, there could be exponentially many valid segmentations
    """
    dp = [[]] * (len(s) + 1)
    dp[0] = [""]  # Base case: empty string can be segmented in one way

    for i in range(1, len(s) + 1):
        prefix = s[:i]

        temp = []
        for j in range(i):
            # Get suffix from current prefix
            suffix = prefix[j:]
            # Check if the suffix is a valid word in our dictionary
            if suffix in word_dict:
                # For each valid segmentation up to position j
                for substring in dp[j]:
                    # Append the current word with a space
                    temp.append((substring + " " + suffix).strip())
        # Store all valid segmentations ending at position i
        dp[i] = temp

    # Return all valid segmentations for the entire string
    return dp[len(s)]


# Driver code to test the implementation
if __name__ == "__main__":
    # Test case 1
    s1 = "catsanddog"
    word_dict1 = {"cat", "cats", "and", "sand", "dog"}
    print(f"Input: s = '{s1}', wordDict = {word_dict1}")
    result1 = word_break(s1, word_dict1)
    print(f"Output: {result1}")
    # Expected: [" cat sand dog", " cats and dog"]

    # Test case 2
    s2 = "pineapplepenapple"
    word_dict2 = {"apple", "pen", "applepen", "pine", "pineapple"}
    print(f"\nInput: s = '{s2}', wordDict = {word_dict2}")
    result2 = word_break(s2, word_dict2)
    print(f"Output: {result2}")

    # Test case 3
    s3 = "catsandog"
    word_dict3 = {"cats", "dog", "sand", "and", "cat"}
    print(f"\nInput: s = '{s3}', wordDict = {word_dict3}")
    result3 = word_break(s3, word_dict3)
    print(f"Output: {result3}")

    # Note: The implementation has a leading space in all results
    # as it always adds a space before appending the suffix
