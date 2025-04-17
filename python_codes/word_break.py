def print_possible_combinations(s, word_dict):
    """
    Print all possible ways to segment the string using words from dictionary.
    Uses backtracking to find all valid segmentations.

    Args:
        s: String to segment
        word_dict: List of valid words to use for segmentation
    """
    # Convert list to set for O(1) lookups
    word_set = set(word_dict)

    def backtrack(start, path):
        """
        Recursive backtracking function to find all valid segmentations.

        Args:
            start: Starting index in string to segment
            path: Current list of words in segmentation
        """
        # Base case: reached end of string - valid segmentation found
        if start == len(s):
            print('"' + " ".join(path) + '"')
            return

        # Try all possible word lengths starting at current position
        for end in range(start + 1, len(s) + 1):
            # If substring is a valid word, include it and continue
            if s[start:end] in word_set:
                path.append(s[start:end])
                backtrack(end, path)
                path.pop()  # Backtrack

    backtrack(0, [])


def word_break(s, word_dict):
    """
    Determine if a string can be segmented into words from the dictionary.

    Uses dynamic programming approach where dp[i] represents whether
    the substring s[0:i] can be segmented into dictionary words.

    Args:
        s: String to segment
        word_dict: List of valid words to use for segmentation

    Returns:
        bool: True if string can be segmented, False otherwise

    Time Complexity: O(n²*m) where n is length of string and m is average word length
    Space Complexity: O(n) for the dp array
    """
    # Convert word_dict to set for O(1) lookups
    word_set = set(word_dict)

    # dp[i] indicates if substring s[0:i] can be segmented
    dp = [False] * (len(s) + 1)

    # Empty string can always be segmented
    dp[0] = True

    # Build solution bottom-up
    for i in range(1, len(s) + 1):
        # Try all possible word endings at current position
        for j in range(i):
            # If s[0:j] can be segmented and s[j:i] is a valid word
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[len(s)]


def main():
    """
    Driver function to test word break algorithms with various test cases.

    Tests include:
    - Strings that can be broken in multiple ways
    - Strings that cannot be broken
    - Strings with overlapping word patterns
    - Edge cases with repeated words

    For each test case:
    1. Prints the input string
    2. Shows all possible word break combinations
    3. Indicates whether any valid segmentation exists
    """
    # Test strings to check for word breaking
    s = [
        "educativeio",
        "applepenapple",
        "catsandog",
        "codecodingcodecoding",
        "vegancookbookcook",
    ]

    # Dictionary of valid words for breaking strings
    word_dict = [
        "ncoo",
        "kboo",
        "inea",
        "icec",
        "ghway",
        "and",
        "anco",
        "hi",
        "way",
        "wa",
        "amic",
        "ed",
        "cecre",
        "ena",
        "tsa",
        "ami",
        "lepen",
        "highway",
        "ples",
        "ookb",
        "epe",
        "nea",
        "cra",
        "lepe",
        "ycras",
        "dog",
        "nddo",
        "hway",
        "ecrea",
        "apple",
        "shp",
        "kbo",
        "yc",
        "cat",
        "tsan",
        "ganco",
        "lescr",
        "ep",
        "penapple",
        "pine",
        "book",
        "cats",
        "andd",
        "vegan",
        "cookbook",
    ]

    print(
        "The list of words we can use to break down the strings are:\n\n",
        word_dict,
        "\n",
    )
    print("-" * 100)

    # Process each test case
    for i in range(len(s)):
        print("Test Case #", i + 1, "\n\nInput: '" + str(s[i]) + "'\n")
        # Show all possible segmentations
        print_possible_combinations(s[i], word_dict)
        # Show whether any valid segmentation exists
        print("\nOutput: " + str(word_break(str(s[i]), word_dict)))
        print("-" * 100)


if __name__ == "__main__":
    main()
