def print_possible_combinations(s, word_dict):
    """
    Prints all possible combinations of breaking the string using words from
    the dictionary.
    Uses dynamic programming to build up combinations incrementally.
    
    Args:
        s: Input string to break into dictionary words
        word_dict: List of valid dictionary words
        
    Returns:
        None: Prints results to console
        
    Time Complexity: O(n³) where n is the length of string s
    Space Complexity: O(n*m) where m is number of possible combinations
    
    Example:
        >>> print_possible_combinations("catdog", ["cat", "dog"])
        All possible combinations:
            cat dog
    """
    n = len(s)
    # dp[i] stores all possible combinations of breaking s[0:i]
    dp = [[] for _ in range(n + 1)]
    dp[0] = [""]  # Empty string has one combination (empty)

    # For each position in string
    for i in range(1, n + 1):
        # Try all possible substrings ending at i
        for j in range(i):
            # Get substring from j to i
            word = s[j:i]
            # If word exists in dictionary
            if word in word_dict:
                # Add new combinations by appending current word
                # to all combinations at j
                for prev in dp[j]:
                    new_comb = prev + " " + word if prev else word
                    dp[i].append(new_comb)

    # Print all combinations
    if dp[n]:
        print("All possible combinations:")
        for combination in dp[n]:
            print(f"\t{combination}")
    else:
        print("\tNo possible combinations")


def word_break(s, word_dict):
    """
    Determine if string s can be segmented into words from the dictionary.
    Uses bottom-up dynamic programming approach to check if valid segmentation exists.
    
    Args:
        s: Input string to check for word break possibility
        word_dict: List of valid dictionary words
        
    Returns:
        bool: True if string can be segmented using dictionary words, False otherwise
        
    Time Complexity: O(n²) where n is the length of string s
    Space Complexity: O(n) for the dp array
    
    Example:
        >>> word_break("catsanddog", ["cats", "and", "dog"])
        True  # Can be broken as "cats and dog"
    """
    dp = [False] * (len(s) + 1)  # dp[i] = can s[0:i] be segmented?
    dp[0] = True  # Empty string can always be segmented

    # For each position in the string
    for i in range(1, len(s) + 1):
        # Try all possible substrings ending at position i
        for j in range(i):
            # If s[0:j] can be segmented and s[j:i] is in dictionary
            if dp[j] and s[j:i] in word_dict:
                dp[i] = True  # s[0:i] can be segmented
                break  # No need to check further substrings
    
    return dp[-1]  # Return result for whole string


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
    # Test strings to check for word break possibility
    s = [
        "vegancookbook",
        "catsanddog",
        "highwaycrash",
        "pineapplepenapple",
        "screamicecream",
        "educativecourse",
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