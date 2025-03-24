def print_possible_combinations(s, word_dict):
    """
    Prints all possible combinations of breaking the string using words from 
    the dictionary
    Args:
        s: Input string
        word_dict: List of dictionary words
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
    dp = [False] * (len(s) + 1)
    dp[0] = True

    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_dict:
                dp[i] = True
                break
    return dp[-1]


def main():

    s = [
        "vegancookbook",
        "catsanddog",
        "highwaycrash",
        "pineapplepenapple",
        "screamicecream",
        "educativecourse",
    ]
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
    for i in range(len(s)):
        print("Test Case #", i + 1, "\n\nInput: '" + str(s[i]) + "'\n")
        print_possible_combinations(s[i], word_dict)
        print("\nOutput: " + str(word_break(str(s[i]), word_dict)))
        print("-" * 100)


if __name__ == "__main__":
    main()
