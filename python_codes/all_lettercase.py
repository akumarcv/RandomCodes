def helper(s: str, index: int, result: list, slate: str) -> None:
    """
    Helper function to generate all lettercase permutations using backtracking

    Args:
        s: Input string to process
        index: Current character position being processed
        result: List to store all valid permutations
        slate: Current permutation being built

    Time Complexity: O(2^n) where n is length of string
    Space Complexity: O(n) for recursion stack
    """
    # Base case: reached end of string
    if index == len(s):
        result.append(slate)
        return

    # Case 1: Current character is a letter
    if s[index].isalpha():
        # Try lowercase version
        helper(s, index + 1, result, slate + s[index].lower())
        # Try uppercase version
        helper(s, index + 1, result, slate + s[index].upper())
    # Case 2: Current character is not a letter
    else:
        # Keep non-letter character as is
        helper(s, index + 1, result, slate + s[index])


def letter_case_permutation(s: str) -> list:
    """
    Generate all possible lettercase permutations of a string

    Args:
        s: Input string containing letters and digits

    Returns:
        List of all possible lettercase permutations

    Example:
        >>> letter_case_permutation("a1b2")
        ['a1b2', 'a1B2', 'A1b2', 'A1B2']
    """
    result = []
    helper(s, 0, result, "")
    return result


def main():
    """
    Driver code to test lettercase permutation functionality
    with various test cases
    """
    strings = ["a1b2", "3z4", "ABC", "123", "xYz"]

    for i, s in enumerate(strings, 1):
        print(f'{i}.\ts: "{s}"')
        print(f"\tOutput: {letter_case_permutation(s)}")
        print("-" * 100)


if __name__ == "__main__":
    main()
