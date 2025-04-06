def longest_palindromic_substring(s: str) -> int:
    """
    Find length of longest palindromic substring using two-pointer technique.
    This approach expands around potential centers of palindromes.

    Args:
        s: Input string to analyze

    Returns:
        int: Length of longest palindromic substring found

    Time Complexity: O(n²) where n is length of string
    Space Complexity: O(1) as we only use constant extra space

    Example:
        >>> longest_palindromic_substring("babad")
        3  # Longest palindrome is "bab" or "aba"
    """
    # Handle empty string case
    if not s:
        return 0

    max_len = 1  # Every string has at least 1 character palindrome

    def expand_around_center(left: int, right: int) -> int:
        """
        Helper function to expand around a potential palindrome center.

        Args:
            left: Left pointer (starting position)
            right: Right pointer (starting position)

        Returns:
            int: Length of palindrome expanded from this center
        """
        # Continue expanding while within bounds and characters match
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1  # Move left pointer leftwards
            right += 1  # Move right pointer rightwards

        # Return length of palindrome (right-left-1 gives correct length)
        return right - left - 1

    # Iterate through each possible center position
    for i in range(len(s)):
        # Case 1: Palindrome with odd length (single character center)
        # Example: "racecar" has center 'e'
        odd_len = expand_around_center(i, i)

        # Case 2: Palindrome with even length (between two characters)
        # Example: "abba" has center between the two 'b's
        even_len = expand_around_center(i, i + 1)

        # Update max_len if we found a longer palindrome
        max_len = max(max_len, odd_len, even_len)

    return max_len


# Driver code
def main():
    """
    Test function for longest palindromic substring.
    Tests various cases including:
    - Short strings with no palindromes
    - Strings with palindromes in middle
    - Strings of repeated characters
    - Long palindromes
    """
    test_cases = [
        ("cat", 1),  # Single character ('c', 'a', or 't')
        ("lever", 4),  # "leve" or "evel"
        ("xyxxyz", 3),  # "xyx"
        ("wwwwwwwwww", 10),  # Whole string is palindrome
        ("tattarrattat", 12),  # Whole string is palindrome
        ("babad", 3),  # "bab" or "aba"
        ("cbbd", 2),  # "bb"
        ("", 0),  # Empty string
        ("a", 1),  # Single character
        ("aacabdkacaa", 3),  # "aca"
    ]

    for i, (input_str, expected) in enumerate(test_cases):
        result = longest_palindromic_substring(input_str)
        status = "✓" if result == expected else "✗"

        print(f"{i + 1}.\tInput: '{input_str}'")
        print(f"\tExpected: {expected}, Result: {result} {status}")
        print("-" * 60)


if __name__ == "__main__":
    main()
