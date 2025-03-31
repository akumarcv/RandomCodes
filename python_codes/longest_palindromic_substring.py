def longest_palindromic_substrings(s: str) -> int:
    """
    Find length of longest palindromic substring using dynamic programming.
    Uses a bottom-up DP approach to build palindromes from smaller substrings.
    
    Args:
        s: Input string to analyze
        
    Returns:
        int: Length of longest palindromic substring found
        
    Time Complexity: O(n²) where n is length of string
    Space Complexity: O(n²) for DP table
    
    Example:
        >>> longest_palindromic_substrings("babad")
        3  # Longest palindrome is "bab" or "aba"
    """
    # Handle empty string case
    if not s:
        return 0
        
    # Create DP table where dp[i][j] represents if s[i:j+1] is palindrome
    dp = [[False for _ in range(len(s))] for _ in range(len(s))]

    # All single characters are palindromes by definition
    for i in range(len(s)):
        dp[i][i] = True

    # Track length of longest palindrome found
    max_len = 1

    # Check for palindromes of length 2
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            max_len = max(max_len, 2)

    # Check for palindromes of length 3 or more
    for i in range(3, len(s) + 1):
        k = 0  # Start index of current window
        for j in range(i - 1, len(s)):
            # Check if substring is palindrome:
            # 1. Inner substring is palindrome (dp[k+1][j-1])
            # 2. Outer characters match (s[k] == s[j])
            dp[k][j] = dp[k + 1][j - 1] and (s[k] == s[j])
            if dp[k][j]:
                max_len = max(max_len, j - k + 1)
            k += 1

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
    strings = ["cat", "lever", "xyxxyz", "wwwwwwwwww", "tattarrattat"]

    for i in range(len(strings)):
        print(i + 1, ".\t Input string: '", strings[i], "'", sep="")
        result = longest_palindromic_substrings(strings[i])
        print("\t Number of palindromic substrings: ", result, sep="")
        print("-" * 100)


if __name__ == "__main__":
    main()