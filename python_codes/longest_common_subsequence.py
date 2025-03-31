def longest_common_subsequence(str1: str, str2: str) -> int:
    """
    Find length of longest common subsequence between two strings using dynamic programming.
    A subsequence is a sequence that can be derived from another sequence by deleting some 
    or no elements without changing the order of the remaining elements.
    
    Args:
        str1: First input string
        str2: Second input string
        
    Returns:
        int: Length of longest common subsequence
        
    Time Complexity: O(m*n) where m,n are lengths of input strings
    Space Complexity: O(m*n) for DP table
    
    Example:
        >>> longest_common_subsequence("abcde", "ace")
        3  # 'ace' is the longest common subsequence
    """
    # Initialize DP table with dimensions [len(str1)+1][len(str2)+1]
    dp = [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]

    # Fill DP table using bottom-up approach
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match, add 1 to previous result
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                # Characters don't match, take max of excluding either character
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    return dp[-1][-1]  # Return length of LCS


def main():
    """
    Driver code to test longest common subsequence functionality.
    Tests various string pairs including:
    - Different length strings
    - Similar strings
    - No common characters
    - Edge cases
    """
    # Test cases with pairs of strings
    first_strings = [
        "qstw",     # Few common chars
        "setter",   # Some common chars
        "abcde",    # Sequential chars
        "partner",  # Random string
        "freedom"   # Multiple common chars
    ]
    second_strings = [
        "gofvn",    # Few matches
        "bat",      # Short string
        "apple",    # Different sequence
        "park",     # Partial match
        "redeem"    # Similar pattern
    ]

    # Process each test case
    for i in range(len(first_strings)):
        print(f"{i + 1}.\tString 1: {first_strings[i]}")
        print(f"\tString 2: {second_strings[i]}")
        result = longest_common_subsequence(first_strings[i], second_strings[i])
        print(f"\tLongest Common Subsequence Length: {result}")
        print("-" * 100)


if __name__ == "__main__":
    main()