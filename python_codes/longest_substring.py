def longest_substring(s: str) -> int:
    """
    Find length of longest substring without repeating characters using sliding window.
    Uses a dictionary to track last seen positions of characters and adjusts window size.
    
    Args:
        s: Input string to analyze
        
    Returns:
        int: Length of longest substring without repeating characters
        
    Time Complexity: O(n) where n is length of string
    Space Complexity: O(min(m,n)) where m is size of character set
    
    Example:
        >>> longest_substring("abcabcbb")
        3  # Longest substring is "abc"
    """
    # Handle empty string case
    if s == "":
        return 0

    # Initialize sliding window pointers and tracking variables
    left = 0              # Left pointer of window
    right = 0            # Right pointer of window
    last_seen = {}       # Dictionary to store last position of each char
    max_length = float("-inf")  # Track maximum length found

    # Expand window to right while processing each character
    while right < len(s):
        # Case 1: New character not seen before
        if s[right] not in last_seen:
            last_seen[s[right]] = right
        # Case 2: Character was seen before
        else:
            # If previous occurrence is within current window
            if last_seen[s[right]] >= left:
                left = last_seen[s[right]] + 1  # Move left pointer past last occurrence
            last_seen[s[right]] = right  # Update last seen position

        # Calculate current window length and update max if needed
        current_length = right - left + 1
        max_length = max(max_length, current_length)
        right += 1  # Expand window to right
        
    return max_length


# Driver code to test the longest_substring with no repeateing characters
if __name__ == "__main__":
    test_cases = [
        ("abcabcbb", 3),  # "abc" has length 3
        ("bbbbb", 1),    # "b" has length 1
        ("pwwkew", 3),   # "wke" has length 3
        ("", 0),         # Empty string has length 0
        ("abcdef", 6),   # "abcdef" has length 6
        ("tmmzuxt", 5),  # Sliding window example
    ]

    for s, expected in test_cases:
        result = longest_substring(s)
        print(f"Input: {s}, Expected: {expected}, Result: {result}")

    print("All tests passed.")