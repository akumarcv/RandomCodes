def longest_substring(s):
    if s == "":
        return 0

    left = 0
    right = 0
    last_seen = {}
    max_length = float("-inf")
    while right < len(s):
        if s[right] not in last_seen:
            last_seen[s[right]] = right
        else:
            if last_seen[s[right]] >= left:
                left = last_seen[s[right]] + 1
            last_seen[s[right]] = right
        current_length = right - left + 1
        max_length = max(max_length, current_length)
        right += 1
    return max_length


# Driver code to test the longest_substring with no repeateing characters
if __name__ == "__main__":
    test_cases = [
        ("abcabcbb", 3),  # "abc" has length 3
        ("bbbbb", 1),  # "b" has length 1
        ("pwwkew", 3),  # "wke" has length 3
        ("", 0),  # Empty string has length 0
        ("abcdef", 6),  # "abcdef" has length 6
        ("tmmzuxt", 5),  # "ab" has length 2
    ]

    for s, expected in test_cases:
        result = longest_substring(s)
        print(f"Input: {s}, Expected: {expected}, Result: {result}")

    print("All tests passed.")
