def remove_duplicates(s):
    """
    Removes adjacent duplicate characters from a string.

    This function iterates through the string and uses a stack to track characters.
    When a character is the same as the top of the stack, both are removed (adjacent duplicates).
    Otherwise, the new character is added to the stack.

    Args:
        s (str): Input string to process

    Returns:
        str: String with all adjacent duplicate characters removed

    Examples:
        >>> remove_duplicates("abbaca")
        "ca"
        >>> remove_duplicates("azxxzy")
        "ay"
    """

    # Handle empty string case
    if s == "":
        return ""

    stack = []

    # Add the first character to start the process
    stack.append(s[0])

    # Process remaining characters
    for i in range(1, len(s)):
        if stack:
            # If current char matches top of stack, remove the duplicate (pop)
            if s[i] == stack[-1]:
                stack.pop()
            else:
                # Otherwise add the new character to the stack
                stack.append(s[i])
        else:
            # If stack is empty, simply add the character
            stack.append(s[i])

    # Convert stack to string and return
    return "".join(stack)


# Driver code
if __name__ == "__main__":
    # Test cases
    test_cases = ["abbaca", "azxxzy", "", "aa", "aaa", "abc", "abccba"]

    # Expected results for each test case
    expected_results = ["ca", "ay", "", "", "a", "abc", ""]

    # Run tests and compare with expected results
    print("Testing remove_duplicates function:")
    print("-" * 50)

    for i, test in enumerate(test_cases):
        result = remove_duplicates(test)
        expected = expected_results[i]

        print(f"Test {i+1}: Input = '{test}'")
        print(f"Expected: '{expected}'")
        print(f"Result:   '{result}'")

        # Check if result matches expected
        if result == expected:
            print("✓ PASS")
        else:
            print("✗ FAIL")
        print("-" * 50)
