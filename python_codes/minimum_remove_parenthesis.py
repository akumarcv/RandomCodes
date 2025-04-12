def min_remove_parentheses(s):
    """
    Remove the minimum number of invalid parentheses to make the string valid.

    Args:
        s: Input string containing parentheses and other characters

    Returns:
        String with minimum number of parentheses removed to make it valid

    Time Complexity: O(n) where n is the length of the input string
    Space Complexity: O(n) for stack and set storage

    Example:
        Input: "a)b(c)d)"
        Output: "ab(c)d"
    """
    # Track indices of characters to be removed
    indices_to_remove = set()
    stack = []

    # First pass: identify unbalanced parentheses
    for i, char in enumerate(s):
        if char == "(":
            stack.append(i)  # Push opening bracket index
        elif char == ")":
            if stack:  # If there's a matching opening bracket
                stack.pop()  # Remove the match
            else:  # No matching opening bracket
                indices_to_remove.add(i)  # Mark this closing bracket for removal

    # Add any remaining unclosed parentheses to the removal set
    indices_to_remove.update(stack)

    # Second pass: build result string excluding marked indices
    result = ""
    for i, char in enumerate(s):
        if i not in indices_to_remove:
            result += char

    return result


# Driver code to test the function
if __name__ == "__main__":

    # Test cases with expected results
    test_cases = [
        ("a)b(c)d)", "ab(c)d"),  # Extra closing parenthesis
        ("(a(b(c)d)", "a(b(c)d)"),  # Extra opening parenthesis
        (")(", ""),  # All invalid
        ("()", "()"),  # Already valid
        ("((())", "(())"),  # One extra opening
        ("())", "()"),  # One extra closing
        ("())()", "()()"),  # Middle invalid
        ("(()()(", "(()()"),  # Trailing opening
        ("", ""),  # Empty string
        ("abc", "abc"),  # No parentheses
        ("(a)b)c(d)e)f", "(a)bc(d)ef"),  # Multiple invalid
    ]

    print("Testing minimum_remove_parentheses function:")
    print("-" * 50)

    # Run each test case and compare with expected result
    for i, (test_input, expected) in enumerate(test_cases):
        result = min_remove_parentheses(test_input)
        status = "✓ PASS" if result == expected else "✗ FAIL"

        print(f"Test {i+1}:")
        print(f"  Input:    '{test_input}'")
        print(f"  Expected: '{expected}'")
        print(f"  Result:   '{result}'")
        print(f"  Status:   {status}")
        print("-" * 50)

    # Count passed tests
    passed = sum(
        1
        for test_input, expected in test_cases
        if min_remove_parentheses(test_input) == expected
    )
    print(f"Passed {passed}/{len(test_cases)} tests")
