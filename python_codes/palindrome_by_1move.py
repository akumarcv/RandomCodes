def is_palindrome(strings):
    """
    Check if a string can be made into a palindrome by removing at most one character.

    A palindrome is a string that reads the same backward as forward.
    This function determines if the input string is already a palindrome or
    can become a palindrome after removing exactly one character.

    Algorithm overview:
    1. Use two pointers (left and right) to compare characters from both ends
    2. If characters match, move pointers inward
    3. If characters don't match, try two possibilities:
       a. Skip the character at left pointer and check if remaining is palindrome
       b. Skip the character at right pointer and check if remaining is palindrome
    4. If either attempt creates a palindrome with exactly one removal, return True

    Args:
        strings (str): The input string to check

    Returns:
        bool: True if the string is already a palindrome or can become one
              by removing at most one character, False otherwise

    Examples:
        >>> is_palindrome("abca")
        True  # Removing 'c' makes it "aba", which is a palindrome
        >>> is_palindrome("racecar")
        True  # Already a palindrome
        >>> is_palindrome("abcd")
        False  # Cannot become a palindrome by removing one character
    """
    # Initialize two pointers: left at the beginning and right at the end of string
    left = 0
    right = len(strings) - 1

    # Track number of character mismatches found
    mismatch = 0

    # Handle edge case: single character strings are always palindromes
    if len(strings) == 1:
        return True

    # Use two pointers approach to check palindrome property
    # Continue until pointers meet in the middle
    while left < right:
        if strings[left] == strings[right]:
            # Case 1: Characters match, so this pair is palindromic
            # Move pointers inward to check next character pair
            left += 1
            right -= 1
        else:
            # Case 2: Characters don't match, so we need to try removing one character
            # We have two choices: remove character at left OR remove character at right

            # Option 1: Try removing character at left pointer position
            # Check if character after left matches with right
            if strings[left + 1] == strings[right]:
                # Start checking from left+1 since we're skipping left character
                i = left + 1
                length = right - left
                # Count this as our first mismatch (the one we're handling)
                mismatch = 1

                # Verify the remaining substring is a palindrome by checking pairs
                # We only need to check half the length since we're matching from both sides
                for k in range(length // 2):
                    # Compare characters moving inward from both sides
                    if strings[i + k] != strings[right - k]:
                        # Found another mismatch - can't make palindrome with just one removal
                        mismatch += 1

                # If we found exactly one mismatch (the one we handled), it's a valid solution
                if mismatch == 1:
                    return True

            # Option 2: Try removing character at right pointer position
            else:
                # Count this as our first mismatch (the one we're handling)
                mismatch = 1
                # Start checking from right-1 since we're skipping right character
                i = right - 1
                length = right - left

                # Verify the remaining substring is a palindrome by checking pairs
                for k in range(length // 2):
                    # Compare characters moving inward from both sides
                    if strings[left + k] != strings[i - k]:
                        # Found another mismatch - can't make palindrome with just one removal
                        mismatch += 1

                # If we found exactly one mismatch (the one we handled), it's a valid solution
                if mismatch == 1:
                    return True

            # Continue checking rest of the string (though this path usually won't find solutions
            # if we've already determined we need more than one character removal)
            left += 1
            right -= 1

    # If we went through entire string:
    # 1. mismatch=0: The string is already a palindrome
    # 2. mismatch>0: We couldn't make a palindrome with just one deletion
    return False if mismatch > 0 else True


# Driver code to test the function
if __name__ == "__main__":
    # Test cases with explanations of expected outcomes
    test_cases = [
        "aba",  # Already a palindrome (reads same forward and backward)
        "abca",  # Can become palindrome by removing 'c' to get "aba"
        "abc",  # Cannot become palindrome by removing any single character
        "racecar",  # Already a palindrome (reads same forward and backward)
        "abcdba",  # Can become palindrome by removing 'd' to get "abcba"
        "abcdefdba",  # Cannot become palindrome with one deletion (needs multiple changes)
        "a",  # Single character is always a palindrome by definition
        "ab",  # Can become palindrome by removing either 'a' or 'b' to get a single character
        "radar",  # Already a palindrome (reads same forward and backward)
        "madam",  # Already a palindrome (reads same forward and backward)
    ]

    print("Testing if strings can be made into palindromes with at most one deletion:")
    print("=" * 75)
    print(f"{'String':<15} | {'Result':<10} | {'Explanation'}")
    print("-" * 75)

    for test in test_cases:
        # Run algorithm on each test case
        result = is_palindrome(test)
        explanation = ""

        if result:
            # If algorithm returns True, determine if it's already a palindrome
            # or which character would need to be removed
            is_already_palindrome = test == test[::-1]
            if is_already_palindrome:
                explanation = "Already a palindrome"
            else:
                # For each position, try removing that character and check if result is palindrome
                # This helps explain which specific character can be removed
                for i in range(len(test)):
                    # Create temporary string without character at position i
                    temp = test[:i] + test[i + 1 :]
                    # Check if resulting string is a palindrome
                    if temp == temp[::-1]:
                        explanation = f"Remove '{test[i]}' at index {i}"
                        break
        else:
            explanation = "Cannot become palindrome with one deletion"

        print(f"{test:<15} | {str(result):<10} | {explanation}")

    print("=" * 75)

    # Interactive testing section allows user to input custom strings to check
    print("\nInteractive testing:")
    while True:
        # Get input from user (or exit if user types 'q')
        user_input = input("\nEnter a string to check (or 'q' to quit): ")
        if user_input.lower() == "q":
            break

        # Run algorithm on user input
        result = is_palindrome(user_input)
        print(f"Result: {result}")

        # Provide detailed explanation of the result
        if result:
            # Check if it's already a palindrome without any changes
            if user_input == user_input[::-1]:
                print("Explanation: Already a palindrome")
            else:
                # Find which specific character can be removed to make a palindrome
                for i in range(len(user_input)):
                    temp = user_input[:i] + user_input[i + 1 :]
                    if temp == temp[::-1]:
                        print(f"Explanation: Remove '{user_input[i]}' at index {i}")
                        break
        else:
            print("Explanation: Cannot become palindrome with one deletion")
