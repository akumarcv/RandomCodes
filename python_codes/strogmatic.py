def is_strobogrammatic(num):
    """
    Determine if a number is strobogrammatic.
    A strobogrammatic number is one that reads the same when rotated 180 degrees.

    Args:
        num: String representation of a number to check

    Returns:
        bool: True if the number is strobogrammatic, False otherwise

    Time Complexity: O(n) where n is the length of the input number string
    Space Complexity: O(1) as we use a fixed-size mapping dictionary

    Example:
        >>> is_strobogrammatic("69")
        True  # When rotated 180°, "69" becomes "69"
        >>> is_strobogrammatic("123")
        False  # When rotated 180°, "123" is not a valid number
    """
    # Map of digits that look the same when rotated 180 degrees
    # 0→0, 1→1, 8→8, 6→9, 9→6
    mapping = {"8": "8", "0": "0", "6": "9", "9": "6", "1": "1"}

    # Use two pointers to check from both ends simultaneously
    left = 0
    right = len(num) - 1

    while left <= right:
        # Check if both characters are valid strobogrammatic digits
        if num[left] not in mapping.keys() or num[right] not in mapping.keys():
            return False  # Contains digits that cannot be strobogrammatic (2,3,4,5,7)

        # Check if left digit maps to right digit when rotated
        if mapping[num[left]] == num[right]:
            left = left + 1  # Move left pointer forward
            right = right - 1  # Move right pointer backward
        else:
            return False  # Not a match when rotated

    # If we've checked all digits without finding a mismatch
    return True


# Driver code
def main():
    """
    Test is_strobogrammatic function with various example numbers.

    Test cases include:
    - Valid strobogrammatic numbers (609, 88, 101)
    - Invalid numbers (962, 123, 619)
    - Different lengths and patterns

    Each test displays:
    - The input number
    - Whether it's strobogrammatic (True/False)
    """
    nums = ["609", "88", "962", "101", "123", "619"]

    i = 0
    for num in nums:
        print(i + 1, ".\tnum: ", num, sep="")
        print("\n\tIs strobogrammatic: ", is_strobogrammatic(num), sep="")
        print("-" * 100)
        i += 1


if __name__ == "__main__":
    main()
