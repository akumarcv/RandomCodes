def is_happy_number(n: int) -> bool:
    """
    Determine if a number is happy using Floyd's Cycle Detection algorithm.
    A happy number is defined by the following process:
    1. Replace the number by sum of squares of its digits
    2. Repeat until number equals 1 (happy) or enters a cycle (not happy)

    Args:
        n: Integer number to check

    Returns:
        bool: True if number is happy, False otherwise

    Time Complexity: O(log n) for digit extraction and sum calculation
    Space Complexity: O(1) as we only use two pointers

    Example:
        >>> is_happy_number(19)
        True  # 1² + 9² = 82 -> 8² + 2² = 68 -> 6² + 8² = 100 -> 1² + 0² + 0² = 1
    """

    def helper(num: int) -> int:
        """
        Calculate sum of squares of digits for given number.

        Args:
            num: Number to process

        Returns:
            int: Sum of squares of digits
        """
        slow_pointer = num
        fast_pointer = sum(int(digit) ** 2 for digit in str(slow_pointer))
        return fast_pointer

    # Initialize Floyd's algorithm pointers
    slow_pointer = n
    fast_pointer = helper(n)

    # Move pointers until cycle is detected or 1 is found
    while fast_pointer != 1 and fast_pointer != slow_pointer:
        slow_pointer = helper(slow_pointer)
        fast_pointer = helper(helper(fast_pointer))  # Move twice as fast

    # Number is happy if we found 1, unhappy if we detected a cycle
    return fast_pointer == 1


def main():
    """
    Driver code to test happy number detection with various inputs.
    Tests different numbers including:
    - Known happy numbers
    - Known unhappy numbers
    - Edge cases
    """
    test_numbers = [
        7,  # Happy number
        19,  # Happy number
        2,  # Unhappy number
        4,  # Unhappy number
        1,  # Happy number (base case)
        13,  # Happy number
        89,  # Happy number
    ]

    for i, num in enumerate(test_numbers, 1):
        result = is_happy_number(num)
        print(f"{i}. Testing number: {num}")
        print(f"\tIs happy number? {result}")
        if result:
            # Show the sequence for happy numbers
            current = num
            sequence = [current]
            while current != 1:
                current = sum(int(digit) ** 2 for digit in str(current))
                sequence.append(current)
            print(f"\tSequence: {' -> '.join(map(str, sequence))}")
        print("-" * 80)


if __name__ == "__main__":
    main()
