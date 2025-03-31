def num_of_decodings(decode_str: str) -> int:
    """
    Calculate number of ways to decode a string of digits into letters (A-Z).
    Uses dynamic programming where dp[i] represents number of ways to decode
    string up to index i.

    Rules:
    - '1' to '9' can be decoded to 'A' to 'I'
    - '10' to '26' can be decoded to 'J' to 'Z'
    - '0' cannot be decoded alone

    Args:
        decode_str: String containing digits to decode

    Returns:
        int: Number of possible ways to decode the string

    Time Complexity: O(n) where n is length of input string
    Space Complexity: O(n) for dp array

    Example:
        >>> num_of_decodings("12")
        2  # Can be decoded as "AB" (1,2) or "L" (12)
    """
    # Initialize dp array with base cases
    dp = [0] * (len(decode_str) + 1)
    dp[0] = 1  # Empty string has one way to decode
    dp[1] = 1 if decode_str[0] != "0" else 0  # Single digit

    # Fill dp array for rest of the string
    for i in range(2, len(decode_str) + 1):
        # Case 1: Single digit decode
        if decode_str[i - 1] != "0":
            dp[i] += dp[i - 1]  # Add ways from previous position

        # Case 2: Two digit decode if valid
        if decode_str[i - 2] == "1" or (
            decode_str[i - 2] == "2" and decode_str[i - 1] < "7"
        ):
            dp[i] += dp[i - 2]  # Add ways from two positions back

    return dp[-1]


def main():
    """
    Driver code to test string decoding functionality.
    Tests various string patterns including:
    - Regular numbers
    - Strings with zeros
    - Invalid patterns
    - Long sequences
    """
    decode_str = [
        "124",  # Multiple valid decodings
        "123456",  # Longer sequence
        "11223344",  # Repeated patterns
        "0",  # Invalid - starts with 0
        "0911241",  # Contains zero
        "10203",  # Multiple zeros
        "999901",  # Invalid pattern at end
    ]

    for i, test_str in enumerate(decode_str, 1):
        ways = num_of_decodings(test_str)
        print(f"{i}.\tString: '{test_str}'")
        print(f"\tNumber of ways to decode: {ways}")
        print("-" * 100)


if __name__ == "__main__":
    main()
