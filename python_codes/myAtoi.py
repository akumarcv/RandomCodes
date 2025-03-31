import sys, os, pdb


class Solution:
    """
    Solution class for string to integer (atoi) conversion.
    Implements functionality similar to C's atoi function with proper bounds checking.
    """

    def myAtoi(self, st):
        """
        Convert string to 32-bit signed integer following these rules:
        1. Ignore leading whitespace
        2. Check for optional +/- sign
        3. Read digits until non-digit or end of string
        4. Clamp result to 32-bit signed integer range

        Args:
            st (str): Input string to convert

        Returns:
            int: Converted integer value with 32-bit bounds
            None: If input is None

        Time Complexity: O(n) where n is length of input string
        Space Complexity: O(1) using constant extra space

        Example:
            >>> Solution().myAtoi("42")
            42
            >>> Solution().myAtoi("   -42")
            -42
            >>> Solution().myAtoi("4193 with words")
            4193
        """
        if st is None:
            return st  # Handle None input

        sign = 1  # Default positive sign
        num = 0  # Accumulated number value
        numberhasstarted = False  # Track if we've started parsing digits/sign
        numberdone = False  # Track if parsing should stop
        count = 0

        for i in st:
            if not numberdone:
                if i == " ":  # Handle whitespace
                    if not numberhasstarted:
                        continue  # Skip leading whitespace
                    else:
                        numberdone = True  # Stop if space after digits/sign
                elif i == "-" or i == "+":  # Handle sign characters
                    if not numberhasstarted:
                        sign = 1 if i == "+" else -1  # Set sign based on character
                        numberhasstarted = True  # Mark parsing as started
                    else:
                        numberdone = True  # Extra signs not allowed
                elif ord(i) >= ord("0") and ord(i) <= ord("9"):  # Handle digits
                    num = num * 10 + (
                        ord(i) - 48
                    )  # Convert char to int and add to result
                    numberhasstarted = True  # Mark parsing as started
                else:
                    numberdone = (
                        True  # Stop on non-numeric, non-whitespace, non-sign char
                    )

        # Check 32-bit integer bounds and clamp if needed
        if sign * num > ((2**31) - 1):
            return 2**31 - 1  # Max 32-bit integer (2^31-1)
        if sign * num < (-(2**31)):
            return -(2**31)  # Min 32-bit integer (-2^31)

        return sign * num  # Return final signed number


# Test code
obj = Solution()
st = "  +- 1213"  # Test case: invalid input with conflicting signs
val = obj.myAtoi(st)
print(val)  # Expected: 0 (parsing stops when reaching the unexpected second sign)
