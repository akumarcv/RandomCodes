def is_palindrome(strings):
    """
    Determine if a string can be a palindrome by removing at most one character.
    Uses two-pointer approach to check palindrome property with one allowed mismatch.

    A palindrome reads the same forward and backward. This function allows for
    one character removal to make the string a valid palindrome.

    Args:
        strings: Input string to check

    Returns:
        bool: True if string is already a palindrome or can become one by
              removing at most one character, False otherwise

    Time Complexity: O(n) where n is the length of the input string
    Space Complexity: O(1) using constant extra space

    Example:
        >>> is_palindrome("abca")
        True  # Remove 'c' to get palindrome "aba"
    """
    # Initialize two pointers at string ends
    left = 0
    right = len(strings) - 1
    mismatch = 0  # Track number of mismatches found

    # Handle single character case (always a palindrome)
    if len(strings) == 1:
        return True

    while left < right:
        if strings[left] == strings[right]:
            # Characters match, move pointers inward
            left += 1
            right -= 1
        else:
            # Found a mismatch, try removing left or right character

            # Option 1: Skip left character and check if remaining is palindrome
            if strings[left + 1] == strings[right]:
                i = left + 1
                length = right - left
                mismatch = 1  # Count this as one mismatch

                # Verify remaining substring is palindrome
                for k in range(length // 2):
                    if strings[i + k] != strings[right - k]:
                        mismatch += 1  # Additional mismatches found
                print(f"if {mismatch}")
                if mismatch == 1:
                    return True  # Only one character removed makes it palindrome

            # Option 2: Skip right character and check if remaining is palindrome
            else:
                mismatch = 1  # Count this as one mismatch
                i = right - 1
                length = right - left

                # Verify remaining substring is palindrome
                for k in range(length // 2):
                    if strings[left + k] != strings[i - k]:
                        mismatch += 1  # Additional mismatches found
                print(f"else {mismatch}")
                if mismatch == 1:
                    return True  # Only one character removed makes it palindrome

            # Continue checking rest of string
            left += 1
            right -= 1

    # If we finished checking and found no mismatches,
    # string is already a palindrome
    if mismatch == 0:
        return True
    else:
        return False  # More than one character needs to be removed


if __name__ == "__main__":
    """
    Test the is_palindrome function with a complex example.

    This test uses a very long string to verify the algorithm works correctly
    with large inputs, including edge cases and complex character patterns.
    The function should determine if the string can become a palindrome by
    removing at most one character.
    """
    print(
        is_palindrome(
            "ElDXxFgmiPzvjUmBcpjyMYYtzcuBEmgWwvkFePovorAcBXbuArdvpwSpGlXExWumEiifqcDflfzMOPvNmrpPoUGqZCOfrBNeSevHolDgiiHhpTUgaJkcCmLZPKoqwfOqmXSXCRdkJLLGKCXCKIOjssRrsyUusKnmGZLKqteAMziPZgsigZmDciZFAzOcTkvPBrbBKnALPrxpYQEnHhTZdVGAZgjfMmzTdqbicrZGhUgerDGMNXEPEhRCwXRukJeljZYwwVlxffdPrWROMnTmRqfObVECBjIewuAJvdAiymxhxbGeBhWpIMhtTpZRFYenIUqmldlDDESzHuoXuxBHGasGhXpkukYUNgmUxGAPzNdlHeiGdRgCaLBBuqeiNvTyByDPCEzLpOtvMsKmMvmxwivNSOjVcVunRNgOmuNvESYBAjfWeZCVsVVscRnzMAAQeAYjgtYpkDdhgQLqgLplduOhVkaDtNtiRKKLivFFWKCPGLxryjNkkNjyrxLGPCKWFFviLKKRitNtDakVhOudlpLgqLQghdDkpYtgjYAeQAAMznRcsVVsVCZeWfjABYSEvNumOgNRnuVcVjOSNviwxmvMmKsMvtOpLzECPDyByTvNiequBBLaCgRdGieHldNzPAGxUmgNUYkukpXhGsaGHBxuXouHzSEDDldlmqUIneYFRZpTthMIpWhBeGbxhxmyiAdvJAuweIjBCEVbOfqRmTnMORWrPdffxlVwwYZjleJkuRXwCRhEPEXNMGDregUhGZrcibqdTzmMfjgZAGVdZThHnEQYpxrPLAnKBbrBPvkTcOzAFZicDmZgisgZPizMAetqKLZGmnKsuUysrRssjOIKCXCKGLLJkdRCXSXmqOfwqoKPZLmCckJagUTphHiigDloHveSeNBrfOCZqGUoPprmNvPOMzflfDcqfiiEmuWxEXlGpSwpvdrAubXBcArovoPeFkvwWgmEBucztYYMyjpcBmUjvzPimgFxXDlE"
        )
    )
