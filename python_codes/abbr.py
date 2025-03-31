def valid_word_abbreviation(word: str, abbr: str) -> bool:
    """
    Validates if a given abbreviation is valid for a word.

    An abbreviation can be created by replacing any number of non-adjacent,
    non-empty substrings with their lengths. The resulting abbreviation
    must not have any leading zeros.

    Args:
        word (str): Original word to check against
        abbr (str): Abbreviation to validate

    Returns:
        bool: True if abbreviation is valid, False otherwise

    Time Complexity: O(n) where n is length of abbreviation
    Space Complexity: O(1)

    Examples:
        >>> valid_word_abbreviation("internationalization", "i12iz4n")
        True
        >>> valid_word_abbreviation("apple", "a2e")
        True
        >>> valid_word_abbreviation("hello", "h02o")
        False  # Leading zero not allowed
    """
    c1 = 0  # Pointer for traversing abbreviation
    c2 = 0  # Pointer for traversing original word

    while c1 < len(abbr):
        # Case 1: Current character is a digit
        if abbr[c1].isdigit():
            # Leading zeros are not allowed in abbreviation
            if abbr[c1] == "0":
                return False

            # Extract the complete number from abbreviation
            count = []
            while c1 < len(abbr) and abbr[c1].isdigit():
                count.append(abbr[c1])
                c1 += 1

            # Convert extracted digits to number and advance word pointer
            count = int("".join(count))
            c2 += count

        # Case 2: Current character is a letter
        else:
            # Check if word pointer is valid and characters match
            if c2 >= len(word) or word[c2] != abbr[c1]:
                return False
            c1 += 1
            c2 += 1

    # Valid only if both pointers have reached the end
    return c1 == len(abbr) and c2 == len(word)


if __name__ == "__main__":
    print(valid_word_abbreviation("helloworld", "4orworld"))
