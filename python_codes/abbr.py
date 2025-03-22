def valid_word_abbreviation(word, abbr):
    c1 = 0  # Pointer for abbreviation
    c2 = 0  # Pointer for word
    while c1 < len(abbr):
        if abbr[c1].isdigit():
            if abbr[c1] == "0":  # Leading zeros are not allowed
                return False
            count = []
            while c1 < len(abbr) and abbr[c1].isdigit():
                count.append(abbr[c1])
                c1 += 1
            count = int("".join(count))  # Convert list of digits to integer
            c2 += count  # Move the word pointer by the count
        else:
            # Check for character match
            if c2 >= len(word) or word[c2] != abbr[c1]:
                return False
            c1 += 1
            c2 += 1

    # Ensure both pointers reach the end
    return c1 == len(abbr) and c2 == len(word)


if __name__ == "__main__":
    print(valid_word_abbreviation("helloworld", "4orworld"))
