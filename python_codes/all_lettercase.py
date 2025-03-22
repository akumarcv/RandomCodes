def helper(s, index, result, slate):
    if index == len(s):
        result.append(slate)
        return
    if s[index].isalpha():
        helper(s, index + 1, result, slate + s[index].lower())
        helper(s, index + 1, result, slate + s[index].upper())
    else:
        helper(s, index + 1, result, slate + s[index])


def letter_case_permutation(s):
    result = []
    helper(s, 0, result, "")
    return result


def main():
    strings = ["a1b2", "3z4", "ABC", "123", "xYz"]

    i = 0
    for s in strings:
        print(i + 1, ".\ts: ", '"', s, '"', sep="")
        print("\n\tOutput: ", letter_case_permutation(s), sep="")
        print("-" * 100)
        i += 1


if __name__ == "__main__":
    main()
