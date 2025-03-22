def find_repeated_sequences(dna, k):

    # Replace this placeholder return statement with your code
    # Rabin karp algorithm
    if k > len(dna):
        return set()

    hash = set()
    output = set()
    hashvalue = 0
    char2int = {"A": 1, "C": 2, "G": 3, "T": 4}
    base = len(char2int.keys())

    input = []
    for i in range(len(dna)):
        input.append(char2int[dna[i]])

    for i in range(len(dna) - k + 1):
        if i == 0:
            for j in range(k):
                hashvalue = hashvalue + input[j] * (base ** (k - j - 1))
        else:
            previous_hash = hashvalue
            hashvalue = (
                (previous_hash - input[i - 1] * (base ** (k - 1))) * base
            ) + input[i + k - 1]
        if hashvalue in hash:
            output.add(dna[i : i + k])

        hash.add(hashvalue)
        print(
            "\tHash value of ",
            dna[i : i + k],
            ":",
            hashvalue,
            "\n\tHash set: ",
            hash,
            "\n\tOutput: ",
            output,
            "\n",
        )
    return output


def main():
    inputs_string = [
        "ACGT",
        "AGACCTAGAC",
        "AAAAACCCCCAAAAACCCCCC",
        "GGGGGGGGGGGGGGGGGGGGGGGGG",
        "TTTTTCCCCCCCTTTTTTCCCCCCCTTTTTTT",
        "TTTTTGGGTTTTCCA",
        "AAAAAACCCCCCCAAAAAAAACCCCCCCTG",
        "ATATATATATATATAT",
    ]
    inputs_k = [3, 3, 8, 12, 10, 14, 10, 6]

    for i in range(len(inputs_k)):
        print(i + 1, ".\tInput Sequence: '", inputs_string[i], "'", sep="")
        print("\tk: ", inputs_k[i], sep="")
        print(
            "\tRepeated Subsequence: ",
            find_repeated_sequences(inputs_string[i], inputs_k[i]),
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
