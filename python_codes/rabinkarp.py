def find_repeated_sequences(dna, k):
    """
    Find all repeated DNA sequences of length k using Rabin-Karp algorithm.
    Uses rolling hash technique for efficient pattern matching.

    Args:
        dna: String representing DNA sequence with characters A, C, G, T
        k: Length of subsequences to find

    Returns:
        set: All repeated subsequences of length k

    Time Complexity: O(n) where n is length of DNA sequence
    Space Complexity: O(n-k+1) for storing hash values and output

    Example:
        >>> find_repeated_sequences("AAAAACCCCCAAAAACCCCCC", 8)
        {'AAAAACCC', 'AAAACCCC'} # These subsequences appear multiple times
    """
    # Handle edge case where k is longer than sequence
    if k > len(dna):
        return set()

    hash = set()  # Store hash values of seen sequences
    output = set()  # Store repeated subsequences
    hashvalue = 0  # Current rolling hash value
    char2int = {"A": 1, "C": 2, "G": 3, "T": 4}  # Map DNA bases to integers
    base = len(char2int.keys())  # Base for polynomial hash function

    # Convert DNA sequence to numeric values for hashing
    input = []
    for i in range(len(dna)):
        input.append(char2int[dna[i]])

    # Process each k-length window in the DNA sequence
    for i in range(len(dna) - k + 1):
        if i == 0:
            # Calculate initial hash value for first window
            for j in range(k):
                hashvalue = hashvalue + input[j] * (base ** (k - j - 1))
        else:
            # Use rolling hash technique for subsequent windows:
            # 1. Remove contribution of character leaving the window
            # 2. Shift remaining hash value
            # 3. Add contribution of new character entering the window
            previous_hash = hashvalue
            hashvalue = (
                (previous_hash - input[i - 1] * (base ** (k - 1))) * base
            ) + input[i + k - 1]

        # If hash value seen before, we found a repeated sequence
        if hashvalue in hash:
            output.add(dna[i : i + k])

        # Remember current hash value
        hash.add(hashvalue)

        # Debug output to visualize algorithm progress
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
    """
    Driver code to test Rabin-Karp DNA sequence finder.
    Tests various DNA sequences with different k values to find repeated patterns.

    Test cases include:
    - Short sequences with no repetitions
    - Sequences with obvious repetitions
    - Edge cases with single-character repetitions
    - Mixed pattern sequences
    """
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
