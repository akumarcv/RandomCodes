def permute_word(word: str) -> list[str]:
    """
    Generate all possible permutations of a given string using recursion.
    
    Args:
        word: Input string to generate permutations for
        
    Returns:
        List of all possible permutations of the input string
        
    Time Complexity: O(n!) where n is length of string
    Space Complexity: O(n!) to store all permutations
    
    Example:
        >>> permute_word("abc")
        ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
    """
    # Base case: empty string has one permutation
    if word == "":
        return [""]
    
    # Recursive case: generate permutations using each character as first
    else:
        result = []
        for i in range(len(word)):
            # Choose current character as first
            first = word[i]
            # Get remaining characters
            rest = word[:i] + word[i + 1:]
            
            # Recursively get permutations of remaining characters
            for p in permute_word(rest):
                # Add current character to front of each sub-permutation
                result.append(first + p)
        return result


def main():
    """
    Driver code to test permutation generation with various inputs.
    Tests strings of different lengths and contents.
    """
    # Test cases with increasing complexity
    input_word = ["ab", "bad", "abcd"]

    for index, word in enumerate(input_word, 1):
        permuted_words = permute_word(word)

        # Print input and all permutations with proper formatting
        print(f"{index}.\t Input string: '{word}'")
        print(f"\t All possible permutations are: [{', '.join(permuted_words)}]")
        print(f"\t Number of permutations: {len(permuted_words)}")
        print("-" * 100)


if __name__ == "__main__":
    main()