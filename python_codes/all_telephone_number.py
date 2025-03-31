def helper(digits: str, digits_mapping: dict, combinations: list, 
          index: int, current_combination: str) -> list:
    """
    Helper function to generate letter combinations using backtracking.
    
    Args:
        digits: Input string of digits
        digits_mapping: Dictionary mapping digits to possible letters
        combinations: List to store valid combinations
        index: Current digit position being processed
        current_combination: Current letter combination being built
        
    Returns:
        List of all valid letter combinations
        
    Time Complexity: O(4^N) where N is length of digits string
    Space Complexity: O(N) for recursion stack
    """
    # Base case: processed all digits
    if index == len(digits):
        combinations.append(current_combination)
        return combinations
        
    # Try each letter mapped to current digit
    for letter in digits_mapping[digits[index]]:
        helper(digits, digits_mapping, combinations, 
               index + 1, current_combination + letter)
    
    return combinations

def letter_combinations(digits: str) -> list:
    """
    Generate all possible letter combinations for a given phone number.
    
    Args:
        digits: String containing digits from 2-9
        
    Returns:
        List of all possible letter combinations
        
    Example:
        >>> letter_combinations("23")
        ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
    """
    combinations = []
    
    # Handle empty input
    if len(digits) == 0:
        return []
    
    # Phone keypad mapping
    digits_mapping = {
        "1": [""],
        "2": ["a", "b", "c"],
        "3": ["d", "e", "f"],
        "4": ["g", "h", "i"],
        "5": ["j", "k", "l"],
        "6": ["m", "n", "o"],
        "7": ["p", "q", "r", "s"],
        "8": ["t", "u", "v"],
        "9": ["w", "x", "y", "z"],
    }
    
    return helper(digits, digits_mapping, combinations, 0, "")

def main():
    """
    Driver code to test letter combination generation with various inputs.
    Tests different length digit strings and combinations.
    """
    digits_array = ["23", "73", "426", "78", "925", "2345"]
    
    for i, digits in enumerate(digits_array, 1):
        combinations = letter_combinations(digits)
        print(f"{i}.\tInput digits: '{digits}'")
        print(f"\tNumber of combinations: {len(combinations)}")
        print(f"\tAll combinations: {combinations}")
        print("-" * 100)

if __name__ == "__main__":
    main()