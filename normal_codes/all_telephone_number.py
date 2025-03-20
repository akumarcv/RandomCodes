def helper(digits, digits_mapping, combinations, index, current_combination):
    if index == len(digits):
        combinations.append(current_combination)
        return combinations
    for i in range(len(digits_mapping[digits[index]])):
       helper(digits, digits_mapping, combinations, index+1, current_combination+digits_mapping[digits[index]][i])
    
    return combinations

def letter_combinations(digits):
    combinations = []
    
    if len(digits) == 0:
        return []

    digits_mapping = {
        "1": [""],
        "2": ["a", "b", "c"],
        "3": ["d", "e", "f"],
        "4": ["g", "h", "i"],
        "5": ["j", "k", "l"],
        "6": ["m", "n", "o"],
        "7": ["p", "q", "r", "s"],
        "8": ["t", "u", "v"],
        "9": ["w", "x", "y", "z"]}
    
    return helper(digits, digits_mapping, combinations, 0, "")
    
# driver code
def main():
    digits_array = ["23", "73", "426", "78", "925", "2345"]
    counter = 1
    for digits in digits_array:
        print(counter, ".\t All letter combinations for '",
              digits, "': ", letter_combinations(digits), sep="")
        counter += 1
        print("-" * 100)


if __name__ == "__main__":
    main()
    