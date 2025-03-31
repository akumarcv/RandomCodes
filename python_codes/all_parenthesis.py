def helper(n: int, left: int, right: int, result: list, current: str = "") -> None:
    """
    Helper function to generate valid parentheses combinations using backtracking.

    Args:
        n: Number of pairs of parentheses to generate
        left: Count of opening parentheses used so far
        right: Count of closing parentheses used so far
        result: List to store valid combinations
        current: Current combination being built

    Time Complexity: O(2^(2n)) - each position can be '(' or ')'
    Space Complexity: O(n) for recursion stack
    """
    # Base case: used all parentheses pairs
    if left == n and right == n:
        result.append(current)
        return

    # Add opening parenthesis if we haven't used all n
    if left < n:
        helper(n, left + 1, right, result, current + "(")

    # Add closing parenthesis if it maintains validity
    if right < left:
        helper(n, left, right + 1, result, current + ")")


def generate_combinations(n: int) -> list:
    """
    Generate all valid combinations of n pairs of parentheses.

    Args:
        n: Number of pairs of parentheses to generate

    Returns:
        List of strings containing all valid combinations

    Example:
        >>> generate_combinations(2)
        ['(())', '()()']
    """
    result = []
    helper(n, 0, 0, result, "")
    return result


def print_result(result: list) -> None:
    """
    Print each parentheses combination on a new line with proper formatting.

    Args:
        result: List of valid parentheses combinations to print
    """
    for rs in result:
        print("\t\t ", rs)


def main():
    """
    Driver code to test parentheses generation with various inputs.
    Tests n from 1 to 5 pairs of parentheses.
    """
    # Test cases: 1 to 5 pairs of parentheses
    n = [1, 2, 3, 4, 5]

    for i in range(len(n)):
        print(f"{i+1}.\t n = {n[i]}")
        print("\t All combinations of valid balanced parentheses: ")

        result = generate_combinations(n[i])
        print_result(result)

        print("-" * 100)


if __name__ == "__main__":
    main()
