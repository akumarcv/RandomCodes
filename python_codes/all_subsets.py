def find_all_subsets(nums: list) -> list[list]:
    """
    Generate all possible subsets of a given list using bit manipulation.

    Args:
        nums: Input list of integers to generate subsets from

    Returns:
        List of all possible subsets (each subset is a list)

    Time Complexity: O(n * 2^n) where n is length of input list
    Space Complexity: O(2^n) to store all subsets

    Example:
        >>> find_all_subsets([1, 2])
        [[], [1], [2], [1, 2]]
    """
    # Handle empty input case
    if len(nums) == 0:
        return [[]]

    # Calculate total number of possible subsets
    total_subsets = 2 ** len(nums)
    subsets = []

    # Generate each subset using bit manipulation
    for i in range(total_subsets):
        subset = []
        # Check each bit position
        for j in range(len(nums)):
            # If jth bit is set in i, include jth element
            if i & (1 << j):
                subset.append(nums[j])
        subsets.append(subset)

    # Print current subset configuration
    print("\tAll subsets:", subsets)
    print(f"\tTotal number of subsets: {len(subsets)}")
    return subsets


def find_subsets_recursive(nums: list, index: int, current: list, result: list) -> None:
    """
    Helper function to generate all subsets using recursion.

    Args:
        nums: Input list of integers
        index: Current index being processed
        current: Current subset being built
        result: List to store all subsets

    Time Complexity: O(2^n) where n is length of input list
    Space Complexity: O(n) for recursion stack
    """
    # Base case: processed all elements
    if index == len(nums):
        result.append(current[:])  # Add copy of current subset
        return

    # Case 1: Include current element
    current.append(nums[index])
    find_subsets_recursive(nums, index + 1, current, result)

    # Case 2: Exclude current element
    current.pop()  # Backtrack
    find_subsets_recursive(nums, index + 1, current, result)


def find_all_subsets(nums: list) -> list[list]:
    """
    Generate all possible subsets of a given list using recursion.

    Args:
        nums: Input list of integers to generate subsets from

    Returns:
        List of all possible subsets (each subset is a list)

    Time Complexity: O(2^n) where n is length of input list
    Space Complexity: O(2^n) to store all subsets

    Example:
        >>> find_all_subsets([1, 2])
        [[], [1], [2], [1, 2]]
    """
    result = []
    find_subsets_recursive(nums, 0, [], result)
    return result


def main():
    """
    Driver code to test subset generation with various inputs.
    Tests empty set, different sized sets, and different number configurations.
    """
    # Test cases with increasing complexity
    nums = [
        [],  # Empty set
        [2, 5, 7],  # 3 elements
        [1, 2],  # 2 elements
        [1, 2, 3, 4],  # 4 elements
        [7, 3, 1, 5],  # 4 elements, different order
    ]

    for i, num_set in enumerate(nums, 1):
        print(f"{i}. Input Set: {num_set}")
        find_all_subsets(num_set)
        print("-" * 100)


if __name__ == "__main__":
    main()
