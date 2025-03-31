def helper(nums: list, current: list, sum: int, subsets: list, k: int) -> list:
    """
    Helper function to find all subsets with sum k using backtracking

    Args:
        nums: Input array of numbers to choose from
        current: Current subset being built
        sum: Current sum of elements in subset
        subsets: List to store valid subsets
        k: Target sum to achieve

    Returns:
        List of all subsets that sum to k

    Time Complexity: O(2^n) where n is length of input array
    Space Complexity: O(n) for recursion stack
    """
    # Base case: found valid subset
    if sum == k:
        subsets.append(current)
        return

    # Base case: sum exceeded target
    if sum > k:
        return

    # Try including each remaining number in subset
    for i in range(len(nums)):
        helper(
            nums[i + 1 :],  # Remaining numbers
            current + [nums[i]],  # Add current number to subset
            sum + nums[i],  # Update running sum
            subsets,
            k,
        )  # Pass through other parameters
    return subsets


def get_k_sum_subsets(nums: list, k: int) -> list:
    """
    Find all subsets of an array that sum to k

    Args:
        nums: Input array of integers
        k: Target sum to achieve

    Returns:
        List of all subsets where elements sum to k

    Example:
        >>> get_k_sum_subsets([1,2,3,4], 5)
        [[1,4], [2,3]]
    """
    subsets = []
    helper(nums, [], 0, subsets, k)
    return subsets


def main():
    """
    Driver code to test k-sum subset functionality with various test cases
    """
    # Test cases: pairs of input arrays and target sums
    nums = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    k = [5, 10, 15]

    # Process each test case
    for i in range(len(nums)):
        print(f"{i+1}. Set: {nums[i]}")
        print(f"\tk = {k[i]}")
        print("\tAll subsets with sum k:")
        print(get_k_sum_subsets(nums[i], k[i]))
        print("-" * 100)


if __name__ == "__main__":
    main()
