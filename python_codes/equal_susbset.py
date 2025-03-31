def can_partition_array(nums: list[int]) -> bool:
    """
    Determine if array can be partitioned into two subsets of equal sum.
    Uses dynamic programming with a 2D table where:
    dp[i][j] = whether subset of first i numbers can sum to j

    Args:
        nums: List of positive integers to partition

    Returns:
        bool: True if array can be partitioned equally, False otherwise

    Time Complexity: O(n*m) where n is length of array and m is sum/2
    Space Complexity: O(n*m) for DP table

    Example:
        >>> can_partition_array([1,5,11,5])
        True  # Can be partitioned as [1,5,5] and [11]
    """
    # Calculate target sum for each partition
    sum_nums = sum(nums)
    if sum_nums % 2 != 0:  # Odd sum can't be partitioned equally
        return False

    target = sum_nums // 2

    # Initialize DP table
    # dp[i][j] = can we make sum j using first i numbers
    dp = [[False for _ in range(target + 1)] for _ in range(len(nums) + 1)]

    # Empty subset can make sum 0
    for i in range(len(nums) + 1):
        dp[i][0] = True

    # Fill DP table
    for i in range(1, len(nums) + 1):
        for j in range(1, target + 1):
            if j >= nums[i - 1]:
                # Either exclude current number (dp[i-1][j])
                # Or include it (dp[i-1][j-nums[i-1]])
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
            else:
                # Current number too large, must exclude
                dp[i][j] = dp[i - 1][j]

    return dp[-1][-1]


def main():
    """
    Driver code to test array partitioning with various inputs.
    Tests different array configurations including:
    - Even/odd total sums
    - Various array sizes
    - Different number distributions
    """
    nums = [
        [1, 5, 11, 5],  # Can be partitioned
        [1, 2, 3, 5],  # Cannot be partitioned
        [1, 2, 3, 4, 5, 6, 7],  # Equal partition possible
        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Larger array
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Even sum
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # Odd sum
    ]

    for i, arr in enumerate(nums, 1):
        print(f"{i}.\tArray: {arr}")
        result = can_partition_array(arr)
        print(f"\tCan be partitioned equally: {result}")
        if result:
            # Calculate and show the partitions for better understanding
            target = sum(arr) // 2
            print(f"\tTarget sum for each partition: {target}")
        print("-" * 100)


if __name__ == "__main__":
    main()
