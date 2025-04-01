def lengthofLIS(nums):
    """
    Calculate the length of the longest increasing subsequence in a given list of numbers.

    Args:
        nums (list[int]): A list of integers.

    Returns:
        int: The length of the longest increasing subsequence.
    """
    dp = [1] * (
        len(nums)
    )  # Initialize dp array with 1s, as each element is an LIS of length 1

    max_len = 1  # Initialize the maximum length to 1

    for i in range(
        1, len(nums)
    ):  # Iterate through the nums list starting from the second element
        for j in range(i):  # Iterate through the elements before the current element
            if (
                nums[i] > nums[j]
            ):  # If the current element is greater than a previous element
                dp[i] = max(
                    dp[i], dp[j] + 1
                )  # Update the LIS length at the current index

        max_len = max(max_len, dp[i])  # Update the overall maximum LIS length
    return max_len


if __name__ == "__main__":
    test_cases = [
        [10, 9, 2, 5, 3, 7, 101, 18],  # Test case 1
        [0, 1, 0, 3, 2, 3],  # Test case 2
        [7, 7, 7, 7, 7, 7, 7],  # Test case 3
        [1, 3, 6, 7, 9, 4, 10, 5, 6],  # Test case 4
        [5, 4, 3, 2, 1],  # Test case 5
    ]

    for nums in test_cases:  # Iterate through the test cases
        result = lengthofLIS(nums)  # Calculate the LIS length for the current test case
        print(
            f"Input: {nums}, Length of LIS: {result}"
        )  # Print the input and the result
