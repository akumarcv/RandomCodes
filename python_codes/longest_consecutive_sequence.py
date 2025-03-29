def longest_consecutive_sequence(nums):
    num_set = set(nums)
    longest_streak = 0

    for n in num_set:
        if n - 1 not in num_set:
            current_num = n
            curr_streak = 1

            while current_num + 1 in num_set:
                current_num += 1
                curr_streak += 1

            longest_streak = max(curr_streak, longest_streak)
    return longest_streak


# ...existing code...


def test_longest_consecutive_sequence():
    """
    Test cases for finding longest consecutive sequence
    Each test case contains an array and expected length of longest consecutive sequence
    """
    test_cases = [
        ([100, 4, 200, 1, 3, 2], 4),  # Basic case [1,2,3,4]
        ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9),  # Longer sequence [0,1,2,3,4,5,6,7,8]
        ([], 0),  # Empty array
        ([1], 1),  # Single element
        ([1, 2, 0, 1], 3),  # Sequence with duplicates [0,1,2]
        ([1, 2, 3, 4, 5], 5),  # Already consecutive
        ([9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6], 7),  # Negative numbers included
    ]

    for i, (nums, expected) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input array: {nums}")
        print(f"Expected longest sequence length: {expected}")

        result = longest_consecutive_sequence(nums)
        print(f"Got sequence length: {result}")

        assert (
            result == expected
        ), f"Test case {i} failed! Expected {expected}, got {result}"
        print("✓ Passed")

        # Print the actual sequence if array is not empty
        if nums:
            sequence = []
            start = min(nums)
            while len(sequence) < result and start <= max(nums):
                if start in set(nums):
                    sequence.append(start)
                start += 1
            print(f"Longest consecutive sequence: {sequence}")


if __name__ == "__main__":
    test_longest_consecutive_sequence()
    print("\nAll test cases passed!")
