def jump_game_two(nums):
    """
    Function to find the minimum number of jumps required to reach the last index.
    :param nums: List of integers representing the maximum jump length from that position.
    :return: Minimum number of jumps required to reach the last index.
    """
    farthest_jump = 0
    current_jump = 0
    jumps = 0

    for i in range(len(nums) - 1):
        farthest_jump = max(farthest_jump, i + nums[i])
        if i == current_jump:
            jumps += 1
            current_jump = farthest_jump

    return jumps


def main():
    """
    Main function to test the jump_game_two function with different inputs.
    """
    nums = [
        [2, 3, 1, 1, 9],
        [3, 2, 1, 1, 4],
        [4, 0, 0, 0, 4],
        [1, 1],
        [1]
    ]

    for i, num in enumerate(nums):
        print(f"{i + 1}. Input array: {num}")
        print(f"\tMinimum number of jumps required to reach the last index: {jump_game_two(num)}")
        print("-" * 100)


if __name__ == '__main__':
    main()