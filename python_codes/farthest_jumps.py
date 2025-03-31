def jump_game_two(nums: list[int]) -> int:
    """
    Find minimum number of jumps required to reach last index in array.
    Uses greedy approach to track farthest possible jump at each step.

    Args:
        nums: List of integers where nums[i] represents maximum jump length
             from position i

    Returns:
        int: Minimum number of jumps needed to reach last index

    Time Complexity: O(n) where n is length of input array
    Space Complexity: O(1) as we only use constant extra space

    Example:
        >>> jump_game_two([2,3,1,1,4])
        2  # First jump to index 1, then jump to last index
    """
    # Edge cases
    if len(nums) <= 1:
        return 0

    farthest_jump = 0  # Tracks farthest reachable position
    current_jump = 0  # Current jump endpoint
    jumps = 0  # Number of jumps taken

    # Iterate through array (except last element)
    for i in range(len(nums) - 1):
        # Update farthest possible jump from current position
        farthest_jump = max(farthest_jump, i + nums[i])

        # When we reach current jump endpoint
        if i == current_jump:
            jumps += 1  # Take a jump
            current_jump = farthest_jump  # Update jump endpoint

    return jumps


def main():
    """
    Driver code to test jump game functionality with various inputs.
    Tests different array configurations including:
    - Regular cases with multiple jumps
    - Arrays requiring maximum jumps
    - Single element arrays
    - Arrays with different jump patterns
    """
    test_cases = [
        [2, 3, 1, 1, 9],  # Regular case
        [3, 2, 1, 1, 4],  # Multiple possible paths
        [4, 0, 0, 0, 4],  # Large initial jump
        [1, 1],  # Minimum case
        [1],  # Single element
        [5, 9, 3, 2, 1, 0, 2, 3, 3, 1, 0, 0],  # Longer sequence
    ]

    for i, nums in enumerate(test_cases, 1):
        print(f"{i}. Input array: {nums}")
        jumps = jump_game_two(nums)
        print(f"\tMinimum jumps required: {jumps}")

        # Show possible path for better understanding
        if len(nums) > 1:
            pos = 0
            path = [0]
            while pos < len(nums) - 1:
                next_pos = min(pos + nums[pos], len(nums) - 1)
                path.append(next_pos)
                pos = next_pos
            print(f"\tPossible path: {path}")
        print("-" * 100)


if __name__ == "__main__":
    main()
