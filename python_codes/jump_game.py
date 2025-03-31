def print_array_with_markers(
    arr, p_value1=-1, mrk1a="«", mrk1b="»", p_value2=-1, mrk2a=">", mrk2b="<"
):
    """
    Print array with special markers to highlight specific indices.
    Useful for visualizing array traversal and positions.
    
    Args:
        arr: Input array to print
        p_value1: First index to mark (default: -1, no marking)
        mrk1a: Opening marker for first position (default: «)
        mrk1b: Closing marker for first position (default: »)
        p_value2: Second index to mark (default: -1, no marking)
        mrk2a: Opening marker for second position (default: >)
        mrk2b: Closing marker for second position (default: <)
        
    Returns:
        str: Formatted string representation of array with markers
    """
    out = "["
    for i in range(len(arr)):
        if p_value1 == i:
            out += mrk1a
            out += str(arr[i]) + mrk1b + ", "
        elif p_value2 == i:
            out += mrk2a
            out += str(arr[i]) + mrk2b + ", "
        else:
            out += str(arr[i]) + ", "
    out = out[0 : len(out) - 2]
    out += "]"
    return out


def jump_game(nums: list[int]) -> bool:
    """
    Determine if last index is reachable starting from first index.
    Uses greedy approach moving backwards from target to find valid path.
    
    Args:
        nums: List where nums[i] represents maximum jump length from index i
        
    Returns:
        bool: True if last index is reachable, False otherwise
        
    Time Complexity: O(n) where n is length of input array
    Space Complexity: O(1) as we only use constant extra space
    
    Example:
        >>> jump_game([2,3,1,1,4])
        True  # Can jump: 0->1->4 (using jumps of length 2,3)
    """
    if len(nums) == 1:  # Single element array is always reachable
        return True
        
    target = len(nums) - 1  # Start from last index
    
    # Move backwards through array
    for i in range(len(nums) - 2, -1, -1):
        # If we can reach target from current position
        if i + nums[i] >= target:
            target = i  # Update target to current position

        # If we reached start, path exists
        if target == 0:
            return True
            
    return False


def main():
    """
    Driver code to test jump game functionality with various inputs.
    Tests different array configurations including:
    - Regular cases with solution
    - Impossible cases
    - Edge cases (empty, single element)
    - Various jump patterns
    """
    nums = [
        [3, 2, 2, 0, 1, 4],    # Regular case
        [2, 3, 1, 1, 9],       # Multiple solutions
        [3, 2, 1, 0, 4],       # Impossible case
        [0],                   # Single element
        [1],                   # Single element
        [4, 3, 2, 1, 0],      # Decreasing jumps
        [1, 1, 1, 1, 1],      # Uniform jumps
        [4, 0, 0, 0, 1],      # Zero jumps
        [3, 3, 3, 3, 3],      # Equal jumps
        [1, 2, 3, 4, 5, 6, 7], # Increasing jumps
    ]

    for i in range(len(nums)):
        print(f"{i + 1}.\tInput array: {nums[i]}")
        result = jump_game(nums[i])
        print(f"\tCan we reach the very last index? {'Yes' if result else 'No'}")
        print("-" * 100)


if __name__ == "__main__":
    main()