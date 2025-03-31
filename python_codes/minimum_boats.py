def rescue_boats(people: list[int], limit: int) -> int:
    """
    Find minimum number of boats needed to rescue people with weight limit.
    Uses two-pointer approach to pair lightest person with heaviest possible person.

    Args:
        people: List of integers representing weights of people
        limit: Maximum weight capacity of each boat

    Returns:
        int: Minimum number of boats required

    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(1) as we sort in-place

    Example:
        >>> rescue_boats([3,2,2,1], 3)
        3  # Can pair [1,2], [2], [3] in three boats
    """
    # Sort people by weight for efficient pairing
    people.sort()
    i, j = 0, len(people) - 1  # Two pointers: lightest and heaviest
    count = 0  # Track number of boats needed

    # Try to pair people while pointers haven't crossed
    while i <= j:
        # If lightest and heaviest can share boat
        if people[i] + people[j] <= limit:
            i += 1  # Move lightest pointer right
        j -= 1  # Always move heaviest pointer left
        count += 1  # One more boat needed
    return count


def main():
    """
    Test function for rescue boats calculation.
    Tests various scenarios including:
    - Small groups
    - Mixed weights
    - Maximum weight groups
    - Sequential weights
    - Different limits
    """
    people = [
        [1, 2],  # Small group
        [3, 2, 2, 1],  # Mixed weights
        [3, 5, 3, 4],  # Larger weights
        [5, 5, 5, 5],  # All maximum weight
        [1, 2, 3, 4],  # Sequential weights
        [1, 2, 3],  # Three people
        [3, 4, 5],  # No small weights
    ]
    limit = [3, 3, 5, 5, 5, 3, 5]  # Different weight limits

    # Process each test case
    for i in range(len(people)):
        print(i + 1, "\tWeights = ", people[i], sep="")
        print("\tWeight Limit = ", limit[i], sep="")
        print(
            "\tThe minimum number of boats required to save people are ",
            rescue_boats(people[i], limit[i]),
            sep="",
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
