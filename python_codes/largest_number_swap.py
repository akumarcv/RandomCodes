import heapq
from typing import List


def largest_integer(num: int) -> int:
    """
    Find largest integer by swapping even digits with even and odd with odd.
    Uses two max heaps to separately track even and odd digits.

    Args:
        num: Input integer to process

    Returns:
        int: Largest possible number after swapping digits

    Time Complexity: O(n log n) where n is number of digits
    Space Complexity: O(n) for storing digits in heaps

    Example:
        >>> largest_integer(1234)
        3412  # Swapped 1->3 and 2->4 to get largest number
    """
    # Convert number to list of digits
    digit_list = [int(d) for d in str(num)]
    odd_max_heap = []  # Max heap for odd digits
    even_max_heap = []  # Max heap for even digits
    result = []  # Store final digit arrangement

    # Separate digits into even and odd heaps
    # Use negative values for max heap behavior
    for i in digit_list:
        if i % 2 == 0:
            heapq.heappush(even_max_heap, -i)  # Even digits
        else:
            heapq.heappush(odd_max_heap, -i)  # Odd digits

    # Reconstruct number using largest available digits
    # while maintaining even/odd positions
    for i in digit_list:
        if i % 2 == 0:
            result.append(-heapq.heappop(even_max_heap))  # Get largest even
        else:
            result.append(-heapq.heappop(odd_max_heap))  # Get largest odd

    # Convert result back to integer
    return int("".join(map(str, result)))


def main():
    """
    Driver code to test largest integer functionality.
    Tests various scenarios including:
    - Mixed even/odd digits
    - All even digits
    - All odd digits
    - Different digit counts
    - Edge cases
    """
    test_cases = [
        1234,  # Basic mixed case
        65875,  # More odd digits
        4321,  # Descending order
        2468,  # All even digits
        98123,  # Random arrangement
    ]

    for num in test_cases:
        print(f"\nTest Case:")
        print(f"\tInput number: {num}")
        result = largest_integer(num)
        print(f"\tLargest possible number: {result}")

        # Show which digits were swapped
        original = list(str(num))
        final = list(str(result))
        swaps = [(i, o, f) for i, (o, f) in enumerate(zip(original, final)) if o != f]
        if swaps:
            print("\tSwaps made:")
            for pos, orig, new in swaps:
                print(f"\t\tPosition {pos}: {orig} → {new}")
        print("-" * 100)


if __name__ == "__main__":
    main()
