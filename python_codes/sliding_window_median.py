from heapq import heappop, heappush, heapify


def median_sliding_window(nums, k):
    """
    Find median of sliding windows of size k moving through an array.
    Uses two heaps approach to efficiently track window medians.

    Args:
        nums: List of integers to process
        k: Size of sliding window

    Returns:
        list: Median values for each window position

    Time Complexity: O(n*log k) where n is array length
    Space Complexity: O(n) for storing results and heaps

    Example:
        >>> median_sliding_window([1,3,-1,-3,5,3,6,7], 3)
        [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]  # Medians of each window
    """
    result = []  # Store median results
    min_heap = []  # Store larger half of elements in min heap
    max_heap = []  # Store smaller half of elements in max heap (negated)
    outgoing_num = {}  # Track numbers leaving window for lazy deletion

    # Initialize first window by adding all elements to max heap
    for i in range(k):
        heappush(max_heap, -nums[i])

    # Rebalance by moving half elements to min heap
    for i in range(0, k // 2):
        element = -heappop(max_heap)
        heappush(min_heap, element)

    balance = (
        0  # Track heap balance (+1 if max_heap has extra, -1 if min_heap has extra)
    )
    i = k  # Current window end position

    while True:
        # Calculate median based on heap tops
        if k % 2 == 0:
            # For even k, median is average of tops of both heaps
            result.append((-max_heap[0] + min_heap[0]) / 2)
        else:
            # For odd k, median is top of max heap
            result.append(-max_heap[0])

        # Exit loop when all windows have been processed
        if i == len(nums):
            break

        # Identify outgoing and incoming numbers for sliding window
        number_going_out = nums[i - k]  # Element leaving the window
        number_coming_in = nums[i]  # Element entering the window

        # Increment window end position
        i += 1

        # Update balance based on where outgoing number was located
        if number_going_out <= -max_heap[0]:
            balance = balance - 1  # Outgoing number was in max heap
        else:
            balance = balance + 1  # Outgoing number was in min heap

        # Track outgoing number in dictionary for lazy deletion
        if number_going_out in outgoing_num:
            outgoing_num[number_going_out] += 1  # Increment count if already tracked
        else:
            outgoing_num[number_going_out] = 1  # Initialize count if first occurrence

        # Add incoming number to appropriate heap
        if not max_heap or number_coming_in <= -max_heap[0]:
            heappush(max_heap, -number_coming_in)  # Add to max heap if smaller
            balance = balance + 1  # Max heap grew
        else:
            heappush(min_heap, number_coming_in)  # Add to min heap if larger
            balance = balance - 1  # Min heap grew

        # Rebalance heaps if needed
        if balance > 0:
            # Move element from max heap to min heap
            heappush(min_heap, -heappop(max_heap))
            balance -= 1
        if balance < 0:
            # Move element from min heap to max heap
            heappush(max_heap, -heappop(min_heap))
            balance += 1
        balance = 0  # Reset balance after rebalancing

        # Lazy deletion: Remove outgoing numbers that are at heap tops
        # Clean max heap first
        while -max_heap[0] in outgoing_num and outgoing_num[-max_heap[0]] > 0:
            outgoing_num[-heappop(max_heap)] -= 1  # Remove and decrement count

        # Clean min heap next
        while (
            min_heap and min_heap[0] in outgoing_num and outgoing_num[min_heap[0]] > 0
        ):
            outgoing_num[heappop(min_heap)] -= 1  # Remove and decrement count

    return result


def main():
    """
    Driver code to test median_sliding_window function.
    Tests various input arrays with different window sizes.
    Each test case includes:
    - Input array
    - Window size k
    - Resulting medians (not explicitly shown)

    Test cases cover:
    - Standard arrays with mixed values
    - Small arrays and windows
    - Large arrays with larger windows
    - Arrays with duplicate values
    """
    input = (
        ([3, 1, 2, -1, 0, 5, 8], 4),  # Mixed values
        ([1, 2], 1),  # Small array, k=1
        ([4, 7, 2, 21], 2),  # Small array, k=2
        ([22, 23, 24, 56, 76, 43, 121, 1, 2, 0, 0, 2, 3, 5], 5),  # Large array
        ([1, 1, 1, 1, 1], 2),  # Duplicate values
    )
    x = 1
    for i in input:
        print(x, ".\tInput array: ", i[0], ", k = ", i[1], sep="")
        print("\tMedians: ", median_sliding_window(i[0], i[1]), sep="")
        print(100 * "-", "\n", sep="")
        x += 1


if __name__ == "__main__":
    main()
