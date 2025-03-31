from heapq import *


class MedianOfStream:
    """
    Class to find running median of a stream of numbers using two heaps.
    Uses min heap for larger half and max heap for smaller half of numbers.
    """

    def __init__(self):
        """
        Initialize empty heaps for storing numbers.
        min_heap stores larger half of numbers
        max_heap stores smaller half (as negatives for max heap behavior)
        """
        self.min_heap = []  # Store larger numbers
        self.max_heap = []  # Store smaller numbers (negated)

    def insert_num(self, num):
        """
        Insert a new number into the stream and maintain heap balance.

        Args:
            num: Number to insert into stream

        Time Complexity: O(log n) for heap operations
        Space Complexity: O(n) for storing all numbers
        """
        # Insert into appropriate heap based on value
        if not self.max_heap or (-self.max_heap[0] >= num):
            heappush(self.max_heap, -num)  # Add to smaller half
        else:
            heappush(self.min_heap, num)  # Add to larger half

        # Rebalance heaps if needed
        # Keep max_heap size equal to or one more than min_heap
        if len(self.max_heap) > len(self.min_heap) + 1:
            heappush(self.min_heap, -heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heappush(self.max_heap, -heappop(self.min_heap))

    def find_median(self):
        """
        Calculate current median from the two heaps.

        Returns:
            float: Current median of all numbers
                  If even count: average of middle two numbers
                  If odd count: middle number

        Time Complexity: O(1) as we maintain sorted halves
        """
        if len(self.max_heap) == len(self.min_heap):
            # Even number of elements, take average of middle two
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        # Odd number of elements, median is in max_heap
        return -self.max_heap[0]


def main():
    """
    Driver code to test MedianOfStream functionality.
    Demonstrates:
    - Stream processing
    - Running median calculation
    - Different stream sizes
    - Even/odd number of elements
    """
    median_num = MedianOfStream()
    nums = [35, 22, 30, 25, 1]  # Test stream of numbers
    numlist = []  # Track numbers for display
    x = 1  # Counter for output formatting

    # Process each number in stream
    for i in nums:
        numlist.append(i)
        print(x, ".\tData stream: ", numlist, sep="")
        median_num.insert_num(i)
        print(
            "\tThe median for the given numbers is: " + str(median_num.find_median()),
            sep="",
        )
        print(100 * "-" + "\n")
        x += 1


if __name__ == "__main__":
    main()
