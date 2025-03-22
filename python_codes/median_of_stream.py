from heapq import *


class MedianOfStream:
    def __init__(self):
        self.min_heap = []
        self.max_heap = []

    # This function should take a number and store it
    def insert_num(self, num):
        # Write your code here
        if not self.max_heap or (-self.max_heap[0] >= num):
            heappush(self.max_heap, -num)
        else:
            heappush(self.min_heap, num)

        if len(self.max_heap) > len(self.min_heap) + 1:
            heappush(self.min_heap, -heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heappush(self.max_heap, -heappop(self.min_heap))

    # This function should return the median of the stored numbers
    def find_median(self):
        # Replace this placeholder return statement with your code
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        return -self.max_heap[0]


def main():
    median_num = MedianOfStream()
    nums = [35, 22, 30, 25, 1]
    numlist = []
    x = 1
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
