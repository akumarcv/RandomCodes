import heapq
def find_right_interval(intervals):
    result = [-1] * len(intervals)
    
    start_heap = []
    end_heap = []
    for i in range(len(intervals)):
        heapq.heappush(start_heap, (intervals[i][0], i))
        heapq.heappush(end_heap, (intervals[i][1], i))
    
    while end_heap:
        end, index = heapq.heappop(end_heap)
        while start_heap and start_heap[0][0]<end:
            heapq.heappop(start_heap)
        if start_heap:
            result[index] = start_heap[0][1]
        
    # Return this placeholder return statement with your code
    return result

def main():
    test_cases = [
        [[1, 2]],
        [[3, 4], [2, 3], [1, 2]],
        [[1, 4], [2, 3], [3, 4]],
        [[5, 6], [1, 2], [3, 4]],
        [[1, 3], [2, 4], [3, 5], [4, 6]],
    ]

    for i, test_case in enumerate(test_cases):
        print(i + 1, "\tintervals:", test_case)
        result = find_right_interval(test_case)
        print("\n\tOutput:", result)
        print("-" * 100)

if __name__ == "__main__":
    main()