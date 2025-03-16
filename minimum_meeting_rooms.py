import heapq


def find_sets(intervals):
    
    intervals.sort(key= lambda x: x[0])
    
    min_heap = []
    heapq.heappush(min_heap, intervals[0][1])
    
    for i in range(1, len(intervals)):
        if intervals[i][0]>=min_heap[0]:
            heapq.heappop(min_heap)
        heapq.heappush(min_heap, intervals[i][1])
    return len(min_heap)

def main():
    inputs =[[2,8],[3,4],[3,9],[5,11],[8,20],[11,15]]

    print("Input: ", inputs)
    print("Minimum Meeting Rooms: ", find_sets(inputs))


if __name__ == "__main__":
    main()