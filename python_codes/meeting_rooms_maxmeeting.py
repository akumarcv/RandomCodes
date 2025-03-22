import heapq


def most_booked(meetings, rooms):
    meetings.sort(key=lambda x: x[0])

    meetings_heap = []
    for i in range(len(meetings)):
        heapq.heappush(meetings_heap, (meetings[i][0], i))

    meetings_in_rooms = [0] * rooms

    heaps = [[] for _ in range(rooms)]

    r = 0
    while meetings_heap:
        for i in range(rooms):
            if heaps[i] and ((-heaps[i][0]) == r):
                heapq.heappop(heaps[i])

        _, idx = heapq.heappop(meetings_heap)
        assigned = False
        for i in range(rooms):
            if not heaps[i]:
                heapq.heappush(heaps[i], -meetings[idx][1])
                meetings_in_rooms[i] += 1
                assigned = True
                break
            elif (-heaps[i][0]) <= meetings[idx][0]:
                heapq.heappush(heaps[i], -meetings[idx][1])
                meetings_in_rooms[i] += 1
                assigned = True
                break
        if not assigned:
            meetings[idx][0] += 1
            meetings[idx][1] += 1
            heapq.heappush(meetings_heap, (meetings[idx][0], idx))

        r += 1

    return meetings_in_rooms.index(max(meetings_in_rooms))


def main():
    meetings = [
        [[0, 10], [1, 11], [2, 12], [3, 13], [4, 14], [5, 15]],
        [[1, 20], [2, 10], [3, 5], [4, 9], [6, 8]],
        [[1, 2], [0, 10], [2, 3], [3, 4]],
        [[0, 2], [1, 2], [3, 4], [2, 4]],
        [[1, 9], [2, 8], [3, 7], [4, 6], [5, 11]],
        [[0, 4], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
    ]
    rooms = [3, 3, 2, 4, 3, 4]

    for i in range(len(meetings)):
        print(i + 1, ".", "\tMeetings: ", meetings[i], sep="")
        print("\tRooms: ", rooms[i], sep="")
        booked_rooms = most_booked(meetings[i], rooms[i])
        print("\n\tRoom that held the most meetings: ", booked_rooms)
        print("-" * 100)


if __name__ == "__main__":
    main()
