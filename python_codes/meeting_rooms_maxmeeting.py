import heapq


def most_booked(meetings: list[list[int]], rooms: int) -> int:
    """
    Find room that held most meetings using multiple heap approach.
    Uses min heaps to track meeting schedules and room availability.
    
    Args:
        meetings: List of [start_time, end_time] for each meeting
        rooms: Number of available meeting rooms
        
    Returns:
        int: Index of room that held the most meetings (0-indexed)
        
    Time Complexity: O(M * log(M)) where M is number of meetings
    Space Complexity: O(R) where R is number of rooms
    
    Example:
        >>> most_booked([[0,10],[1,5],[2,7],[3,4]], 2)
        0  # Room 0 held more meetings than room 1
    """
    # Sort meetings by start time for processing in order
    meetings.sort(key=lambda x: x[0])

    # Initialize heap for tracking meeting start times and indices
    meetings_heap = []
    for i in range(len(meetings)):
        heapq.heappush(meetings_heap, (meetings[i][0], i))

    # Track number of meetings held in each room
    meetings_in_rooms = [0] * rooms

    # Initialize heaps for each room to track meeting end times
    heaps = [[] for _ in range(rooms)]

    # Current time tracker
    r = 0
    while meetings_heap:
        # Check if any meetings have ended in any room
        for i in range(rooms):
            if heaps[i] and ((-heaps[i][0]) == r):
                heapq.heappop(heaps[i])

        # Get next meeting to schedule
        _, idx = heapq.heappop(meetings_heap)
        assigned = False
        
        # Try to assign meeting to an available room
        for i in range(rooms):
            # Room is empty or previous meeting has ended
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
                
        # If no room available, delay meeting by 1 unit
        if not assigned:
            meetings[idx][0] += 1
            meetings[idx][1] += 1
            heapq.heappush(meetings_heap, (meetings[idx][0], idx))

        r += 1

    # Return index of room that held most meetings
    return meetings_in_rooms.index(max(meetings_in_rooms))


def main():
    """
    Driver code to test meeting room allocation with various scenarios.
    Tests different configurations including:
    - Sequential meetings
    - Overlapping meetings
    - Different room counts
    - Various meeting durations
    """
    meetings = [
        [[0, 10], [1, 11], [2, 12], [3, 13], [4, 14], [5, 15]],  # Sequential starts
        [[1, 20], [2, 10], [3, 5], [4, 9], [6, 8]],              # Various durations
        [[1, 2], [0, 10], [2, 3], [3, 4]],                       # Mixed schedule
        [[0, 2], [1, 2], [3, 4], [2, 4]],                        # Short meetings
        [[1, 9], [2, 8], [3, 7], [4, 6], [5, 11]],              # Nested meetings
        [[0, 4], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],       # Continuous schedule
    ]
    rooms = [3, 3, 2, 4, 3, 4]  # Different room configurations

    # Process each test case
    for i in range(len(meetings)):
        print(f"{i+1}.\tMeetings: {meetings[i]}")
        print(f"\tRooms: {rooms[i]}")
        booked_rooms = most_booked(meetings[i], rooms[i])
        print(f"\n\tRoom that held the most meetings: {booked_rooms}")
        print("-" * 100)


if __name__ == "__main__":
    main()