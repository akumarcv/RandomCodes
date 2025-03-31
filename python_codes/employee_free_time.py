class Interval:
    """
    Class representing a time interval with start and end times.
    Can be either closed [start, end] or open (start, end).
    
    Attributes:
        start: Start time of interval
        end: End time of interval
        closed: Boolean indicating if interval is closed (True) or open (False)
    """
    def __init__(self, start: int, end: int):
        """Initialize interval with start and end times."""
        self.start = start
        self.end = end
        self.closed = True  # by default, the interval is closed

    def set_closed(self, closed: bool) -> None:
        """Set whether interval is closed or open."""
        self.closed = closed

    def __str__(self) -> str:
        """Return string representation of interval."""
        return (
            f"[{self.start}, {self.end}]" if self.closed
            else f"({self.start}, {self.end})"
        )


def employee_free_time(schedule: list[list[Interval]]) -> list[Interval]:
    """
    Find free time intervals common to all employees.
    
    Uses merge interval approach:
    1. Flatten all schedules into single list
    2. Sort intervals by start time
    3. Merge overlapping intervals
    4. Find gaps between merged intervals
    
    Args:
        schedule: List of employee schedules, where each schedule is
                 a list of Interval objects
        
    Returns:
        List of Interval objects representing free time periods
        
    Time Complexity: O(N log N) where N is total number of intervals
    Space Complexity: O(N) for storing merged intervals
    
    Example:
        >>> schedule = [[Interval(1,3)], [Interval(2,4)]]
        >>> free_time = employee_free_time(schedule)
        >>> print(free_time)  # No free time as intervals overlap
        []
    """
    # Flatten the schedule and sort by start time
    intervals = [i for s in schedule for i in s]
    intervals.sort(key=lambda x: x.start)

    # Merge overlapping intervals
    merged = []
    for interval in intervals:
        if not merged or merged[-1].end < interval.start:
            merged.append(interval)
        else:
            merged[-1].end = max(merged[-1].end, interval.end)

    # Find gaps between merged intervals (free time)
    free_time = []
    for i in range(1, len(merged)):
        if merged[i].start > merged[i-1].end:
            free_time.append(Interval(merged[i-1].end, merged[i].start))

    return free_time


def display(vec: list[Interval]) -> str:
    """
    Create string representation of interval list.
    
    Args:
        vec: List of Interval objects to display
        
    Returns:
        String representation of intervals in format [int1, int2, ...]
    """
    return f"[{', '.join(str(interval) for interval in vec)}]"



def main():
    inputs = [
        [[Interval(1, 2), Interval(5, 6)], [Interval(1, 3)], [Interval(4, 10)]],
        [
            [Interval(1, 3), Interval(6, 7)],
            [Interval(2, 4)],
            [Interval(2, 5), Interval(9, 12)],
        ],
        [[Interval(2, 3), Interval(7, 9)], [Interval(1, 4), Interval(6, 7)]],
        [
            [Interval(3, 5), Interval(8, 10)],
            [Interval(4, 6), Interval(9, 12)],
            [Interval(5, 6), Interval(8, 10)],
        ],
        [
            [Interval(1, 3), Interval(6, 9), Interval(10, 11)],
            [Interval(3, 4), Interval(7, 12)],
            [Interval(1, 3), Interval(7, 10)],
            [Interval(1, 4)],
            [Interval(7, 10), Interval(11, 12)],
        ],
        [
            [Interval(1, 2), Interval(3, 4), Interval(5, 6), Interval(7, 8)],
            [Interval(2, 3), Interval(4, 5), Interval(6, 8)],
        ],
        [
            [
                Interval(1, 2),
                Interval(3, 4),
                Interval(5, 6),
                Interval(7, 8),
                Interval(9, 10),
                Interval(11, 12),
            ],
            [
                Interval(1, 2),
                Interval(3, 4),
                Interval(5, 6),
                Interval(7, 8),
                Interval(9, 10),
                Interval(11, 12),
            ],
            [
                Interval(1, 2),
                Interval(3, 4),
                Interval(5, 6),
                Interval(7, 8),
                Interval(9, 10),
                Interval(11, 12),
            ],
            [
                Interval(1, 2),
                Interval(3, 4),
                Interval(5, 6),
                Interval(7, 8),
                Interval(9, 10),
                Interval(11, 12),
            ],
        ],
    ]
    i = 1
    for schedule in inputs:
        print(i, ".\tEmployee Schedules:", sep="")
        for s in schedule:
            print("\t\t", display(s), sep="")
        print("\tEmployees' free time", display(employee_free_time(schedule)))
        print("-" * 100)
        i += 1


if __name__ == "__main__":
    main()
