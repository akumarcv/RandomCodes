import heapq


def min_refuel_stops(target, start_fuel, stations):
    """
    Find minimum number of refueling stops to reach target distance.
    Uses greedy approach with max heap to select optimal refueling stations.
    
    Args:
        target: Target distance to reach
        start_fuel: Initial fuel amount in the vehicle
        stations: List of [position, fuel] pairs representing refueling stations
        
    Returns:
        int: Minimum number of stops needed to reach target
        0: If target can be reached without refueling
        -1: If target cannot be reached regardless of stops
        
    Time Complexity: O(n log n) where n is number of stations
    Space Complexity: O(n) for the max heap storage
    
    Example:
        >>> min_refuel_stops(15, 3, [[2, 5], [3, 1], [6, 3], [12, 6]])
        2  # Need to refuel at stations with position 2 and 6
    """
    # Store current remaining fuel
    current_fuel_capacity = start_fuel

    # If we already have enough fuel to reach target, no stops needed
    if current_fuel_capacity >= target:
        return 0

    i = 0                   # Current station index
    stops = 0               # Number of refueling stops made
    max_heap = []           # Max heap to track stations with most fuel
    
    # Continue until we reach the target or cannot go further
    while current_fuel_capacity < target:
        # Add all reachable stations to the heap
        while i < len(stations) and stations[i][0] <= current_fuel_capacity:
            # Use negative fuel value for max-heap behavior
            heapq.heappush(max_heap, -stations[i][1])
            i += 1
            
        # If no stations available and target not reached, impossible
        if not max_heap:
            return -1
            
        # Take fuel from station with maximum fuel
        fuel = heapq.heappop(max_heap)
        current_fuel_capacity = current_fuel_capacity - fuel  # Adding fuel (negative of negative)
        stops += 1  # Count this refueling stop

    return stops


def main():
    """
    Driver code to test minimum refueling stops calculation.
    Tests various scenarios including:
    - No stations needed
    - Impossible to reach target
    - Multiple optimal paths
    - Long distance with strategic refueling
    """
    input = (
        (3, 3, []),                                                    # Can reach without stops
        (59, 14, [[9, 12], [11, 7], [13, 16], [21, 18], [47, 6]]),     # Multiple stations
        (15, 3, [[2, 5], [3, 1], [6, 3], [12, 6]]),                    # Short journey
        (570, 140, [[140, 200], [160, 130], [310, 200], [330, 250]]),  # Medium journey
        (1360, 380, [[310, 160], [380, 620], [700, 89], [850, 190], [990, 360]]),  # Long journey
    )
    num = 1
    for i in input:
        print(num, ".\tStations : ", i[2], sep="")
        print("\tTarget : ", i[0])
        print("\tStarting fuel : ", i[1])
        print(
            "\n\tMinimum number of refueling stops :",
            min_refuel_stops(i[0], i[1], i[2]),
        )
        num += 1
        print("-" * 100, "\n")


if __name__ == "__main__":
    main()