def gas_station_journey(gas: list[int], cost: list[int]) -> int:
    """
    Find starting gas station index for completing circular journey.
    Uses greedy approach to find valid starting point.
    
    Args:
        gas: List of integers where gas[i] is amount of gas at station i
        cost: List of integers where cost[i] is gas needed to reach station i+1
        
    Returns:
        int: Valid starting station index, or -1 if no solution exists
        
    Time Complexity: O(n) where n is number of stations
    Space Complexity: O(1) as we only use constant extra space
    
    Example:
        >>> gas_station_journey([1,2,3,4,5], [3,4,5,1,2])
        3  # Starting from station 3 allows complete circuit
    """
    # First check if journey is possible at all
    if sum(gas) < sum(cost):
        return -1
    
    current_gas = 0  # Track remaining gas
    start = 0        # Potential starting station
    
    # Try each station as starting point
    for i in range(len(gas)):
        # Update gas after visiting current station
        current_gas = gas[i] - cost[i] + current_gas
        
        # If we run out of gas
        if current_gas < 0:
            current_gas = 0  # Reset gas
            start = i + 1    # Try next station as start
            
    return start


def main():
    """
    Driver code to test gas station journey with various inputs.
    Tests different configurations including:
    - Regular cases with solution
    - Cases with no solution
    - Equal gas and cost
    - Different array sizes
    - Edge cases
    """
    # Test cases: pairs of gas amounts and costs
    gas = [
        [1, 2, 3, 4, 5],     # Regular case with solution
        [2, 3, 4],           # Smaller array
        [1, 1, 1, 1, 1],     # Equal values
        [1, 1, 1, 1, 10],    # Large last station
        [1, 1, 1, 1, 1],     # Balanced case
        [1, 2, 3, 4, 5],     # No solution case
    ]
    
    cost = [
        [3, 4, 5, 1, 2],     # Solution exists
        [3, 4, 3],           # Small array
        [1, 2, 3, 4, 5],     # Increasing costs
        [2, 2, 1, 3, 1],     # Variable costs
        [1, 0, 1, 2, 3],     # Zero cost included
        [1, 2, 3, 4, 5],     # Equal to gas
    ]
    
    for i in range(len(gas)):
        print(f"{i + 1}.\tGas stations = {gas[i]}")
        print(f"\tTravel costs = {cost[i]}")
        result = gas_station_journey(gas[i], cost[i])
        
        if result != -1:
            print(f"\tStart journey from station {result}")
            # Verify solution by simulating journey
            total_gas = 0
            for j in range(len(gas[i])):
                station = (result + j) % len(gas[i])
                total_gas += gas[station] - cost[station]
                if total_gas < 0:
                    print("\tWarning: Solution verification failed!")
                    break
        else:
            print("\tNo valid starting station exists")
        print("-" * 100)


if __name__ == "__main__":
    main()