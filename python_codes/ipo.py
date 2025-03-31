from heapq import *
from typing import List


def maximum_capital(c: int, k: int, capitals: List[int], profits: List[int]) -> int:
    """
    Find maximum capital after selecting k projects using two heaps approach.
    
    Uses min heap for capital requirements and max heap for profits to:
    1. Track available projects within current capital
    2. Select most profitable project at each step
    
    Args:
        c: Initial capital available
        k: Number of projects to select
        capitals: List of capital requirements for each project
        profits: List of expected profits for each project
        
    Returns:
        int: Maximum capital achievable after selecting k projects
        
    Time Complexity: O(NlogN + KlogN) where N is number of projects
    Space Complexity: O(N) for storing heaps
    
    Example:
        >>> maximum_capital(0, 2, [1,1,2], [1,2,3])
        4  # Select project 1 (capital=1, profit=2), then project 2 (capital=2, profit=3)
    """
    current_capital = c
    capital_min_heap = []  # Min heap for tracking available projects
    profit_max_heap = []   # Max heap for selecting most profitable project

    # Add all projects to capital min heap with their indices
    for i in range(len(capitals)):
        heappush(capital_min_heap, (capitals[i], i))

    # Select k projects
    for j in range(k):
        # Add all projects that can be funded with current capital to profit heap
        while capital_min_heap and capital_min_heap[0][0] <= current_capital:
            capital, i = heappop(capital_min_heap)
            heappush(profit_max_heap, -profits[i])  # Negative for max heap

        # If no projects available within current capital, break
        if not profit_max_heap:
            break
            
        # Select most profitable project and update capital
        current_capital += -heappop(profit_max_heap)

    return current_capital


def main():
    """
    Driver code to test maximum capital calculation with various inputs.
    Tests different scenarios including:
    - Zero initial capital
    - Different project counts
    - Various profit/capital combinations
    - Edge cases
    """
    # Test cases: (initial_capital, projects_to_select, capital_requirements, profits)
    input = (
        (0, 1, [1, 1, 2], [1, 2, 3]),              # Zero initial capital
        (1, 2, [1, 2, 2, 3], [2, 4, 6, 8]),        # Multiple projects
        (2, 3, [1, 3, 4, 5, 6], [1, 2, 3, 4, 5]),  # Increasing capitals
        (1, 3, [1, 2, 3, 4], [1, 3, 5, 7]),        # Linear growth
        (7, 2, [6, 7, 8, 10], [4, 8, 12, 14]),     # High initial capital
        (2, 4, [2, 3, 5, 6, 8, 12], [1, 2, 5, 6, 8, 9]),  # Many projects
    )

    # Process each test case
    # ...existing code...


if __name__ == "__main__":
    main()