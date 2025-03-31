def find_content_children(greed_factors: list[int], cookie_sizes: list[int]) -> int:
    """
    Find maximum number of children that can be satisfied with given cookies.
    Uses greedy approach by assigning smallest sufficient cookie to each child.
    
    Args:
        greed_factors: List of greed factors for each child
        cookie_sizes: List of available cookie sizes
        
    Returns:
        int: Number of children that can be satisfied
        
    Time Complexity: O(nlogn) where n is max length of input lists (for sorting)
    Space Complexity: O(1) as sorting is done in-place
    
    Example:
        >>> find_content_children([1,2,3], [1,1])
        1  # Only one child (with greed factor 1) can be satisfied
    """
    # Sort both arrays to use greedy approach
    greed_factors.sort()  # Sort children by greed factor
    cookie_sizes.sort()   # Sort cookies by size
    
    greed, cookie = 0, 0  # Pointers for current child and cookie
    count = 0            # Count of satisfied children
    
    # Try to satisfy each child with smallest sufficient cookie
    while greed < len(greed_factors) and cookie < len(cookie_sizes):
        if greed_factors[greed] <= cookie_sizes[cookie]:
            # Current cookie can satisfy current child
            count += 1
            greed += 1   # Move to next child
            cookie += 1  # Use current cookie
        else:
            # Current cookie too small, try next cookie
            cookie += 1
            
    return count

def main():
    """
    Driver code to test cookie assignment with various test cases.
    Tests different combinations of greed factors and cookie sizes.
    """
    # Test cases pairing greed factors with cookie sizes
    greed_factors = [
        [1, 2, 3],                              # Basic case
        [10, 20, 30, 40, 50, 60, 70, 80],      # Large values
        [3, 4, 5, 6, 7, 8],                    # Sequential values
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],  # Same values
        [10, 9, 8, 7],                         # Descending values
        [1000, 996, 867, 345, 23, 12],         # Large range
    ]

    cookie_sizes = [
        [1, 1],                                # Limited cookies
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # More cookies than children
        [1, 2],                                # Insufficient cookies
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], # Matching sizes
        [5, 6, 7, 8],                          # All smaller than greed
        [],                                     # Empty cookie list
    ]

    # Process each test case
    for i in range(len(greed_factors)):
        result = find_content_children(greed_factors[i], cookie_sizes[i])
        print(f"{i + 1}.\tGreed factors: {greed_factors[i]}")
        print(f"\tCookie sizes: {cookie_sizes[i]}")
        print(f"\n\tSatisfied children: {result}")
        print("-" * 100)

if __name__ == "__main__":
    main()