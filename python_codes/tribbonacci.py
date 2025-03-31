def trib_helper(n, table):
    """
    Helper function for calculating the nth Tribonacci number using memoization.
    Tribonacci sequence: each number is the sum of the three preceding ones.
    
    Args:
        n: The position in the Tribonacci sequence to calculate
        table: Memoization table to store previously calculated values
        
    Returns:
        int: The nth Tribonacci number
        
    Time Complexity: O(n) as each value is calculated exactly once
    Space Complexity: O(n) for the memoization table and recursion stack
    """
    # Return cached result if available
    if table[n] != -1:
        return table[n]

    # Calculate Tribonacci value using recursive formula
    new_val = (
        trib_helper(n - 1, table)  # First preceding number
        + trib_helper(n - 2, table)  # Second preceding number
        + trib_helper(n - 3, table)  # Third preceding number
    )

    # Store result in memoization table
    table[n] = new_val

    return table[n]


def find_tribonacci(n):
    """
    Calculate the nth Tribonacci number.
    The Tribonacci sequence starts with [0, 1, 1] and then each number is 
    the sum of the three preceding ones.
    
    Args:
        n: The position in the Tribonacci sequence (0-indexed)
        
    Returns:
        int: The nth Tribonacci number
        
    Time Complexity: O(n) using dynamic programming with memoization
    Space Complexity: O(n) for the memoization table
    
    Example:
        >>> find_tribonacci(4)
        4  # The sequence is [0,1,1,2,4,...]
    """
    # Handle base cases
    if n == 0:
        return 0  # First Tribonacci number
    if n == 1:
        return 1  # Second Tribonacci number
    if n == 2:
        return 1  # Third Tribonacci number

    # Initialize memoization table with -1 (uncomputed values)
    table = [-1] * (n + 1)
    # Set base cases in table
    table[0] = 0
    table[1] = 1
    table[2] = 1

    # Calculate nth Tribonacci using helper function
    val = trib_helper(n, table)

    return val


# Driver code
def main():
    """
    Test the Tribonacci function with multiple inputs.
    Displays the Tribonacci number for positions 0 through 15.
    
    The expected sequence begins:
    0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149, 274, 504, 927, 1705, 3136, ...
    
    Each test case demonstrates how the sequence grows based on the
    sum of the three preceding values.
    """
    n_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for i in n_values:
        print(i, ".\t Tribonacci value: ", find_tribonacci(i), sep="")
        print("-" * 100)


if __name__ == "__main__":
    main()