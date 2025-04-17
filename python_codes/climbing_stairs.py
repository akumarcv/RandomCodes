"""
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
"""

def climbStairs(n: int) -> int:
    """
    Calculate the number of distinct ways to climb n stairs, taking either 1 or 2 steps at a time.
    
    This is a dynamic programming solution that builds up the answer using the fact that:
    - To reach stair n, you can either:
        - Take 1 step from stair (n-1), or
        - Take 2 steps from stair (n-2)
    - Therefore, ways to reach n = ways to reach (n-1) + ways to reach (n-2)
    - This forms a Fibonacci-like sequence
    
    Args:
        n (int): The number of stairs to climb
        
    Returns:
        int: The number of distinct ways to climb to the top
        
    Time Complexity: O(n) - We need to calculate the result for each step from 3 to n
    Space Complexity: O(n) - We use an array of size n+1 to store intermediate results
    """
    # Handle the base case
    if n == 1:
        return 1
    
    # Initialize the DP array
    dp = [0] * (n + 1)
    
    # Base cases for DP
    dp[0] = 0  # This is not actually used
    dp[1] = 1  # There is 1 way to climb 1 stair
    dp[2] = 2  # There are 2 ways to climb 2 stairs: 1+1 or 2
    
    # Fill the DP array in bottom-up manner
    for i in range(3, n + 1):
        # Ways to reach stair i = ways to reach (i-1) + ways to reach (i-2)
        dp[i] = dp[i - 1] + dp[i - 2]
    
    # Return the final result
    return dp[n]


# Driver code to test the solution
if __name__ == "__main__":
    # Test cases
    test_cases = [1, 2, 3, 4, 5, 10]
    
    print("Testing climbStairs function:")
    print("-" * 30)
    
    for stairs in test_cases:
        ways = climbStairs(stairs)
        print(f"Number of ways to climb {stairs} stairs: {ways}")
    
    # Expected outputs:
    # 1 stairs: 1 way (1)
    # 2 stairs: 2 ways (1+1, 2)
    # 3 stairs: 3 ways (1+1+1, 1+2, 2+1)
    # 4 stairs: 5 ways (1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2)
    # 5 stairs: 8 ways
    # 10 stairs: 89 ways
    
    # Verify with user input
    try:
        user_input = int(input("\nEnter number of stairs to calculate ways: "))
        if user_input > 0:
            print(f"Number of ways to climb {user_input} stairs: {climbStairs(user_input)}")
        else:
            print("Please enter a positive integer")
    except ValueError:
        print("Invalid input. Please enter a positive integer.")