def combination_sum(nums: list[int], target: int) -> list[list[int]]:
    """
    Find all unique combinations where numbers from nums sum to target.
    Numbers can be used multiple times.
    
    Uses dynamic programming approach where:
    dp[i] = list of all combinations that sum to i
    
    Args:
        nums: List of candidate numbers to use
        target: Target sum to achieve
        
    Returns:
        List of lists, each inner list is a valid combination
        
    Time Complexity: O(target * len(nums) * x) where x is avg combination length
    Space Complexity: O(target * number of combinations)
    
    Example:
        >>> combination_sum([2,3,5], 8)
        [[2,2,2,2], [2,3,3], [3,5]]
    """
    # Initialize dp array where dp[i] stores combinations summing to i
    dp = [[] for _ in range(target + 1)]
    dp[0].append([])  # Empty combination sums to 0
    
    # Build combinations for each sum from 1 to target
    for current_sum in range(1, target + 1):
        for num in nums:
            # If current number can be used
            if num <= current_sum:
                # Get combinations for remaining sum after using current number
                for prev_combination in dp[current_sum - num]:
                    # Add current number to previous combination
                    new_combination = prev_combination + [num]
                    new_combination.sort()  # Sort to avoid duplicates
                    # Add if unique
                    if new_combination not in dp[current_sum]:
                        dp[current_sum].append(new_combination)
    
    return dp[target] if dp[target] else []


def main():
    """
    Driver code to test combination sum functionality with various inputs.
    Tests different combinations of numbers and target sums.
    """
    # Test cases: pairs of numbers and target sums
    test_cases = [
        ([2, 3, 5], 5),              # Small numbers, small target
        ([3, 6, 7, 8], 15),          # Medium numbers, medium target
        ([4, 5, 6, 9], 11),          # Mixed numbers
        ([20, 25, 30, 35, 40], 40),  # Large numbers
        ([3, 5, 7], 15)              # Few numbers, larger target
    ]
    
    for i, (nums, target) in enumerate(test_cases, 1):
        print(f"{i}.\tNumbers: {nums}")
        print(f"\tTarget Sum: {target}")
        combinations = combination_sum(nums, target)
        print(f"\tValid Combinations: {combinations}")
        print(f"\tNumber of Combinations: {len(combinations)}")
        print("-" * 100, "\n")


if __name__ == "__main__":
    main()