"""
You have k bags. You are given a 0-indexed integer array weights where weights[i] is the weight of the ith marble. You are also given the integer k.

Divide the marbles into the k bags according to the following rules:

No bag is empty.
If the ith marble and jth marble are in a bag, then all marbles with an index between the ith and jth indices should also be in that same bag.
If a bag consists of all the marbles with an index from i to j inclusively, then the cost of the bag is weights[i] + weights[j].
The score after distributing the marbles is the sum of the costs of all the k bags.

Return the difference between the maximum and minimum scores among marble distributions.
"""
from typing import List

class Solution:
    def putMarbles(self, weights: List[int], k: int) -> int:
        """
        Calculate the difference between maximum and minimum scores of distributing marbles into k bags.
        
        The problem can be solved by realizing that when we split the array into k subarrays,
        we are essentially selecting k-1 splitting points. Each splitting point contributes
        the sum of the weights at both ends of the split to the total score.
        
        Time Complexity: O(n log n) due to sorting the pair weights
        Space Complexity: O(n) for storing the pair weights
        
        Args:
            weights (List[int]): Weights of the marbles
            k (int): Number of bags to distribute marbles into
            
        Returns:
            int: Difference between maximum and minimum possible scores
        """
        # Edge case: if k is 1, there's only one way to distribute - all marbles in one bag
        if k == 1:
            return 0
            
        # For any distribution into k bags, we need to select k-1 split points
        # When we split at position i, we add weights[i] + weights[i+1] to the total score
        n = len(weights)
        
        # Calculate the cost of each potential split point (adjacency pair weights)
        pairWeights = [weights[i] + weights[i + 1] for i in range(n - 1)]

        # Sort to easily find the k-1 split points that give min/max scores
        pairWeights.sort()

        # Maximum score: sum of the largest k-1 pair weights
        maxScore = sum(pairWeights[-(k - 1):])
        # Minimum score: sum of the smallest k-1 pair weights
        minScore = sum(pairWeights[:k - 1])
        # The difference between max and min scores
        return maxScore - minScore


# Driver code to test the solution
if __name__ == "__main__":
    sol = Solution()
    
    # Test Case 1
    weights1 = [1, 3, 5, 1]
    k1 = 2
    print(f"Test Case 1: weights = {weights1}, k = {k1}")
    print(f"Result: {sol.putMarbles(weights1, k1)}")
    print(f"Expected: 4")
    # Explanation:
    # - Maximum score: Split [1,3,5,1] into [1,3,5] and [1] with score (1+5)+(1+1)=8
    # - Minimum score: Split [1,3,5,1] into [1] and [3,5,1] with score (1+1)+(3+1)=4
    # - Difference: 8-4=4
    
    # Test Case 2
    weights2 = [1, 3]
    k2 = 2
    print(f"\nTest Case 2: weights = {weights2}, k = {k2}")
    print(f"Result: {sol.putMarbles(weights2, k2)}")
    print(f"Expected: 0")
    # Explanation: 
    # - Only one way to split into 2 bags: [1] and [3], resulting in score (1+1)+(3+3)=8
    # - Since there's only one way, max score = min score, so difference is 0
    
    # Test Case 3
    weights3 = [1, 4, 2, 5, 2]
    k3 = 3
    print(f"\nTest Case 3: weights = {weights3}, k = {k3}")
    print(f"Result: {sol.putMarbles(weights3, k3)}")
    print(f"Expected: 3")
    # This case demonstrates a more complex scenario with 3 bags