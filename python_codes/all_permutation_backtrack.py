from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        Generate all possible permutations of the given list of numbers.

        Args:
            nums: A list of integers

        Returns:
            A list containing all possible permutations of the input list
        """
        result = []

        def backtrack(current_perm, remaining):
            # Base case: if no numbers remain, we've found a complete permutation
            if not remaining:
                result.append(current_perm[:])  # Add a copy of the current permutation
                return

            # Try placing each of the remaining numbers at the current position
            for i in range(len(remaining)):
                # Add the current number to our permutation
                current_perm.append(remaining[i])

                # Recurse with the remaining numbers (excluding the one we just used)
                backtrack(current_perm, remaining[:i] + remaining[i + 1 :])

                # Backtrack by removing the number we just tried
                current_perm.pop()

        # Start the backtracking with an empty permutation and all numbers available
        backtrack([], nums)
        return result


# Driver code to test the solution
if __name__ == "__main__":
    # Test cases
    test_cases = [[1, 2, 3], [0, 1], [1]]

    solution = Solution()

    for tc in test_cases:
        print(f"\nInput: {tc}")
        result = solution.permute(tc)
        print(f"Output: {result}")
        print(f"Number of permutations: {len(result)}")
