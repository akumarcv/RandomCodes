from typing import List


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        """
        Find the longest common prefix string amongst an array of strings.
        Uses character-by-character comparison approach.

        Args:
            strs: List of strings to find common prefix from

        Returns:
            str: Longest common prefix string, or empty string if none exists

        Time Complexity: O(S) where S is sum of all characters in all strings
        Space Complexity: O(1) as we only store the prefix string

        Example:
            >>> Solution().longestCommonPrefix(["flower", "flow", "flight"])
            "fl"  # "fl" is the longest common prefix
        """
        # Handle empty input case
        if len(strs) == 0:
            return ""

        prefix = ""  # Store the common prefix

        # Find length of shortest string (maximum possible prefix length)
        min_len = min([len(s) for s in strs])

        # Check each character position across all strings
        for i in range(min_len):
            common = True  # Flag to track if current character is common

            # Compare current character across adjacent strings
            for j in range(len(strs) - 1):
                if strs[j][i] != strs[j + 1][i]:
                    common = False
                    break

            # Add character to prefix if common, otherwise we're done
            if common:
                prefix = prefix + strs[0][i]
            else:
                break

        return prefix


def test_longest_common_prefix():
    """
    Test cases for finding longest common prefix.
    Each test case contains a list of strings and expected common prefix.

    Tests various scenarios including:
    - Basic cases with common prefixes
    - No common prefix
    - Same strings
    - Single string
    - Empty cases
    """
    test_cases = [
        (["flower", "flow", "flight"], "fl"),  # Basic case
        (["dog", "racecar", "car"], ""),  # No common prefix
        (["interspecies", "interstellar", "interstate"], "inters"),  # Longer prefix
        (["throne", "throne"], "throne"),  # Same strings
        (["car", "cir"], "c"),  # Single character prefix
        (["a"], "a"),  # Single string
        (["prefix", "prefix", "prefix"], "prefix"),  # All same strings
        ([], ""),  # Empty array
        (["cat", "catch", "cathedral"], "cat"),  # All starting with same word
    ]

    solution = Solution()
    for i, (strings, expected) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input strings: {strings}")
        print(f"Expected prefix: '{expected}'")

        result = solution.longestCommonPrefix(strings)
        print(f"Got prefix: '{result}'")

        assert result == expected, (
            f"Test case {i} failed!\n" f"Expected: '{expected}'\n" f"Got: '{result}'"
        )
        print("✓ Passed")


if __name__ == "__main__":
    test_longest_common_prefix()
    print("\nAll test cases passed!")
