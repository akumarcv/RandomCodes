from typing import List


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs) == 0:
            return ""

        prefix = ""

        min_len = min([len(s) for s in strs])

        for i in range(min_len):
            common = True
            for j in range(len(strs) - 1):
                if strs[j][i] != strs[j + 1][i]:
                    common = False
                    break
            if common:
                prefix = prefix + strs[0][i]
            else:
                break

        return prefix
    

# ...existing code...

def test_longest_common_prefix():
    """
    Test cases for finding longest common prefix
    Each test case contains a list of strings and expected common prefix
    """
    test_cases = [
        (["flower", "flow", "flight"], "fl"),    # Basic case
        (["dog", "racecar", "car"], ""),         # No common prefix
        (["interspecies", "interstellar", "interstate"], "inters"),  # Longer prefix
        (["throne", "throne"], "throne"),        # Same strings
        (["car", "cir"], "c"),                              # Single empty string
        (["a"], "a"),                            # Single character
        (["prefix", "prefix", "prefix"], "prefix"), # All same strings
        ([], ""),                                # Empty array
        (["cat", "catch", "cathedral"], "cat")   # All starting with same word
    ]
    
    solution = Solution()
    for i, (strings, expected) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input strings: {strings}")
        print(f"Expected prefix: '{expected}'")
        
        result = solution.longestCommonPrefix(strings)
        print(f"Got prefix: '{result}'")
        
        assert result == expected, (
            f"Test case {i} failed!\n"
            f"Expected: '{expected}'\n"
            f"Got: '{result}'"
        )
        print("✓ Passed")

if __name__ == "__main__":
    test_longest_common_prefix()
    print("\nAll test cases passed!")