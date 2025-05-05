from typing import List, Set


class Trie:
    """
    Trie data structure for efficient word lookup.
    Used to store and search words in the word search problem.
    """

    def __init__(self):
        self.root = {}  # Root of the trie

    def insert(self, word: str) -> None:
        """
        Insert a word into the trie

        Args:
            word: The word to be inserted

        Time Complexity: O(m) where m is the length of the word
        Space Complexity: O(m)
        """
        root = self.root
        for ch in word:
            if ch not in root:
                root[ch] = {}
            root = root[ch]
        root["*"] = True  # Mark the end of a word


class Solution:
    """
    Solution for Word Search II problem (LeetCode 212)
    Find all words from the board that are in the given word list
    """

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Find all words from the board that are in the given word list using DFS and Trie

        Args:
            board: 2D board containing characters
            words: List of words to search for

        Returns:
            List of words found on the board

        Time Complexity: O(M * N * 4^L) where:
            - M, N are the dimensions of the board
            - L is the maximum length of any word
        Space Complexity: O(K) where K is the total number of characters in all words
        """

        def dfs(r: int, c: int, root: dict, partial: str) -> None:
            """
            Depth-first search to find words on the board

            Args:
                r: Current row
                c: Current column
                root: Current trie node
                partial: Partial word formed so far
            """
            if 0 <= r < M and 0 <= c < N:
                letter = board[r][c]
                if letter in root:
                    next_trie_root = root[letter]
                    if "*" in next_trie_root:
                        self.result.add(partial + letter)  # Found a complete word
                        # Optimization: if this is a leaf node with no other children, we can stop
                        if len(next_trie_root) == 1:
                            return

                    # Mark the cell as visited by emptying it
                    board[r][c] = ""
                    # Explore all four directions
                    dfs(r + 1, c, next_trie_root, partial + letter)  # Down
                    dfs(r, c + 1, next_trie_root, partial + letter)  # Right
                    dfs(r - 1, c, next_trie_root, partial + letter)  # Up
                    dfs(r, c - 1, next_trie_root, partial + letter)  # Left
                    # Restore the cell after exploration
                    board[r][c] = letter

        # Initialize trie and insert all words
        self.trie = Trie()
        for word in words:
            self.trie.insert(word)

        # Set to store found words (using set to avoid duplicates)
        self.result = set()
        M = len(board)
        N = len(board[0]) if M > 0 else 0

        # Start DFS from each cell
        for r in range(M):
            for c in range(N):
                dfs(r, c, self.trie.root, "")

        return list(self.result)


# Driver code
if __name__ == "__main__":
    # Example 1
    board = [
        ["o", "a", "a", "n"],
        ["e", "t", "a", "e"],
        ["i", "h", "k", "r"],
        ["i", "f", "l", "v"],
    ]
    words = ["oath", "pea", "eat", "rain"]
    solution = Solution()
    result = solution.findWords(board, words)
    print("Example 1 Result:", result)  # Expected output: ['eat', 'oath']

    # Example 2
    board2 = [["a", "b"], ["c", "d"]]
    words2 = ["abcb"]
    result2 = solution.findWords(board2, words2)
    print("Example 2 Result:", result2)  # Expected output: []

    # Example 3
    board3 = [["a"]]
    words3 = ["a"]
    result3 = solution.findWords(board3, words3)
    print("Example 3 Result:", result3)  # Expected output: ['a']
