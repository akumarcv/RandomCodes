from collections import deque, defaultdict, Counter
from typing import List, Dict, Tuple


def word_ladder(src: str, dest: str, words: List[str]) -> int:
    """
    Find the length of the shortest transformation sequence from source to destination word.

    A transformation sequence is a sequence where each word differs by just one letter from the previous word.
    All words in the sequence must exist in the given word list.

    Args:
        src: Source word from which to start the transformation
        dest: Destination word to reach
        words: List of valid words that can be used in the transformation

    Returns:
        int: Length of shortest transformation sequence from src to dest.
             Returns 0 if no such transformation exists.

    Example:
        word_ladder("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) -> 5
        Transformation: hit -> hot -> dot -> dog -> cog
    """
    # Edge cases: if source and destination are the same, or if destination is not in word list
    if src == dest:
        return 0
    if dest not in words:
        return 0

    # Convert word list to set for O(1) lookups and ensure source and destination are included
    words = set(words)
    words.add(src)
    words.add(dest)

    # BFS implementation using queue
    queue = deque([src])
    visited = set([src])
    level = 0

    while queue:
        level += 1
        # Process all words at the current level
        for _ in range(len(queue)):
            word = queue.popleft()

            # Try changing each character position with all possible letters
            for i in range(len(word)):
                for c in "abcdefghijklmnopqrstuvwxyz":
                    new_word = word[:i] + c + word[i + 1 :]

                    # If we found the destination word, return the current level
                    if new_word == dest:
                        return level

                    # If the new word is valid and hasn't been visited, add to queue
                    if new_word in words and new_word not in visited:
                        visited.add(new_word)
                        queue.append(new_word)

    # If no transformation sequence exists
    return 0


# Example usage:
if __name__ == "__main__":
    src = "hit"
    dest = "cog"
    words = ["hot", "dot", "dog", "lot", "log", "cog"]
    result = word_ladder(src, dest, words)
    print(f"The length of the shortest transformation sequence is: {result}")
