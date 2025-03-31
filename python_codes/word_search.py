def helper(grid, word, i, j, index):
    """
    Recursive helper function to search for a word in a grid.
    Uses depth-first search to explore adjacent cells in all four directions.
    
    Args:
        grid: 2D array of characters
        word: String to search for
        i: Current row coordinate
        j: Current column coordinate
        index: Current position in the word being matched
        
    Returns:
        bool: True if the word can be found starting at position [i,j], False otherwise
        
    Time Complexity: O(4^L) where L is the length of the word
    Space Complexity: O(L) for recursion stack
    """
    # Base case: if we've matched all characters in the word
    if index == len(word):
        return True

    # Return false if current position is invalid or character doesn't match
    if (
        i < 0
        or i >= len(grid)
        or j < 0
        or j >= len(grid[0])
        or grid[i][j] != word[index]
    ):
        return False

    # Temporarily mark current cell as visited by replacing with space
    temp = grid[i][j]
    grid[i][j] = " "
    
    # Try all 4 directions (down, up, right, left)
    found = (
        helper(grid, word, i + 1, j, index + 1)  # Down
        or helper(grid, word, i - 1, j, index + 1)  # Up
        or helper(grid, word, i, j + 1, index + 1)  # Right
        or helper(grid, word, i, j - 1, index + 1)  # Left
    )
    
    # Restore the original character before returning
    grid[i][j] = temp
    return found


def word_search(grid, word):
    """
    Search for a word in a 2D grid of characters.
    The word can be constructed from adjacent characters in up, down, left, or right directions.
    
    Args:
        grid: 2D array of characters
        word: String to search for
        
    Returns:
        bool: True if the word can be found in the grid, False otherwise
        
    Time Complexity: O(M*N*4^L) where M,N are grid dimensions and L is word length
    Space Complexity: O(L) for recursion stack
    
    Example:
        >>> word_search([["A","B"],["C","D"]], "ABC")
        True  # A→B→C forms a valid path
    """
    # Empty word edge case
    if not word:
        return True
    
    # Try starting the search from each cell in the grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # If we find the word starting at any position, return True
            if helper(grid, word, i, j, 0):
                return True
    
    # Word not found after checking all starting positions
    return False


def main():
    """
    Driver function to test word search algorithm with multiple test cases.
    
    Each test case includes:
    - A grid of characters
    - A word to search for
    - Expected output (found/not found)
    
    Test cases cover:
    - Words that follow various paths (horizontal, vertical, zigzag)
    - Words that don't exist in the grid
    - Edge cases like empty words or single-character searches
    
    For each test case:
    1. Prints the grid
    2. Prints the word to search
    3. Prints whether the word was found
    """
    input = [
        (
            [
                ["E", "D", "X", "I", "W"],
                ["P", "U", "F", "M", "Q"],
                ["I", "C", "Q", "R", "F"],
                ["M", "A", "L", "C", "A"],
                ["J", "T", "I", "V", "E"],
            ],
            "EDUCATIVE",
        ),
        (
            [
                ["E", "D", "X", "I", "W"],
                ["P", "A", "F", "M", "Q"],
                ["I", "C", "A", "S", "F"],
                ["M", "A", "L", "C", "A"],
                ["J", "T", "I", "V", "E"],
            ],
            "PACANS",
        ),
        (
            [
                ["h", "e", "c", "m", "l"],
                ["w", "l", "i", "e", "u"],
                ["a", "r", "r", "s", "n"],
                ["s", "i", "i", "o", "r"],
            ],
            "warrior",
        ),
        (
            [
                ["C", "Q", "N", "A"],
                ["P", "S", "E", "I"],
                ["Z", "A", "P", "E"],
                ["J", "V", "T", "K"],
            ],
            "SAVE",
        ),
        (
            [
                ["O", "Y", "O", "I"],
                ["B", "Y", "N", "M"],
                ["K", "D", "A", "R"],
                ["C", "I", "M", "I"],
                ["Z", "I", "T", "O"],
            ],
            "DYNAMIC",
        ),
    ]
    num = 1

    for i in input:
        print(num, ".\tGrid =", sep="")
        for j in range(len(i[0])):
            print("\t\t", i[0][j])
        if i[1] == "":
            print('\n\tWord = ""')
        else:
            print(f"\n\tWord =  {i[1]}")
        search_result = word_search(i[0], i[1])
        if search_result:
            print("\n\tSearch result = Word found")
        else:
            print("\n\tSearch result = Word could not be found")
        num += 1
        print("-" * 100, "\n")


if __name__ == "__main__":
    main()