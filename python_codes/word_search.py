def helper(grid, word, i, j, index):
    if index == len(word):
        return True

    if (
        i < 0
        or i >= len(grid)
        or j < 0
        or j >= len(grid[0])
        or grid[i][j] != word[index]
    ):
        return False

    temp = grid[i][j]
    grid[i][j] = " "
    found = (
        helper(grid, word, i + 1, j, index + 1)
        or helper(grid, word, i - 1, j, index + 1)
        or helper(grid, word, i, j + 1, index + 1)
        or helper(grid, word, i, j - 1, index + 1)
    )
    grid[i][j] = temp
    return found


def word_search(grid, word):

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if helper(grid, word, i, j, 0):
                return True
    return False


def main():
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
