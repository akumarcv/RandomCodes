def permute_word(word):
    if word=="":
        return [""]
    else:
        result = []
        for i in range(len(word)):
            first = word[i]
            rest = word[:i] + word[i+1:]
            for p in permute_word(rest):
                result.append(first + p)
        return result

# Driver code
def main():
    input_word = ["ab", "bad", "abcd"]

    for index in range(len(input_word)):
        permuted_words = permute_word(input_word[index])

        print(index + 1, ".\t Input string: '", input_word[index], "'", sep="")
        print("\t All possible permutations are: ",
              "[", ', '.join(permuted_words), "]", sep="")
        print('-' * 100)


if __name__ == '__main__':
    main()
