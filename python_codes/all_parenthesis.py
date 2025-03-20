def helper(n, left, right, result, current=""):
    if left == n and right == n:
        result.append(current)
        return

    if left < n:
        helper(n, left + 1, right, result, current + "(")
    if right < left:
        helper(n, left, right + 1, result, current + ")")
    
    
def generate_combinations(n):
    result = []
    
    helper(n, 0, 0, result, "")
    return result 

def print_result(result):
    for rs in result:
        print("\t\t ", rs)


# Driver code
def main():
    n = [1, 2, 3, 4, 5]

    for i in range(len(n)):
        print(i + 1, ".\t n = ", n[i], sep="")
        print("\t All combinations of valid balanced parentheses: ")

        result = generate_combinations(n[i])
        print_result(result)

        print("-" * 100)


if __name__ == '__main__':
    main()