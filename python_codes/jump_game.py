def print_array_with_markers(arr, p_value1 = -1, mrk1a = '«', mrk1b = '»', 
                            p_value2 = -1, mrk2a = '>', mrk2b = '<'):
    out = "["
    for i in range(len(arr)):
        if p_value1 == i:
            out += mrk1a
            out += str(arr[i]) + mrk1b + ", "
        elif p_value2 == i:
            out += mrk2a
            out += str(arr[i]) + mrk2b + ", "
        else:
            out += str(arr[i]) + ", "
    out = out[0:len(out) - 2]
    out += "]"
    return out


    
def jump_game(nums):
    if len(nums) == 1:
        return True
    target = len(nums) - 1
    for i in range(len(nums)-2, -1, -1):
        if i+nums[i]>=target:
            target = i
    
        if target == 0:
            return True
    return False

def main():
    nums = [
        [3, 2, 2, 0, 1, 4],
        [2, 3, 1, 1, 9],
        [3, 2, 1, 0, 4],
        [0],
        [1],
        [4, 3, 2, 1, 0],
        [1, 1, 1, 1, 1],
        [4, 0, 0, 0, 1],
        [3, 3, 3, 3, 3],
        [1, 2, 3, 4, 5, 6, 7]
    ]

    for i in range(len(nums)):
        print(i + 1, ".\tInput array: ", nums[i], sep="")
        print("\tCan we reach the very last index? ",
              "Yes" if jump_game(nums[i]) else "No", sep="")
        print("-" * 100)


if __name__ == '__main__':
    main()
