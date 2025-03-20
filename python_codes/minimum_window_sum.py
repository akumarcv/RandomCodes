def min_sub_array_len(target, nums):

    min_length = float("inf")
    left = 0
    
    while left<len(nums):
        sum = 0
        right = left 
        while sum<target and right<len(nums):
            sum = sum + nums[right]
            right = right + 1
        if sum>=target:
            temp = left 
            while temp<right and sum>=target:
                sum = sum - nums[temp]
                temp = temp+1
            
            min_length = min(min_length, right-temp+1)
            
        left = left + 1
    # Replace this placeholder return statement with your code
    
    return min_length if min_length!=float("inf") else 0


def main():
    target = [7, 4, 11, 10, 5, 15]
    input_arr = [[2, 3, 1, 2, 4, 3], [1, 4, 4], [1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 2, 3, 4], [1, 2, 1, 3], [5, 4, 9, 8, 11, 3, 7, 12, 15, 44]]
    for i in range(len(input_arr)):
        window_size = min_sub_array_len(target[i], input_arr[i])
        print(i+1, ".\t Input array: ", input_arr[i],"\n\t Target: ", target[i],
            "\n\t Minimum Length of Subarray: ", window_size, sep="")
        print("-"*100)


if __name__ == "__main__":
    main()