import numpy as np

def moving_average(nums, k):
    result = []
    if k>len(nums):
        print("window size greater than nums")
        return 
    if k == 1:
        return nums
    for i in range(len(nums)-k+1):
        if i == 0:
            sums = sum(nums[:k])
            result.append(sums/k)
        else:
            sums = sums - nums[i-1]
            sums = sums + nums[i+k-1]
            result.append(sums/k)
    return result 

if __name__=="__main__":
    nums = [1,2,2,3,4,5,6,6,7,7,7,8,8]
    k = 3
    print(moving_average(nums, k))