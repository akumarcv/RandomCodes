import numpy as np
from typing import List


def reservoir_sampling(k: int, nums: list):
    reservoir: List = []

    for i in range( len(nums)):
        if i<k:
            reservoir.append(nums[i])
        else:
            rand_int = np.random.randint(0, i)
            
            if rand_int < k:
                reservoir[rand_int] = nums[i]

    return reservoir


# Driver code
if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    k = 5  # Size of the reservoir

    result = reservoir_sampling(k, nums)
    print(f"Input array: {nums}")
    print(f"Reservoir sampling results (k={k}): {result}")