import numpy as np


def pickIndex(nums):

    cumsum = [sum(nums[: i + 1]) for i in range(len(nums))]

    uniform = np.random.uniform(0, cumsum[-1])
    for i, val in enumerate(cumsum):
        if uniform < val:
            return i


# Driver code
if __name__ == "__main__":
    test_arrays = [
        [1, 3, 2, 4],
        [10, 20, 30, 40],
        [5, 5, 5, 5],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5],
    ]

    for nums in test_arrays:

        results = [pickIndex(nums) for _ in range(20)]
        print(f"Array: {nums}")
        print(f"Picked indices: {results}\n")
