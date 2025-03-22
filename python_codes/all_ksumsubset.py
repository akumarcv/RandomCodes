def helper(nums, current, sum, subsets, k):
    if sum == k:
        subsets.append(current)
        return
    if sum > k:
        return
    for i in range(len(nums)):
        helper(nums[i + 1 :], current + [nums[i]], sum + nums[i], subsets, k)
    return subsets


def get_k_sum_subsets(nums, k):

    # Replace this placeholder return statement with your code
    subsets = []
    helper(nums, [], 0, subsets, k)
    return subsets


def main():
    nums = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    k = [5, 10, 15]

    for i in range(len(nums)):
        print(i + 1, ". Set:     ", nums[i], sep="")
        print("\t k = ", k[i], sep="")
        print("\t All subsets with sum k: ")
        print(get_k_sum_subsets(nums[i], k[i]))
        print("-" * 100)


if __name__ == "__main__":
    main()
