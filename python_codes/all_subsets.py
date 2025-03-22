def find_all_subsets(nums):
    if len(nums) == 0:
        return [[]]
    else:
        total_subsets = 2 ** len(nums)
        subsets = []
        for i in range(total_subsets):
            subset = []
            for j in range(len(nums)):
                if i & (1 << j):
                    subset.append(nums[j])
            subsets.append(subset)
        print(subsets)
        return subsets
    # Replace this placeholder return statement with your code
    return [[]]


def main():
    nums = [[], [2, 5, 7], [1, 2], [1, 2, 3, 4], [7, 3, 1, 5]]

    for i in range(len(nums)):
        print(i + 1, ". Set:     ", nums[i], sep="")
        find_all_subsets(nums[i])
        print("-" * 100)


if __name__ == "__main__":
    main()
