def max_product(nums):

    max_so_far = nums[0]
    min_so_far = nums[0]
    result = max_so_far

    for i in range(1, len(nums)):
        # three terms take care of 0, positive and negative numbers
        previos_max = max_so_far
        max_so_far = max(nums[i], max_so_far * nums[i], min_so_far * nums[i])
        min_so_far = min(nums[i], previos_max * nums[i], min_so_far * nums[i])
        result = max(result, max_so_far)
    return result


# Driver code
def main():
    input_bits = [
        [-2, 0, -1],
        [2, 3, -2, 4],
        [2, -5, 3, 1, -4, 0, -10, 2],
        [1, 2, 3, 0, 4],
        [5, 4, 3, 10, 4, 1],
    ]

    for i in range(len(input_bits)):
        print(i + 1, ".\t Input array: ", input_bits[i], sep="")
        print("\n\t Maximum product: ", max_product(input_bits[i]), sep="")
        print("-" * 100)


if __name__ == "__main__":
    main()
