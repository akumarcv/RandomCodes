import numpy as np
from typing import List, TypeVar, Generic

T = TypeVar("T")  # Define a type variable for generic list elements


def reservoir_sampling(k: int, nums: List[T]) -> List[T]:
    """
    Perform reservoir sampling to select k random elements from a stream of data.

    Reservoir sampling is an algorithm that allows selecting a random sample of k items
    from a list of n items where n is either unknown in advance or very large.
    This implementation uses Algorithm R by Jeffrey Vitter, which runs in O(n) time and O(k) space.

    Parameters:
    -----------
    k : int
        The number of elements to sample from the input list
    nums : List[T]
        The input list/stream of elements from which to sample

    Returns:
    --------
    List[T]
        A list containing k randomly selected elements from the input list

    Notes:
    ------
    The algorithm works as follows:
    1. First, put the first k elements from the input into the reservoir
    2. For each subsequent element (from k+1 to n):
       - Generate a random number r between 0 and the current index i
       - If r < k, replace the element at position r in the reservoir
         with the current element

    This ensures that each element in the input has an equal k/n probability
    of being in the final sample.

    Example:
    --------
    >>> reservoir_sampling(3, [1, 2, 3, 4, 5, 6, 7])
    [4, 6, 7]  # Example output (will vary due to randomness)
    """
    # Initialize the reservoir with the first k elements
    reservoir: List[T] = []

    for i in range(len(nums)):
        if i < k:
            # First fill the reservoir with the first k elements
            reservoir.append(nums[i])
        else:
            # For each subsequent element, randomly decide whether it should replace
            # an element in the reservoir
            rand_int = np.random.randint(
                0, i + 1
            )  # Generate random index between 0 and i

            # If random index is within the reservoir range, replace that element
            if rand_int < k:
                reservoir[rand_int] = nums[i]

    return reservoir


# Driver code
if __name__ == "__main__":
    # Test the reservoir sampling algorithm with a simple example
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    k = 5  # Size of the reservoir

    # Perform reservoir sampling
    result = reservoir_sampling(k, nums)
    print(f"Input array: {nums}")
    print(f"Reservoir sampling results (k={k}): {result}")

    # Demonstrate with a different value of k
    k2 = 3
    result2 = reservoir_sampling(k2, nums)
    print(f"Reservoir sampling results (k={k2}): {result2}")

    # Verify the distribution with multiple runs
    print("\nDistribution test - running 1000 times with k=3:")
    count_dict = {i: 0 for i in nums}

    # Run the sampling 1000 times and count occurrences of each number
    num_trials = 1000
    for _ in range(num_trials):
        sampled = reservoir_sampling(3, nums)
        for item in sampled:
            count_dict[item] += 1

    # Print the selection frequencies
    print("Element frequencies (should be approximately equal):")
    for num, count in count_dict.items():
        expected = (
            k2 * num_trials / len(nums)
        )  # Expected count for uniform distribution
        print(f"Number {num}: {count} times (Expected: ~{expected:.1f})")
