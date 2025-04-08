import numpy as np
from typing import List, TypeVar, Any, Callable
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
import time

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


def reservoir_sampling_heap(k: int, nums: List[T]) -> List[T]:
    heap = []
    for i, num in enumerate(nums):
        # Generate a random weight for this element
        weight = np.random.random()

        if i < k:
            # For first k elements, simply add to heap
            heapq.heappush(heap, (weight, num))
        elif weight > heap[0][0]:
            # For remaining elements, if weight is larger than smallest in heap
            # replace the smallest element
            heapq.heappushpop(heap, (weight, num))

    # Extract just the elements (not the weights)
    return [(item[1], item[0]) for item in heap]  # Return (element, weight)


def test_correctness(
    sampling_func: Callable, k: int, nums: List[Any], num_trials: int = 10000
) -> dict:
    """
    Test the statistical correctness of a sampling function by running it many times
    and checking if each element appears with approximately equal probability.

    Parameters:
    -----------
    sampling_func : Callable
        The sampling function to test
    k : int
        Number of elements to sample
    nums : List[Any]
        List to sample from
    num_trials : int
        Number of sampling trials to run

    Returns:
    --------
    dict:
        Dictionary containing counts of each element's occurrence
    """
    # Initialize count dictionary
    count_dict = {i: 0 for i in nums}

    # Run multiple trials
    for _ in range(num_trials):
        sampled = sampling_func(k, nums)
        # Extract elements from tuples if sampling_heap was used
        if isinstance(sampled[0], tuple):
            sampled = [item[0] for item in sampled]
        for item in sampled:
            count_dict[item] += 1

    return count_dict


def test_performance(
    sampling_func: Callable, k: int, nums_sizes: List[int], num_trials: int = 10
) -> List[float]:
    """
    Test the performance of a sampling function with different input sizes.

    Parameters:
    -----------
    sampling_func : Callable
        The sampling function to test
    k : int
        Number of elements to sample
    nums_sizes : List[int]
        List of different input sizes to test
    num_trials : int
        Number of trials to run for each size (results will be averaged)

    Returns:
    --------
    List[float]:
        List of average execution times for each input size
    """
    times = []

    for size in nums_sizes:
        # Create a list of specified size
        nums = list(range(1, size + 1))
        total_time = 0

        # Run multiple trials to get reliable timing
        for _ in range(num_trials):
            start_time = time.time()
            sampling_func(k, nums)
            total_time += time.time() - start_time

        # Calculate average time
        avg_time = total_time / num_trials
        times.append(avg_time)
        print(f"Input size {size}: {avg_time:.6f} seconds")

    return times


def main():
    # Set seed for reproducibility
    np.random.seed(42)

    print("==== Testing Statistical Correctness ====")

    # Test data
    nums = list(range(1, 101))  # Numbers 1-100
    k = 10  # Sample size
    num_trials = 10000  # Number of sampling trials

    # Test both algorithms
    print(f"Running {num_trials} trials of reservoir sampling with k={k}...")
    standard_counts = test_correctness(reservoir_sampling, k, nums, num_trials)

    print(f"Running {num_trials} trials of heap-based reservoir sampling with k={k}...")
    heap_counts = test_correctness(reservoir_sampling_heap, k, nums, num_trials)

    # Calculate expected count for uniform distribution
    expected_count = k * num_trials / len(nums)
    print(f"Expected count per element: {expected_count:.2f}")

    # Calculate statistical measures
    standard_variance = np.var(list(standard_counts.values()))
    heap_variance = np.var(list(heap_counts.values()))

    print(f"Standard algorithm - variance of counts: {standard_variance:.2f}")
    print(f"Heap algorithm - variance of counts: {heap_variance:.2f}")

    # Create visualization
    plt.figure(figsize=(14, 7))

    # Plot standard algorithm counts
    plt.subplot(1, 2, 1)
    plt.bar(standard_counts.keys(), standard_counts.values(), alpha=0.7)
    plt.axhline(
        y=expected_count,
        color="r",
        linestyle="--",
        label=f"Expected ({expected_count:.1f})",
    )
    plt.title("Standard Reservoir Sampling")
    plt.xlabel("Element")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot heap algorithm counts
    plt.subplot(1, 2, 2)
    plt.bar(heap_counts.keys(), heap_counts.values(), alpha=0.7, color="green")
    plt.axhline(
        y=expected_count,
        color="r",
        linestyle="--",
        label=f"Expected ({expected_count:.1f})",
    )
    plt.title("Heap-based Reservoir Sampling")
    plt.xlabel("Element")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("reservoir_sampling_distribution.png", dpi=300)

    print("\n==== Testing Performance ====")

    # Test performance with different input sizes
    sizes = [100, 1000, 10000, 100000, 1000000]
    k_perf = 20

    print("\nStandard Algorithm:")
    standard_times = test_performance(reservoir_sampling, k_perf, sizes)

    print("\nHeap Algorithm:")
    heap_times = test_performance(reservoir_sampling_heap, k_perf, sizes)

    # Plot performance comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, standard_times, marker="o", label="Standard Algorithm")
    plt.plot(sizes, heap_times, marker="s", label="Heap Algorithm")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Input Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"Performance Comparison (k={k_perf})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reservoir_sampling_performance.png", dpi=300)


if __name__ == "__main__":
    main()
