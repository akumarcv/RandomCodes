"""
We have n jobs, where every job is scheduled to be done from startTime[i] to endTime[i], obtaining a profit of profit[i].

You're given the startTime, endTime and profit arrays, return the maximum profit you can take such that there are no two jobs in the subset with overlapping time range.

If you choose a job that ends at time X you will be able to start another job that starts at time X.
Input: startTime = [1,2,3,3], endTime = [3,4,5,6], profit = [50,10,40,70]
Output: 120
Explanation: The subset chosen is the first and fourth job.
Time range [1-3]+[3-6] , we get profit of 120 = 50 + 70.
"""

import bisect


def jobScheduling(starttime, endtime, profits):
    """
    Calculate the maximum profit by selecting non-overlapping jobs.

    This function uses dynamic programming to find the optimal subset of jobs
    that don't overlap and yield the maximum possible profit.

    Algorithm:
    1. Sort all jobs by end time to process them in order of completion
    2. For each job, decide whether to include it or not:
       - If we include it, we add its profit to the max profit possible before it starts
       - If we don't include it, we take the max profit so far
    3. Use binary search to efficiently find the latest non-overlapping job

    Time Complexity: O(n log n) where n is the number of jobs
    Space Complexity: O(n) for the DP array

    Args:
        starttime (list[int]): List of job start times
        endtime (list[int]): List of job end times
        profits (list[int]): List of job profits

    Returns:
        int: Maximum profit achievable without job overlaps
    """
    # Create list of jobs and sort by end time for efficient processing
    jobs = sorted(zip(starttime, endtime, profits), key=lambda x: x[1])

    # DP array where dp[i] represents max profit up to job i
    dp = [0] * (len(jobs) + 1)

    # Custom binary search to find the latest job that finishes before the current job starts
    def binary_search(jobs, target, hi):
        """
        Binary search to find the latest non-overlapping job.

        Finds the largest index i such that jobs[i] ends before or at target time.

        Args:
            jobs (list): List of sorted jobs
            target (int): Start time of current job
            hi (int): Upper bound for search (exclusive)

        Returns:
            int: Index of the latest non-overlapping job
        """
        lo = 0
        while lo < hi:
            mid = (lo + hi) // 2
            if (
                jobs[mid][1] <= target
            ):  # If this job ends before or at our target start time
                lo = mid + 1  # Search in right half
            else:
                hi = mid  # Search in left half
        return (
            lo - 1 if lo > 0 else 0
        )  # Return the index of the latest non-overlapping job

    # Compute maximum profit using dynamic programming
    max_profit = 0
    for i in range(1, len(jobs) + 1):
        # Get current job details
        start, end, profit = jobs[i - 1]

        # Find the latest job that doesn't overlap with current job
        index = binary_search(jobs, start, i)

        # DP state transition:
        # Either take current job + max profit up to the last non-overlapping job,
        # or skip current job and take max profit so far
        dp[i] = max(dp[i - 1], dp[index] + profit)

        # Update overall maximum profit
        max_profit = max(max_profit, dp[i])

    return max_profit


# Driver code to test the function
if __name__ == "__main__":
    # Test case from the problem description
    startTime1 = [1, 2, 3, 3]
    endTime1 = [3, 4, 5, 6]
    profit1 = [50, 10, 40, 70]
    print("Test Case 1:")
    print(f"Start Times: {startTime1}")
    print(f"End Times: {endTime1}")
    print(f"Profits: {profit1}")
    print(f"Maximum Profit: {jobScheduling(startTime1, endTime1, profit1)}")
    print()

    # Additional test case
    startTime2 = [1, 2, 3, 4, 6]
    endTime2 = [3, 5, 10, 6, 9]
    profit2 = [20, 20, 100, 70, 60]
    print("Test Case 2:")
    print(f"Start Times: {startTime2}")
    print(f"End Times: {endTime2}")
    print(f"Profits: {profit2}")
    print(f"Maximum Profit: {jobScheduling(startTime2, endTime2, profit2)}")
    print()

    # Edge case with one job
    startTime3 = [1]
    endTime3 = [2]
    profit3 = [10]
    print("Test Case 3 (Single Job):")
    print(f"Start Times: {startTime3}")
    print(f"End Times: {endTime3}")
    print(f"Profits: {profit3}")
    print(f"Maximum Profit: {jobScheduling(startTime3, endTime3, profit3)}")
