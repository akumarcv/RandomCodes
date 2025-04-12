def daily_temperatures(temperatures):
    """
    Given an array of daily temperatures, return an array such that for each day in the input,
    tells you how many days you would have to wait until a warmer temperature.
    If there is no future day for which this is possible, put 0 instead.

    Args:
        temperatures (List[int]): List of daily temperatures

    Returns:
        List[int]: List where each element is the number of days to wait for a warmer temperature

    Time Complexity: O(n) - where n is the length of temperatures list
    Space Complexity: O(n) - for the output and stack
    """
    n = len(temperatures)
    output = [0] * n  # Initialize output with 0's
    stack = []  # Stack to store indices of temperatures

    for i in range(n):
        # While we have temperatures in the stack and current temperature is higher
        # than the temperature at the top of the stack
        while stack and temperatures[i] > temperatures[stack[-1]]:
            j = stack.pop()  # Get the index from the stack
            output[j] = i - j  # Calculate days difference and update output
        stack.append(i)  # Add current index to stack

    return output


# Driver code
if __name__ == "__main__":
    # Test case 1
    temperatures1 = [73, 74, 75, 71, 69, 72, 76, 73]
    expected1 = [1, 1, 4, 2, 1, 1, 0, 0]
    result1 = daily_temperatures(temperatures1)
    print("Test case 1:")
    print(f"Input: {temperatures1}")
    print(f"Expected: {expected1}")
    print(f"Result: {result1}")
    print(f"Test passed: {result1 == expected1}\n")

    # Test case 2
    temperatures2 = [30, 40, 50, 60]
    expected2 = [1, 1, 1, 0]
    result2 = daily_temperatures(temperatures2)
    print("Test case 2:")
    print(f"Input: {temperatures2}")
    print(f"Expected: {expected2}")
    print(f"Result: {result2}")
    print(f"Test passed: {result2 == expected2}\n")

    # Test case 3
    temperatures3 = [30, 60, 90]
    expected3 = [1, 1, 0]
    result3 = daily_temperatures(temperatures3)
    print("Test case 3:")
    print(f"Input: {temperatures3}")
    print(f"Expected: {expected3}")
    print(f"Result: {result3}")
    print(f"Test passed: {result3 == expected3}\n")

    # Test case 4
    temperatures4 = [90, 60, 30]
    expected4 = [0, 0, 0]
    result4 = daily_temperatures(temperatures4)
    print("Test case 4:")
    print(f"Input: {temperatures4}")
    print(f"Expected: {expected4}")
    print(f"Result: {result4}")
    print(f"Test passed: {result4 == expected4}")
