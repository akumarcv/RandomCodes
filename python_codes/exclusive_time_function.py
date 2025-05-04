from collections import deque  # Import deque from collections module


# Creating a stack
def create_stack():  # Function to create an empty stack
    stack = []  # Initialize an empty list to serve as a stack
    return stack  # Return the initialized stack


# Checking if stack is empty or not
def check_empty(stack):  # Function to check if stack is empty
    return len(stack) == 0  # Return True if stack length is 0, otherwise False


# Push function adds items into the stack
def push(stack, item):  # Function to add an item to the stack
    stack.append(item)  # Append the item to the end of the stack
    print("Pushed item: " + item)  # Print a message indicating the item was pushed


# Pop function removes an element from the stack
def pop(stack):  # Function to remove and return the top element from the stack
    if check_empty(stack):  # Check if the stack is empty
        return "Stack is empty"  # Return a message if stack is empty

    return stack.pop()  # Remove and return the top element of the stack


class Log:  # Class to represent a function execution log
    def __init__(self, content):  # Constructor method that takes a log string
        content = content.replace(" ", "")  # Remove any spaces from the content
        content = content.split(":")  # Split the content by colon
        self.id = int(content[0])  # Convert the first part to int as function id
        self.is_start = content[1] == "start"  # Check if it's a start log
        self.time = int(content[2])  # Convert the third part to int as timestamp


def exclusive_time(n, logs):  # Function to calculate exclusive time for each function
    """
    Calculate the exclusive time of each function in a single-threaded CPU.

    Args:
        n (int): The number of functions.
        logs (list): List of strings representing function execution logs.
                     Each log is in the format "function_id:start/end:timestamp".

    Returns:
        list: An array of length n where the ith element represents the exclusive
              time spent by the function with id i.

    Time Complexity: O(m), where m is the number of logs
    Space Complexity: O(m) for the stack in worst case, where all functions start before any ends
    """
    stack = []  # Initialize an empty stack to track function calls
    result = [0] * n  # Initialize result array with zeros for each function
    for i in range(len(logs)):  # Iterate through each log entry
        if "start" in logs[i]:  # If the log entry is a start log
            stack.append(logs[i])  # Push the log to the stack
        elif "end" in logs[i]:  # If the log entry is an end log
            item = stack.pop()  # Pop the corresponding start log
            splits = item.split(":")  # Split the start log
            duration = (
                int(logs[i].split(":")[-1]) - int(splits[-1]) + 1
            )  # Calculate execution time
            result[
                int(splits[0])
            ] += duration  # Add duration to the function's total time
            if stack:  # If there are other functions on the stack
                result[int(stack[-1].split(":")[0])] -= (
                    int(logs[i].split(":")[-1]) - int(splits[-1]) + 1
                )  # Subtract time from parent function
    return result  # Return the array of exclusive times


def main():  # Main function to test the exclusive_time function
    """
    Driver function to test the exclusive_time function with different test cases.
    """
    # Test cases: List of logs for different scenarios
    logs = [
        [
            "0:start:0",
            "1:start:2",
            "1:end:3",
            "2:start:4",
            "2:end:7",
            "0:end:8",
        ],  # Test case 1
        [
            "0:start:0",
            "0:start:2",
            "0:end:5",
            "1:start:6",
            "1:end:6",
            "0:end:7",
        ],  # Test case 2
        ["0:start:0", "1:start:5", "1:end:6", "0:end:7"],  # Test case 3
        [
            "0:start:0",
            "1:start:5",
            "2:start:8",
            "3:start:12",
            "4:start:15",
            "5:start:19",
            "5:end:22",
            "4:end:24",
            "3:end:27",
            "2:end:32",
            "1:end:35",
            "0:end:36",
        ],  # Test case 4
        ["0:start:0", "1:start:3", "1:end:6", "0:end:10"],  # Test case 5
    ]
    # Number of functions for each test case
    n = [3, 2, 2, 6, 2]  # Number of functions for each test case
    x = 1  # Counter for test case numbering

    # Run each test case and print results
    for i in range(len(n)):  # Iterate through each test case
        print(x, ".\tn = ", n[i], sep="")  # Print test case number and n value
        print("\tlogs = ", logs[i], sep="")  # Print the logs for current test case
        print(
            "\tOutput: ", exclusive_time(n[i], logs[i]), sep=""
        )  # Print the output of exclusive_time function
        print("-" * 100, "\n", sep="")  # Print a separator line
        x += 1  # Increment the test case counter


if __name__ == "__main__":  # Check if the script is being run directly
    main()  # Call the main function
