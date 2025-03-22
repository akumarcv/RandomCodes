def is_happy_number(n):
    def helper(num):
        slow_pointer = num
        fast_pointer = sum([int(i) ** 2 for i in list(str(slow_pointer))])
        return fast_pointer

    slow_pointer = n
    fast_pointer = helper(n)

    while fast_pointer != 1 and fast_pointer != slow_pointer:
        slow_pointer = helper(slow_pointer)
        print(f"inside {slow_pointer, fast_pointer}")
        fast_pointer = helper(helper(fast_pointer))
    if fast_pointer == 1:
        return True

    # Replace this placeholder return statement with your code
    return False


if __name__ == "__main__":
    print(is_happy_number(7))
