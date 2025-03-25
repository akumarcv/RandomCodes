import numpy as np

"""
Given the API rand7() that generates a uniform random integer in the range [1, 7], 
write a function rand10() that generates a uniform random integer in the range [1, 10]. 
You can only call the API rand7(), and you shouldn't call any other API. 
Please do not use a language's built-in random API.
"""

np.random.seed(22)


def rand7():
    return np.random.randint(1, 7)


def rand10(n):
    output = []
    for i in range(n):
        random_num = 7 * (rand7() - 1) + rand7()
        print(random_num)
        if random_num < 40:
            output.append(random_num // 4 + 1)
    return output


# Driver code
if __name__ == "__main__":
    n = 10  # Number of random numbers to generate
    random_numbers = rand10(n)
    print(f"Generated {n} random numbers in the range [1, 10]: {random_numbers}")
