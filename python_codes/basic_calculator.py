class Solution:

    def evaluate_expr(self, stack):

        # If stack is empty or the expression starts with
        # a symbol, then append 0 to the stack.
        # i.e. [1, '-', 2, '-'] becomes [1, '-', 2, '-', 0]
        if not stack or type(stack[-1]) == str:
            stack.append(0)

        res = stack.pop()

        # Evaluate the expression till we get corresponding ')'
        while stack and stack[-1] != ")":
            sign = stack.pop()
            if sign == "+":
                res += stack.pop()
            else:
                res -= stack.pop()
        return res

    def calculate(self, s: str) -> int:

        stack = []
        n, operand = 0, 0

        for i in range(len(s) - 1, -1, -1):
            ch = s[i]

            if ch.isdigit():

                # Forming the operand - in reverse order.
                operand = (10**n * int(ch)) + operand
                n += 1

            elif ch != " ":
                if n:
                    # Save the operand on the stack
                    # As we encounter some non-digit.
                    stack.append(operand)
                    n, operand = 0, 0

                if ch == "(":
                    res = self.evaluate_expr(stack)
                    stack.pop()

                    # Append the evaluated result to the stack.
                    # This result could be of a sub-expression within the parenthesis.
                    stack.append(res)

                # For other non-digits just push onto the stack.
                else:
                    stack.append(ch)

        # Push the last operand to stack, if any.
        if n:
            stack.append(operand)

        # Evaluate any left overs in the stack.
        return self.evaluate_expr(stack)


from basic_calculator import Solution


def test_calculator():
    solution = Solution()

    test_cases = [
        # Simple cases
        ("1 + 1", 2),
        ("2-1", 1),
        ("2 + 3 - 4", 1),
        # Expressions with spaces
        ("   2   ", 2),
        ("2 + 3   ", 5),
        # Expressions with parentheses
        ("(1+(4+5+2)-3)+(6+8)", 23),
        ("(1)", 1),
        ("2+(1)", 3),
        # Nested parentheses
        ("(1+(4+5+2)-(3))+(6+8)", 23),
        ("10 - (4 + 5 - 2)", 3),
        # More complex cases
        ("1 + (2 + 3) - 4", 2),
        ("(7)-(0)+(4)", 11),
        ("(1+2)-(3+4)", -4),
        # Edge cases
        ("0", 0),
        ("1", 1),
        ("(0)", 0),
    ]

    for i, (expression, expected) in enumerate(test_cases):
        result = solution.calculate(expression)
        status = "✓" if result == expected else "✗"
        print(f"Test case {i+1}: {expression}")
        print(f"Expected: {expected}, Result: {result} {status}")
        print("-" * 40)


if __name__ == "__main__":
    test_calculator()
