from typing import Optional


class ListNode:
    def __init__(self, k, next=None):
        self.val = k
        self.next = None


class Solution:
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        c = 0
        i = 0
        head = None
        prev = None
        i = 0
        while l1 is not None and l2 is not None:
            digit = (l1.val + l2.val + c) % 10
            c = (l1.val + l2.val + c) // 10
            node = ListNode(digit)
            if prev is None:
                head = node
                prev = node
            else:
                prev.next = node
                prev = node
            l1 = l1.next
            l2 = l2.next

        if l2 is not None:
            while l2 is not None:
                digit = (l2.val + c) % 10
                c = (l2.val + c) // 10
                node = ListNode(digit)
                prev.next = node
                prev = node
                l2 = l2.next

        if l1 is not None:
            while l1 is not None:
                digit = (l1.val + c) % 10
                c = (l1.val + c) // 10
                node = ListNode(digit)
                prev.next = node
                prev = node
                l1 = l1.next
        if c != 0:
            node = ListNode(c)
            prev.next = node
        return head


# Helper function to create a linked list from a list of values
def create_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for value in values[1:]:
        current.next = ListNode(value)
        current = current.next
    return head

# Helper function to convert a linked list to a list of values
def linked_list_to_list(node):
    result = []
    while node:
        result.append(node.val)
        node = node.next
    return result

# Driver code to test the addTwoNumbers function
if __name__ == "__main__":
    test_cases = [
        ([2, 4, 3], [5, 6, 4], [7, 0, 8]),  # 342 + 465 = 807
        ([0], [0], [0]),                    # 0 + 0 = 0
        ([9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9], [8, 9, 9, 9, 0, 0, 0, 1]),  # 9999999 + 9999 = 10009998
        ([1, 8], [0], [1, 8]),              # 81 + 0 = 81
        ([5], [5], [0, 1])                  # 5 + 5 = 10
    ]

    solution = Solution()
    for l1_values, l2_values, expected_values in test_cases:
        l1 = create_linked_list(l1_values)
        l2 = create_linked_list(l2_values)
        result = solution.addTwoNumbers(l1, l2)
        result_values = linked_list_to_list(result)
        print(f"Input: {l1_values} + {l2_values}, Expected: {expected_values}, Result: {result_values}")
        assert result_values == expected_values, f"Test failed for input {l1_values} + {l2_values}"
    print("All tests passed.")