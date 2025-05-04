from linked_list_node import LinkedListNode as ListNode
from typing import List, Optional


class Solution:
    """
    Solution class for merging k sorted linked lists.

    This implementation uses a divide and conquer approach to merge k sorted linked lists
    with an overall time complexity of O(N log k), where N is the total number of nodes
    across all lists and k is the number of linked lists.
    """

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Merge k sorted linked lists into one sorted linked list.

        Args:
            lists: List of ListNode objects representing the heads of k sorted linked lists

        Returns:
            Head of the merged sorted linked list

        Time Complexity: O(N log k) where N is the total number of nodes and k is the number of lists
        Space Complexity: O(log k) for the recursion stack
        """
        # Handle base cases
        if len(lists) == 0:
            return None
        if len(lists) == 1:
            return lists[0]
        result = self.mergek(lists)
        return result

    def mergek(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Helper method that recursively divides the lists and merges them.

        Args:
            lists: List of ListNode objects

        Returns:
            Head of the merged sorted linked list
        """
        # Base case: when only one list remains
        if len(lists) < 2:
            return lists[0]

        # Divide the lists into two halves
        middle = len(lists) // 2

        # Recursively merge each half
        left = self.mergeKLists(lists[:middle])
        right = self.mergeKLists(lists[middle:])

        # Merge the two sorted halves
        return self.merge2(left, right)

    def merge2(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        """
        Merge two sorted linked lists into one sorted linked list.

        Args:
            l1: Head of the first sorted linked list
            l2: Head of the second sorted linked list

        Returns:
            Head of the merged sorted linked list

        Time Complexity: O(n + m) where n and m are the lengths of the two lists
        """
        # Create a dummy head to simplify the merging process
        head = dummy = ListNode(0)

        # Iterate through both lists and append the smaller node to the result
        while l1 and l2:
            if l1.value <= l2.value:
                dummy.next = l1
                l1 = l1.next
            else:
                dummy.next = l2
                l2 = l2.next
            dummy = dummy.next

        # Append remaining nodes from either list
        dummy.next = l1 if l1 is not None else l2

        return head.next


# Driver code
if __name__ == "__main__":
    # Helper function to create a linked list from a list
    def create_linked_list(values):
        if not values:
            return None
        head = ListNode(values[0])
        current = head
        for val in values[1:]:
            current.next = ListNode(val)
            current = current.next
        return head

    # Helper function to print a linked list
    def print_list(head):
        result = []
        current = head
        while current:
            result.append(str(current.value))
            current = current.next
        print(" -> ".join(result))

    # Create sample input lists
    list1 = create_linked_list([1, 4, 5])
    list2 = create_linked_list([1, 3, 4])
    list3 = create_linked_list([2, 6])

    lists = [list1, list2, list3]

    # Run the solution
    solution = Solution()
    merged_list = solution.mergeKLists(lists)

    # Print the result
    print("Merged list:")
    print_list(merged_list)  # Expected: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6

    # Test with edge cases
    print("\nEdge cases:")
    print("Empty list:", "None" if solution.mergeKLists([]) is None else "Not None")
    print(
        "Single empty list:",
        "None" if solution.mergeKLists([None]) is None else "Not None",
    )
    print(
        "Multiple empty lists:",
        "None" if solution.mergeKLists([None, None]) is None else "Not None",
    )
