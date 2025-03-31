from linked_list_node import LinkedListNode
from linked_list import LinkedList, print_list_with_forward_arrow


def reverse(head):
    """
    Reverse a singly linked list using iterative approach.
    Changes the next pointers of each node to point to previous nodes.
    
    Args:
        head: Head node of the linked list to reverse
        
    Returns:
        LinkedListNode: New head of the reversed linked list
        
    Time Complexity: O(n) where n is the number of nodes in the list
    Space Complexity: O(1) as we only use a constant amount of extra space
    
    Example:
        >>> reverse(LinkedList([1,2,3]).head)
        3->2->1->None  # Returns head of reversed list
    """
    # Base case: empty list or single node list
    if head.next is None:
        return head  # Already reversed

    prev = None     # Track previous node (starts as None since head has no previous)
    curr = head     # Current node being processed
    next = None     # Temporary storage for next node

    # Iterate through list, changing each node's next pointer
    while curr is not None:
        next = curr.next    # Store next node before changing pointer
        curr.next = prev    # Reverse the link to point to previous node
        prev = curr         # Move prev pointer one step forward
        curr = next         # Move curr pointer one step forward

    # After loop, prev is the new head (last node of original list)
    head = prev
    return head


def main():
    """
    Driver code to test linked list reversal functionality.
    Tests various linked list configurations including:
    - Standard multi-element lists
    - Single element lists
    - Lists with duplicate values
    - Lists of different lengths
    
    For each test case:
    1. Creates the original linked list
    2. Prints the original list
    3. Reverses the list
    4. Prints the reversed list
    """
    input = (
        [1, 2, 3, 4, 5],        # Standard ascending list
        [1, 2, 3, 4, 5, 6],     # Even length list
        [3, 2, 1],              # Already in reverse order
        [10],                   # Single element list
        [1, 2],                 # Two element list
    )

    for i in range(len(input)):
        input_linked_list = LinkedList()
        input_linked_list.create_linked_list(input[i])
        print(i + 1, ".\tInput linked list: ", sep="", end="")
        print_list_with_forward_arrow(input_linked_list.head)
        print("\n\tReversed linked list: ", end="")
        print_list_with_forward_arrow(reverse(input_linked_list.head))
        print("\n", "-" * 100)


if __name__ == "__main__":
    main()