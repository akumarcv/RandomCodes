import math
from linked_list import (
    LinkedList,
    reverse_linked_list_k,
    traverse_linked_list,
    print_list_with_forward_arrow,
)
from linked_list_node import LinkedListNode


def reverse_k_groups(head, k):
    """
    Reverse linked list nodes in groups of k.
    For each group of k nodes, the function reverses their order.
    Groups less than k nodes remain in original order.
    
    Args:
        head: Head node of the linked list
        k: Size of groups to reverse
        
    Returns:
        LinkedListNode: Head of the modified linked list
        
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) using constant extra space
    
    Example:
        >>> reverse_k_groups(LinkedList([1,2,3,4,5]).head, 2)
        2->1->4->3->5  # Nodes reversed in groups of 2
    """
    # Create dummy node to handle edge cases
    dummy = LinkedListNode(0)  # Dummy head for easier list manipulation
    dummy.next = head
    ptr = dummy  # Pointer to track position in list

    while ptr != None:
        # Check if there are k more nodes to process
        tracker = ptr  # Temporary pointer to look ahead k nodes
        
        for i in range(k):
            # Advance tracker k steps if possible
            if tracker == None:
                break
            
            tracker = tracker.next

        # If we couldn't move k steps, remaining nodes < k, so don't reverse
        if tracker == None:
            break

        # Reverse next k nodes
        previous, current = reverse_linked_list_k(ptr.next, k)
        
        # Store the node that will become the last after reversal
        last_node_of_reversed_group = ptr.next
        
        # Connect the reversed group's end to the rest of the list
        last_node_of_reversed_group.next = current
        
        # Connect the previous part to the start of reversed group
        ptr.next = previous
        
        # Move ptr to the last node of the reversed group
        ptr = last_node_of_reversed_group

    return dummy.next  # Return new head (skipping dummy node)


def main():
    """
    Driver code to test linked list group reversal.
    Tests various linked list configurations with different k values:
    - Standard lists with various k values
    - Edge cases (k=1, list length = k, etc.)
    - Lists with duplicate values
    - Single element lists
    
    For each test case:
    1. Creates the original linked list
    2. Prints the original list
    3. Reverses the list in groups of k
    4. Prints the modified list
    """
    input_list = [
        [1, 2, 3, 4, 5, 6, 7, 8],  # Standard list, k=3
        [3, 4, 5, 6, 2, 8, 7, 7],  # List with duplicates, k=2
        [1, 2, 3, 4, 5],           # k=1 (no actual reversal)
        [1, 2, 3, 4, 5, 6, 7],     # k greater than list length
        [1],                       # Single element list
    ]
    k = [3, 2, 1, 7, 1]  # Different group sizes to test

    for i in range(len(input_list)):
        input_linked_list = LinkedList()
        input_linked_list.create_linked_list(input_list[i])

        print(i + 1, ".\tLinked list: ", end=" ")
        print_list_with_forward_arrow(input_linked_list.head)
        print("\n")
        print("\tReversed linked list: ", end=" ")
        result = reverse_k_groups(input_linked_list.head, k[i])
        print_list_with_forward_arrow(result)
        print("\n")
        print("-" * 100)


if __name__ == "__main__":
    main()