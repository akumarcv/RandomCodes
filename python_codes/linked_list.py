from linked_list_node import LinkedListNode


class LinkedList:
    """
    Implementation of a singly linked list data structure.
    Supports basic operations like insertion and traversal.
    """
    
    def __init__(self):
        """Initialize an empty linked list."""
        self.head = None  # Reference to first node

    def insert_node_at_head(self, node: LinkedListNode) -> None:
        """
        Insert a new node at the beginning of the list.
        
        Args:
            node: LinkedListNode to insert at head
        """
        if self.head:
            node.next = self.head
            self.head = node
        else:
            self.head = node

    def create_linked_list(self, lst: list) -> None:
        """
        Create linked list from given list of values.
        
        Args:
            lst: List of values to convert to linked list
        """
        for x in reversed(lst):
            new_node = LinkedListNode(x)
            self.insert_node_at_head(new_node)

    def __str__(self) -> str:
        """
        String representation of linked list.
        
        Returns:
            str: Comma-separated values in list
        """
        result = ""
        temp = self.head
        while temp:
            result += str(temp.value)  # Changed from temp.data to temp.value
            temp = temp.next
            if temp:
                result += ", "
        return result


def print_list_with_forward_arrow(linked_list_node: LinkedListNode) -> None:
    """
    Print linked list with arrow notation showing connections.
    
    Args:
        linked_list_node: Head of the linked list to print
    
    Example:
        >>> print_list_with_forward_arrow(head)
        1 → 2 → 3 → null
    """
    temp = linked_list_node
    while temp:
        print(temp.value, end=" ")
        temp = temp.next
        if temp:
            print("→", end=" ")
        else:
            print("→ null", end=" ")


def traverse_linked_list(head: LinkedListNode) -> None:
    """
    Simple traversal of linked list without modifications.
    
    Args:
        head: First node of the linked list
    """
    current, nxt = head, None
    while current:
        nxt = current.next
        current = nxt


def reverse_linked_list(head: LinkedListNode) -> LinkedListNode:
    """
    Reverse a linked list in-place.
    
    Args:
        head: First node of the linked list
        
    Returns:
        LinkedListNode: New head of reversed list
        
    Time Complexity: O(n) where n is length of list
    Space Complexity: O(1) as reversal is done in-place
    """
    prev, curr = None, head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev


def reverse_linked_list_k(head: LinkedListNode, k: int) -> tuple[LinkedListNode, LinkedListNode]:
    """
    Reverse first k nodes of linked list.
    
    Args:
        head: First node of the linked list
        k: Number of nodes to reverse
        
    Returns:
        tuple: (new_head, next_node) where:
            - new_head is head of reversed section
            - next_node is first node after reversed section
            
    Time Complexity: O(k)
    Space Complexity: O(1)
    """
    previous, current, next = None, head, None
    for _ in range(k):
        next = current.next      # Store next node
        current.next = previous  # Reverse current node
        previous = current       # Move previous pointer
        current = next          # Move to next node
    return previous, current