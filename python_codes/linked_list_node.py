class LinkedListNode:
    """
    A node in a singly linked list.

    Attributes:
        value: The data stored in this node
        next: Reference to the next node in the list, or None if last node

    Example:
        >>> node = LinkedListNode(5)
        >>> node.value
        5
        >>> print(node.next)
        None
    """

    def __init__(self, value):
        """
        Initialize a new node with given value.

        Args:
            value: Data to store in the node
        """
        self.value = value  # Store the node's data
        self.next = None  # Initialize next pointer to None

    def __str__(self) -> str:
        """
        String representation of the node.

        Returns:
            str: Node's value as string
        """
        return str(self.value)

    def __repr__(self) -> str:
        """
        Detailed string representation of the node.

        Returns:
            str: Node details including value and next pointer status
        """
        return f"LinkedListNode(value={self.value}, next={'None' if self.next is None else 'Node'})"


def main():
    """
    Driver code to test LinkedListNode functionality.
    Creates sample nodes and demonstrates basic operations.
    """
    # Create sample nodes
    node1 = LinkedListNode(1)
    node2 = LinkedListNode(2)
    node3 = LinkedListNode(3)

    # Link nodes together
    node1.next = node2
    node2.next = node3

    # Print node information
    print("Created linked list:")
    current = node1
    while current:
        print(f"Node value: {current.value}")
        print(f"Has next node: {current.next is not None}")
        current = current.next
        print("-" * 50)


if __name__ == "__main__":
    main()
