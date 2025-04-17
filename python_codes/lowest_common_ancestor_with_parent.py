"""
This module implements a solution for finding the Lowest Common Ancestor (LCA) of two nodes in a binary tree.
Unlike traditional LCA problems, this implementation utilizes parent pointers, which allows for an efficient
solution without the need to traverse from the root each time.

The algorithm uses a two-pointer approach similar to finding the intersection point of two linked lists.
"""

from typing import List
from queue import Queue
from collections import deque
from binary_tree import display_tree


class EduTreeNode:
    """
    A binary tree node implementation that includes a parent pointer.

    Attributes:
        data: The value stored in the node
        left: Reference to the left child node
        right: Reference to the right child node
        parent: Reference to the parent node
    """

    def __init__(self, data):
        """
        Initialize a new tree node with the given data.

        Args:
            data: The value to be stored in the node
        """
        self.data = data
        self.left = None
        self.right = None
        self.parent = None


class EduBinaryTree:
    """
    Binary tree implementation that supports creation from a list and node lookup.
    Each node in this tree has a parent pointer for easier traversal.

    Attributes:
        root: The root node of the binary tree
    """

    def __init__(self, nodes):
        """
        Initialize a binary tree from a list of node values.

        Args:
            nodes: A list of values to create the binary tree with level-order traversal
        """
        self.root = self.createBinaryTree(nodes)

    def createBinaryTree(self, nodes):
        """
        Create a binary tree from a list of values using level-order traversal.

        Args:
            nodes: A list of values to create the binary tree

        Returns:
            The root node of the created binary tree or None if the input is empty
        """
        # Return None for empty input
        if not nodes or nodes[0] is None:
            return None

        # Create root node
        root = EduTreeNode(nodes[0])
        q = deque([root])  # Queue for level-order traversal
        i = 1  # Index to track current position in nodes list

        # Process nodes level by level
        while i < len(nodes):
            curr = q.popleft()  # Get the next node to add children to

            # Add left child if available
            if i < len(nodes) and nodes[i] is not None:
                curr.left = EduTreeNode(nodes[i])
                curr.left.parent = curr  # Set parent reference
                q.append(curr.left)
            i += 1

            # Add right child if available
            if i < len(nodes) and nodes[i] is not None:
                curr.right = EduTreeNode(nodes[i])
                curr.right.parent = curr  # Set parent reference
                q.append(curr.right)
            i += 1
        return root

    def find(self, root, value):
        """
        Find a node with the given value in the binary tree.

        Args:
            root: The root node of the binary tree
            value: The value to search for

        Returns:
            The node with the given value or None if not found
        """
        if not root:
            return None
        q = deque([root])  # Queue for level-order traversal
        while q:
            currentNode = q.popleft()
            if currentNode.data == value:
                return currentNode
            if currentNode.left:
                q.append(currentNode.left)
            if currentNode.right:
                q.append(currentNode.right)
        return None


def lowest_common_ancestor(p, q):
    """
    Find the Lowest Common Ancestor (LCA) of two nodes in a binary tree.

    Args:
        p: The first node
        q: The second node

    Returns:
        The LCA node
    """
    ptr1, ptr2 = p, q

    # Traverse upwards until the two pointers meet
    while ptr1 != ptr2:
        # Move ptr1 to its parent or reset to q
        if ptr1.parent:
            ptr1 = ptr1.parent
        else:
            ptr1 = q

        # Move ptr2 to its parent or reset to p
        if ptr2.parent:
            ptr2 = ptr2.parent
        else:
            ptr2 = p

    return ptr1


# Driver code
def main():
    """
    Main function to demonstrate the LCA algorithm on multiple binary trees.
    """
    input_trees = [
        [100, 50, 200, 25, 75, 350],
        [100, 200, 75, 50, 25, 350],
        [350, 100, 75, 50, 200, 25],
        [100, 50, 200, 25, 75, 350],
        [25, 50, 75, 100, 200, 350],
    ]
    input_nodes = [[25, 75], [50, 350], [100, 200], [50, 25], [350, 200]]

    for i in range(len(input_trees)):
        tree = EduBinaryTree(input_trees[i])
        print((i + 1), ".\tBinary tree:", sep="")
        display_tree(tree.root)
        print("\n\tp = ", input_nodes[i][0])
        print("\tq = ", input_nodes[i][1])

        p = tree.find(tree.root, input_nodes[i][0])
        q = tree.find(tree.root, input_nodes[i][1])

        lca = lowest_common_ancestor(p, q)
        print("\n\tLowest common ancestor:", lca.data)
        print("-" * 100)


if __name__ == "__main__":
    main()
