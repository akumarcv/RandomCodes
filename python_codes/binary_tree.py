from typing import List, Optional
from queue import Queue


def display_tree(root: "TreeNode", space: int = 0, level_space: int = 10) -> None:
    """
    Display a binary tree in a 2D format using recursive inorder traversal.

    Args:
        root: Root node of the tree to display
        space: Current space from left margin
        level_space: Space between levels

    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(h) where h is height of tree for recursion stack
    """
    if root is None:
        return

    space += level_space
    display_tree(root.right, space, level_space)  # Process right subtree
    print(" " * (space - level_space) + str(root.data))  # Print current node
    display_tree(root.left, space, level_space)  # Process left subtree


class TreeNode:
    """
    Node class for Binary Tree implementation.

    Attributes:
        data: Value stored in the node
        left: Reference to left child node
        right: Reference to right child node
    """

    def __init__(self, data):
        """Initialize a new tree node with given data."""
        self.data = data
        self.left = None
        self.right = None


class BinaryTree:
    """
    Binary Tree class that creates a tree from a list of nodes.
    Uses level-order traversal (BFS) for tree construction.

    Attributes:
        root: Root node of the binary tree
    """

    def __init__(self, nodes: List[Optional[TreeNode]]):
        """
        Initialize binary tree from list of nodes.

        Args:
            nodes: List of TreeNode objects or None values representing tree structure
        """
        self.root = self.createBinaryTree(nodes)

    def createBinaryTree(self, nodes: List[Optional[TreeNode]]) -> Optional[TreeNode]:
        """
        Create binary tree from list of nodes using level-order traversal.

        Args:
            nodes: List of TreeNode objects or None values

        Returns:
            Root node of created binary tree or None if input is empty

        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(w) where w is maximum width of tree
        """
        if len(nodes) == 0:
            return None

        # Create the root node of the binary tree
        root = TreeNode(nodes[0].data)

        # Create a queue and add the root node to it
        queue = Queue()
        queue.put(root)

        # Start iterating over the list of nodes starting from the second node
        i = 1
        while i < len(nodes):
            # Get the next node from the queue
            curr = queue.get()

            # Process left child
            if nodes[i] is not None:
                curr.left = TreeNode(nodes[i].data)
                queue.put(curr.left)
            i += 1

            # Process right child if there are more nodes
            if i < len(nodes) and nodes[i] is not None:
                curr.right = TreeNode(nodes[i].data)
                queue.put(curr.right)
            i += 1

        return root
