from typing import List, Optional
from queue import Queue


def display_tree(root: "TreeNode", space: int = 0, level_space: int = 3) -> None:
    """
    Display a binary tree with proper vertical alignment between parents and children.

    Args:
        root: Root node of the tree to display
        space: Not used in this implementation but kept for API compatibility
        level_space: Controls the spacing between nodes

    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(n) for the grid representation
    """
    if root is None:
        print("Empty tree")
        return

    # Calculate height of the tree
    def get_height(node):
        if not node:
            return 0
        return max(get_height(node.left), get_height(node.right)) + 1

    height = get_height(root)
    width = 2**height - 1

    # Initialize the grid with spaces
    grid = [[" " for _ in range(width * level_space)] for _ in range(height)]

    # Fill the grid with tree values
    def fill_grid(node, h, left, right):
        if not node:
            return

        # Calculate middle position for current node
        mid = (left + right) // 2

        # Place the node value in the grid
        node_str = str(node.data)
        start_pos = mid - len(node_str) // 2
        for i, char in enumerate(node_str):
            grid[h][start_pos + i] = char

        # Process children
        fill_grid(node.left, h + 1, left, mid - 1)
        fill_grid(node.right, h + 1, mid + 1, right)

    fill_grid(root, 0, 0, width - 1)

    # Print the grid
    for row in grid:
        print("".join(row))


def print_tree(root, level=0, prefix="Root: "):
    """
    Print the binary tree in a readable hierarchical format.

    Args:
        root: The root node of the tree or subtree
        level: Current level in the tree (for indentation)
        prefix: String to print before the node value
    """
    if root is None:
        return

    indent = "    " * level
    print(f"{indent}{prefix}{root.data}")

    if root.left or root.right:
        if root.left:
            print_tree(root.left, level + 1, "L── ")
        else:
            print(f"{indent}    L── None")

        if root.right:
            print_tree(root.right, level + 1, "R── ")
        else:
            print(f"{indent}    R── None")


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

    def display(self):
        """
        Display the binary tree in a level-by-level format.
        """
        display_tree(self.root)
