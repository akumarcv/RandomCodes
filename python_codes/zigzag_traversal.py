from collections import deque, defaultdict


class TreeNode:
    """
    A binary tree node with data and pointers to left and right children.
    """

    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


def zigzag_level_order(root):
    """
    Performs a zigzag level order traversal of a binary tree.

    In a zigzag traversal, nodes at even levels are visited left-to-right,
    while nodes at odd levels are visited right-to-left.

    Args:
        root: The root node of the binary tree.

    Returns:
        A list of lists, where each inner list contains the node values
        at that level in zigzag order. Empty list if tree is empty.

    Time Complexity: O(n) where n is the number of nodes in the tree
    Space Complexity: O(n) for storing all nodes
    """

    if root is None:
        return []

    # Initialize queue with the root node and its level (starting at 0)
    queue = deque()
    queue.append((0, root))

    # Dictionary to store nodes at each level
    levels = defaultdict(list)

    while queue:
        # Pop from right side (stack-like behavior)
        level, node = queue.pop()

        # Add current node's value to its level in the dictionary
        levels[level].append(node.data)

        # Add children to queue with their level numbers
        # Note: Right child is added before left child so that
        # when using pop(), left child is processed first (DFS approach)
        if node.right:
            queue.append((level + 1, node.right))
        if node.left:
            queue.append((level + 1, node.left))

    # Construct the zigzag pattern from the collected levels
    output = []
    # Sort by level to ensure correct order
    for k in sorted(levels.keys()):
        # Even levels left-to-right, odd levels right-to-left
        if k % 2 == 0:
            output.append(levels[k])
        else:
            output.append(levels[k][::-1])
    return output


# Driver code
if __name__ == "__main__":
    # Create a sample tree
    #       1
    #     /   \
    #    2     3
    #   / \   / \
    #  4   5 6   7

    # Construct the example tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(6)
    root.right.right = TreeNode(7)

    # Test the zigzag traversal function
    result = zigzag_level_order(root)
    print("Zigzag Level Order Traversal:")
    for level in result:
        print(level)

    # Expected output:
    # [1]           - Level 0: left to right
    # [3, 2]        - Level 1: right to left
    # [4, 5, 6, 7]  - Level 2: left to right

    # Test with None input
    print("\nTest with empty tree:")
    print(zigzag_level_order(None))
