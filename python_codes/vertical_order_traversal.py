from binary_tree import TreeNode, BinaryTree, display_tree
from collections import deque, defaultdict


def vertical_order(root):
    """
    Performs a vertical order traversal of a binary tree.

    Vertical order traversal visits nodes column-by-column from left to right.
    Nodes in the same column are visited from top to bottom.

    Args:
        root (TreeNode): Root node of the binary tree

    Returns:
        list: List of lists, where each inner list contains node values in a vertical column
              from left to right. Returns None if the tree is empty.
    """
    # Handle empty tree case
    if not root:
        return None

    # Initialize BFS queue and hash map for column-wise node values
    queue = deque()
    levels = defaultdict(list)
    queue.append((0, root))  # Start with root at column 0
    max_left_offset, max_right_offset = 0, 0  # Track boundary columns

    # BFS traversal
    while queue:
        offset, node = queue.popleft()  # Process nodes level by level

        # Add current node to its column
        levels[offset].append(node.data)

        # Process left child (decreasing column index)
        if node.left:
            queue.append((offset - 1, node.left))
            max_left_offset = min(max_left_offset, offset - 1)  # Update leftmost column

        # Process right child (increasing column index)
        if node.right:
            queue.append((offset + 1, node.right))
            max_right_offset = max(
                max_right_offset, offset + 1
            )  # Update rightmost column

    # Collect results from left to right columns
    result = []
    for i in range(max_left_offset, max_right_offset + 1):
        result.append(levels[i])

    return result


# Driver code to demonstrate vertical order traversal
if __name__ == "__main__":
    # Create a sample binary tree
    #       1
    #      / \
    #     2   3
    #    / \   \
    #   4   5   6
    #      / \
    #     7   8

    # Create tree with the nodes parameter
    # Using array representation where index formula is:
    # left_child = 2*i + 1, right_child = 2*i + 2
    nodes = [1, 2, 3, 4, 5, None, 6, None, None, 7, 8]
    try:
        # Try creating the tree using the provided BinaryTree constructor
        tree = BinaryTree(nodes)
    except (TypeError, AttributeError):
        # If the constructor doesn't work as expected, manually create the tree
        tree = BinaryTree([])  # Initialize with empty list if required
        tree.root = TreeNode(1)
        tree.root.left = TreeNode(2)
        tree.root.right = TreeNode(3)
        tree.root.left.left = TreeNode(4)
        tree.root.left.right = TreeNode(5)
        tree.root.right.right = TreeNode(6)
        tree.root.left.right.left = TreeNode(7)
        tree.root.left.right.right = TreeNode(8)

    print("Binary Tree Structure:")
    try:
        display_tree(tree.root)
    except NameError:
        print("Tree display function not available")

    print("\nVertical Order Traversal:")
    result = vertical_order(tree.root)

    # Store the left_offset to use in column numbering
    left_offset = abs(
        min(
            0,
            tree.root.left and -1 or 0,
            tree.root.left and tree.root.left.left and -2 or 0,
        )
    )

    for i, column in enumerate(result):
        print(f"Column {i - left_offset}: {column}")

    # Expected output:
    # Column -2: [4]
    # Column -1: [2, 7]
    # Column 0: [1, 5]
    # Column 1: [3, 8]
    # Column 2: [6]
