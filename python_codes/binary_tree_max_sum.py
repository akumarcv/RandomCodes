from binary_tree import TreeNode, print_tree, BinaryTree


def max_path_sum_helper(root, max_sum):
    """
    Helper function for finding maximum path sum in a binary tree.

    Args:
        root: Current node in the binary tree
        max_sum: Maximum path sum found so far

    Returns:
        tuple: Updated max_sum and maximum path sum ending at current node
    """
    if not root:
        return max_sum, 0

    # Recursively compute max path sum for left and right subtrees
    max_sum, max_left = max_path_sum_helper(root.left, max_sum)
    max_sum, max_right = max_path_sum_helper(root.right, max_sum)

    # Ignore negative path sums as they won't contribute to maximum path
    max_left = 0 if max_left < 0 else max_left
    max_right = 0 if max_right < 0 else max_right

    # Update max_sum with path through current node (left->root->right)
    max_sum = max(max_sum, max_left + root.data + max_right)

    # Return max_sum and the maximum path sum ending at current node
    # (can only choose one path - either through left child or right child)
    return max_sum, max(max_left + root.data, max_right + root.data)


def max_path_sum(root):
    """
    Find the maximum path sum in a binary tree.
    A path is defined as any sequence of nodes from some starting node to any node
    in the tree along the parent-child connections. The path must contain at least
    one node and does not need to go through the root.

    Args:
        root: Root node of the binary tree

    Returns:
        int: Maximum path sum in the tree
    """
    max_sum = float("-inf")
    max_sum, _ = max_path_sum_helper(root, max_sum)
    return max_sum


# Driver code
if __name__ == "__main__":
    # Test Case 1: Simple tree with positive values
    # Tree:      10
    #          /    \
    #         2      10
    #        / \    /
    #       20  1  -25
    #               / \
    #              3   4
    root1 = TreeNode(10)
    root1.left = TreeNode(2)
    root1.right = TreeNode(10)
    root1.left.left = TreeNode(20)
    root1.left.right = TreeNode(1)
    root1.right.left = TreeNode(-25)
    root1.right.left.left = TreeNode(3)
    root1.right.left.right = TreeNode(4)

    print("Test Case 1:")
    print("Tree:")
    print_tree(root1)
    print(f"Maximum path sum: {max_path_sum(root1)}")  # Expected: 42 (20->2->10->10)
    print()

    # Test Case 2: Tree with negative values
    # Tree:      -10
    #          /     \
    #         9       20
    #               /    \
    #              15     7
    root2 = TreeNode(-10)
    root2.left = TreeNode(9)
    root2.right = TreeNode(20)
    root2.right.left = TreeNode(15)
    root2.right.right = TreeNode(7)

    print("Test Case 2:")
    print("Tree:")
    print_tree(root2)
    print(f"Maximum path sum: {max_path_sum(root2)}")  # Expected: 42 (15->20->7)
    print()

    # Test Case 3: Single node tree
    root3 = TreeNode(5)

    print("Test Case 3:")
    print("Tree:")
    print_tree(root3)
    print(f"Maximum path sum: {max_path_sum(root3)}")  # Expected: 5
    print()

    # Test Case 4: Tree with all negative values
    root4 = TreeNode(-10)
    root4.left = TreeNode(-5)
    root4.right = TreeNode(-3)

    print("Test Case 4:")
    print("Tree:")
    print_tree(root4)
    print(f"Maximum path sum: {max_path_sum(root4)}")  # Expected: -3
    print()
