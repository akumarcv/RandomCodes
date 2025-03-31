from binary_tree import TreeNode, BinaryTree


def diameter_helper(root: TreeNode, diameter: int) -> tuple[int, int]:
    """
    Helper function to calculate tree diameter recursively.

    For each node, calculates:
    1. Maximum diameter passing through this node
    2. Height of subtree rooted at this node

    Args:
        root: Current node being processed
        diameter: Maximum diameter found so far

    Returns:
        tuple: (max_diameter, height_of_subtree)

    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(h) where h is height of tree for recursion stack
    """
    # Base case: empty subtree
    if root is None:
        return diameter, 0

    # Recursively process left and right subtrees
    diameter, left_height = diameter_helper(root.left, diameter)
    diameter, right_height = diameter_helper(root.right, diameter)

    # Update diameter if path through current node is longer
    diameter = max(diameter, left_height + right_height)

    # Return updated diameter and height of current subtree
    return diameter, max(left_height, right_height) + 1


def diameter_of_binaryTree(root: TreeNode) -> int:
    """
    Calculate diameter (longest path) of binary tree.
    Diameter is longest path between any two nodes, may not pass through root.

    Args:
        root: Root node of binary tree

    Returns:
        int: Length of longest path (number of edges)

    Example:
        >>> tree = TreeNode(1)
        >>> tree.left = TreeNode(2)
        >>> tree.right = TreeNode(3)
        >>> diameter_of_binaryTree(tree)
        2  # Longest path is from left leaf through root to right leaf
    """
    # Handle empty tree case
    if root is None:
        return 0

    # Get diameter using helper function (ignore height)
    diameter, _ = diameter_helper(root, 0)
    return diameter


def test_diameter():
    """
    Test cases for binary tree diameter calculation
    Creates different tree structures and verifies diameter calculation
    """
    # Test Case 1: Simple tree with diameter through root
    #       1
    #      / \
    #     2   3
    tree1 = TreeNode(1)
    tree1.left = TreeNode(2)
    tree1.right = TreeNode(3)

    # Test Case 2: Linear tree (all nodes to left)
    #     1
    #    /
    #   2
    #  /
    # 3
    tree2 = TreeNode(1)
    tree2.left = TreeNode(2)
    tree2.left.left = TreeNode(3)

    # Test Case 3: Complex tree with diameter not through root
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    #  /     \
    # 6       7
    tree3 = TreeNode(1)
    tree3.left = TreeNode(2)
    tree3.right = TreeNode(3)
    tree3.left.left = TreeNode(4)
    tree3.left.right = TreeNode(5)
    tree3.left.left.left = TreeNode(6)
    tree3.left.right.right = TreeNode(7)

    test_cases = [
        (tree1, 2, "Simple tree with diameter through root"),
        (tree2, 2, "Linear tree"),
        (tree3, 5, "Complex tree with diameter not through root"),
        (None, 0, "Empty tree"),
        (TreeNode(1), 0, "Single node tree"),
    ]

    for i, (root, expected, description) in enumerate(test_cases, 1):
        result = diameter_of_binaryTree(root)
        print(f"\nTest Case {i}: {description}")
        print(f"Expected diameter: {expected}")
        print(f"Got diameter: {result}")

        assert (
            result == expected
        ), f"Test case {i} failed! Expected {expected}, got {result}"
        print("✓ Passed")


if __name__ == "__main__":
    test_diameter()
    print("\nAll test cases passed!")
