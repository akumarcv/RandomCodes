from binary_tree import TreeNode, BinaryTree


def diameter_helper(root, diameter):
    if root is None:
        return diameter, 0
    diameter, left_height = diameter_helper(root.left, diameter)
    diameter, right_height = diameter_helper(root.right, diameter)

    diameter = max(diameter, left_height + right_height)
    return diameter, max(left_height, right_height) + 1


def diameter_of_binaryTree(root):
    if root is None:
        return 0

    diameter, _ = diameter_helper(root, 0)
    return diameter


# ...existing code...


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
