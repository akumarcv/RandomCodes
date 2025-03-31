# Definition for a binary tree node.

from binary_tree import BinaryTree, TreeNode


class Solution:
    """
    Solution class for finding lowest common ancestor (LCA) in a binary tree.
    Uses recursive approach to traverse tree and track ancestor relationships.
    """

    def __init__(self):
        """Initialize solution with None LCA."""
        self.lca = None  # Store the LCA once found

    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        """
        Find lowest common ancestor of two nodes in binary tree.

        Args:
            root: Root node of binary tree
            p: First target node
            q: Second target node

        Returns:
            TreeNode: Lowest common ancestor of nodes p and q

        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree for recursion stack
        """
        self.lowest_common_ancestor_helper(root, p, q)
        return self.lca

    def lowest_common_ancestor_helper(self, root, p, q):
        """
        Helper function to recursively find LCA using post-order traversal.

        Args:
            root: Current node being processed
            p: First target node
            q: Second target node

        Returns:
            bool: True if either p or q is found in current subtree
        """
        if not root:
            return False

        # Track if current node matches targets or found in subtrees
        left, right, mid = False, False, False

        # Check if current node is one of target nodes
        if root == p or root == q:
            mid = True

        # Search left subtree
        left = self.lowest_common_ancestor_helper(root.left, p, q)

        # Only search right if LCA not found yet
        if not self.lca:
            right = self.lowest_common_ancestor_helper(root.right, p, q)

        # If we found both targets (any combination of mid/left/right)
        if mid + right + left >= 2:
            self.lca = root

        # Return True if target found in current subtree
        return mid or left or right


def test_lowest_common_ancestor():
    """
    Test cases for lowest common ancestor in binary tree
    Creates different tree structures and verifies LCA calculation
    """
    # Test Case 1: Basic tree
    #       3
    #      / \
    #     5   1
    #    / \
    #   6   2
    tree1 = TreeNode(3)
    tree1.left = TreeNode(5)
    tree1.right = TreeNode(1)
    tree1.left.left = TreeNode(6)
    tree1.left.right = TreeNode(2)

    # Test Case 2: Linear tree
    #     1
    #    /
    #   2
    #  /
    # 3
    tree2 = TreeNode(1)
    tree2.left = TreeNode(2)
    tree2.left.left = TreeNode(3)

    # Test Case 3: Complete binary tree
    #       1
    #      / \
    #     2   3
    #    / \ / \
    #   4  5 6  7
    tree3 = TreeNode(1)
    tree3.left = TreeNode(2)
    tree3.right = TreeNode(3)
    tree3.left.left = TreeNode(4)
    tree3.left.right = TreeNode(5)
    tree3.right.left = TreeNode(6)
    tree3.right.right = TreeNode(7)

    test_cases = [
        (tree1, tree1.left, tree1.right, tree1, "Basic tree, LCA of 5 and 1 is 3"),
        (
            tree1,
            tree1.left.left,
            tree1.left.right,
            tree1.left,
            "Basic tree, LCA of 6 and 2 is 5",
        ),
        (
            tree2,
            tree2.left,
            tree2.left.left,
            tree2.left,
            "Linear tree, LCA of 2 and 3 is 2",
        ),
        (
            tree3,
            tree3.left.left,
            tree3.right.right,
            tree3,
            "Complete tree, LCA of 4 and 7 is 1",
        ),
        (
            tree3,
            tree3.left.left,
            tree3.left.right,
            tree3.left,
            "Complete tree, LCA of 4 and 5 is 2",
        ),
    ]

    solution = Solution()
    for i, (root, p, q, expected_lca, description) in enumerate(test_cases, 1):
        # Reset LCA for new test case
        solution.lca = None
        result = solution.lowestCommonAncestor(root, p, q)

        print(f"\nTest Case {i}: {description}")
        print(f"Node p value: {p.data}")
        print(f"Node q value: {q.data}")
        print(f"Expected LCA value: {expected_lca.data}")
        print(f"Got LCA value: {result.data}")

        assert (
            result == expected_lca
        ), f"Test case {i} failed! Expected {expected_lca.val}, got {result.val}"
        print("✓ Passed")


if __name__ == "__main__":
    test_lowest_common_ancestor()
    print("\nAll test cases passed!")
