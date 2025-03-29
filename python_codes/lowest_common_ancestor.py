# Definition for a binary tree node.

from binary_tree import BinaryTree, TreeNode


class Solution:
    def __init__(self):
        self.lca = None

    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        self.lowest_common_ancestor_helper(root, p, q)

        return self.lca

    def lowest_common_ancestor_helper(self, root, p, q):
        if not root:
            return False

        left, right, mid = False, False, False
        if root == p or root == q:
            mid = True
        left = self.lowest_common_ancestor_helper(root.left, p, q)

        if not self.lca:
            right = self.lowest_common_ancestor_helper(root.right, p, q)

        if mid + right + left >= 2:
            self.lca = root

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
