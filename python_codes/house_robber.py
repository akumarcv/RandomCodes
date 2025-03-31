from binary_tree import TreeNode, BinaryTree, display_tree


def rob_helper(root: TreeNode) -> list[int]:
    """
    Helper function to calculate maximum money that can be robbed from tree.
    Uses post-order traversal to process subtrees.

    Args:
        root: Current node being processed

    Returns:
        list[int]: [include_root_money, exclude_root_money] where:
            - include_root_money: Maximum money if we rob current house
            - exclude_root_money: Maximum money if we skip current house

    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(h) where h is height of tree for recursion stack
    """
    if root is None:
        return [0, 0]

    # Process left and right subtrees
    left_money = rob_helper(root.left)  # [include_left, exclude_left]
    right_money = rob_helper(root.right)  # [include_right, exclude_right]

    # If we include root, must exclude its children
    include_root_money = root.data + left_money[1] + right_money[1]

    # If we exclude root, can take maximum from each subtree
    exclude_root_money = max(left_money) + max(right_money)

    return [include_root_money, exclude_root_money]


def rob(root: TreeNode) -> int:
    """
    Find maximum amount of money that can be robbed from houses arranged in binary tree.
    Houses cannot be robbed if they are adjacent (parent-child relationship).

    Args:
        root: Root node of binary tree where each node value represents money in house

    Returns:
        int: Maximum amount that can be robbed without alerting police

    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(h) where h is height of tree

    Example:
        >>> tree = BinaryTree([TreeNode(3), TreeNode(2), TreeNode(3), None, TreeNode(3), None, TreeNode(1)])
        >>> rob(tree.root)
        7  # Rob houses with values 3 + 3 + 1
    """
    return max(rob_helper(root))


if __name__ == "__main__":

    # Create a list of list of TreeNode objects to represent binary trees
    list_of_trees = [
        [TreeNode(10), TreeNode(9), TreeNode(20), TreeNode(15), TreeNode(7)],
        [TreeNode(7), TreeNode(9), TreeNode(10), TreeNode(15), TreeNode(20)],
        [
            TreeNode(8),
            TreeNode(2),
            TreeNode(17),
            TreeNode(1),
            TreeNode(4),
            TreeNode(19),
            TreeNode(5),
        ],
        [TreeNode(7), TreeNode(3), TreeNode(4), TreeNode(1), TreeNode(3)],
        [TreeNode(9), TreeNode(5), TreeNode(7), TreeNode(1), TreeNode(3)],
        [
            TreeNode(9),
            TreeNode(7),
            None,
            None,
            TreeNode(1),
            TreeNode(8),
            TreeNode(10),
            None,
            TreeNode(12),
        ],
    ]

    # Create the binary trees using the BinaryTree class
    input_trees = []
    for list_of_nodes in list_of_trees:
        tree = BinaryTree(list_of_nodes)
        input_trees.append(tree)

    # Print the input trees
    x = 1
    for tree in input_trees:
        print(x, ".\tInput Tree:", sep="")
        display_tree(tree.root)
        x += 1
        print(
            "\tMaximum amount we can rob without getting caught: ",
            rob(tree.root),
            sep="",
        )
        print("-" * 100)
