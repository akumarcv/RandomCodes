from binary_tree import TreeNode, BinaryTree, display_tree


def rob_helper(root):
    if root is None:
        return [0, 0]

    left_money = rob_helper(root.left)
    right_money = rob_helper(root.right)
    include_root_money = root.data + left_money[1] + right_money[1]
    exclude_root_money = max(left_money) + max(right_money)
    return [include_root_money, exclude_root_money]


def rob(root):
    """
    Function to find the maximum amount of money you can rob tonight without alerting the police.
    The houses are arranged in a binary tree. The root of the tree is the house you start at.
    The houses are represented by the TreeNode objects. The value of each node is the amount of money in each house.
    The constraint is that you cannot rob two adjacent houses. That is, if you rob a house, you cannot rob its parent
    or its children. The function returns the maximum amount of money you can rob without getting caught.
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
