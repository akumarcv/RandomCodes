# Definition for a binary tree node
# class TreeNode:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None

from binary_tree import TreeNode, BinaryTree, display_tree
from collections import deque, defaultdict


def level_order_traversal(root):
    """
    Performs a modified level order traversal on a binary tree.

    This implementation uses a stack-like approach (using pop() instead of popleft())
    which processes nodes in a right-to-left manner within each level.
    The result is organized by level number.

    Args:
        root (TreeNode): The root node of the binary tree to traverse

    Returns:
        str: A string representation of the traversal with format:
             "level1_values : level2_values : ... : levelN_values"
             Returns "None" if the tree is empty

    Time Complexity: O(n) where n is the number of nodes in the tree
    Space Complexity: O(n) for storing all nodes in the queue and dictionary
    """

    if root is None:
        return "None"

    # Initialize queue with the root node and its level (starting at 1)
    queue = deque()
    queue.append((1, root))

    # Dictionary to store nodes at each level
    levels = defaultdict(list)

    while queue:
        # Pop from right side (stack-like behavior)
        level, node = queue.pop()

        # Add current node's value to its level in the dictionary
        levels[level].append(str(node.data))

        # Add children to queue with their level numbers
        # Note: Right child is added before left child so that
        # when using pop(), left child is processed first
        if node.right:
            queue.append((level + 1, node.right))
        if node.left:
            queue.append((level + 1, node.left))

    # Construct the output string from the collected levels
    output_string = ""
    for k, v in levels.items():
        # Join values at each level with commas and append level separator
        output_string = output_string + ", ".join(v) + " : "

    # Remove the trailing separator
    return output_string[:-3]


def main():
    """
    Main function to test the level_order_traversal implementation with various binary trees.

    Creates multiple test cases with different tree structures and displays both
    the tree visualization and its level order traversal output.
    """
    test_cases_roots = []

    # Test Case 1: Balanced BST
    input1 = [
        TreeNode(100),  # Root
        TreeNode(50),  # Left child of root
        TreeNode(200),  # Right child of root
        TreeNode(25),  # Left child of 50
        TreeNode(75),  # Right child of 50
        TreeNode(350),  # Right child of 200
    ]
    tree1 = BinaryTree(input1)
    test_cases_roots.append(tree1.root)

    # Test Case 2: Unbalanced tree with left emphasis
    input2 = [
        TreeNode(25),  # Root
        TreeNode(50),  # Left child of root
        None,  # Right child of root (empty)
        TreeNode(100),  # Left child of 50
        TreeNode(200),  # Right child of 50
        TreeNode(350),  # Left child of None (will be ignored)
    ]
    tree2 = BinaryTree(input2)
    test_cases_roots.append(tree2.root)

    # Test Case 3: Unbalanced tree with right emphasis
    input3 = [
        TreeNode(350),  # Root
        None,  # Left child of root (empty)
        TreeNode(100),  # Right child of root
        None,  # Left child of None (will be ignored)
        TreeNode(50),  # Left child of 100
        TreeNode(25),  # Right child of 100
    ]
    tree3 = BinaryTree(input3)
    test_cases_roots.append(tree3.root)

    # Test Case 4: Single node tree
    tree4 = BinaryTree([TreeNode(100)])
    test_cases_roots.append(tree4.root)

    # Test Case 5: Empty tree
    test_cases_roots.append(None)

    # Display all test cases and their level order traversals
    for i in range(len(test_cases_roots)):
        if i > 0:
            print()
        print(i + 1, ".\tBinary Tree")
        display_tree(test_cases_roots[i])
        print("\n\tLevel order traversal: ")
        print("\t", level_order_traversal(test_cases_roots[i]))
        print("\n" + "-" * 100)


if __name__ == "__main__":
    main()
