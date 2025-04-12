from binary_tree import *


def mirror_helper(node):
    """
    Helper function that creates a mirrored copy of a binary tree.

    Args:
        node: Current node being processed

    Returns:
        A new mirrored node or None if input is None
    """
    # Base case: if node is None, return None
    if node is None:
        return None

    # Create a new node with the same data
    new_node = TreeNode(node.data)

    # Recursively build the mirrored tree by swapping left and right children
    new_node.left = mirror_helper(
        node.right
    )  # Left child becomes mirror of original right
    new_node.right = mirror_helper(
        node.left
    )  # Right child becomes mirror of original left

    return new_node


def mirror_binary_tree(root):
    """
    Creates a mirrored copy of a binary tree.

    A mirrored copy has the left and right children swapped at each node.

    Args:
        root: Root node of the binary tree

    Returns:
        Root node of the new mirrored binary tree, or None if input is None
    """
    # Handle empty tree case
    if root is None:
        return None

    # Use helper function to create mirrored copy
    new_root = mirror_helper(root)
    return new_root


def print_tree(root, level=0, prefix="Root: "):
    """
    Print the binary tree in a readable hierarchical format.

    Args:
        root: The root node of the tree or subtree
        level: Current level in the tree (for indentation)
        prefix: String to print before the node value
    """
    if root is None:
        return

    indent = "    " * level
    print(f"{indent}{prefix}{root.data}")

    if root.left or root.right:
        if root.left:
            print_tree(root.left, level + 1, "L── ")
        else:
            print(f"{indent}    L── None")

        if root.right:
            print_tree(root.right, level + 1, "R── ")
        else:
            print(f"{indent}    R── None")


# Driver code
if __name__ == "__main__":
    """
    Test cases for the binary tree mirroring functionality.
    
    This driver code:
    1. Creates several test trees
    2. Mirrors each tree
    3. Verifies the mirroring operation works correctly
    4. Prints both original and mirrored trees for visualization
    """
    # Test Case 1: Standard binary tree
    print("-" * 50)
    print("Test Case 1: Standard Binary Tree")
    print("-" * 50)
    
    # Create a sample binary tree
    #        1
    #       / \
    #      2   3
    #     / \   \
    #    4   5   6
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.right = TreeNode(6)

    # Print the original tree
    print("Original Tree:")
    print_tree(root)

    # Mirror the tree
    mirrored_root = mirror_binary_tree(root)

    # Print the mirrored tree
    print("\nMirrored Tree:")
    print_tree(mirrored_root)
    
    # Verify mirroring: Expected structure after mirroring
    #        1
    #       / \
    #      3   2
    #     /   / \
    #    6   5   4
    print("\nVerification:")
    is_correct = (
        mirrored_root.data == 1 and
        mirrored_root.left.data == 3 and
        mirrored_root.right.data == 2 and
        mirrored_root.left.left.data == 6 and
        mirrored_root.right.left.data == 5 and
        mirrored_root.right.right.data == 4
    )
    print(f"Mirror correct: {is_correct}")
    
    # Test Case 2: Single node tree
    print("\n" + "-" * 50)
    print("Test Case 2: Single Node Tree")
    print("-" * 50)
    
    single_node = TreeNode(10)
    print("Original Tree:")
    print_tree(single_node)
    
    mirrored_single = mirror_binary_tree(single_node)
    print("\nMirrored Tree:")
    print_tree(mirrored_single)
    
    print("\nVerification:")
    print(f"Mirror correct: {mirrored_single.data == 10 and mirrored_single.left is None and mirrored_single.right is None}")
    
    # Test Case 3: Empty tree
    print("\n" + "-" * 50)
    print("Test Case 3: Empty Tree")
    print("-" * 50)
    
    empty_tree = None
    print("Original Tree: None")
    
    mirrored_empty = mirror_binary_tree(empty_tree)
    print("Mirrored Tree: None")
    
    print("\nVerification:")
    print(f"Mirror correct: {mirrored_empty is None}")
    
    # Test Case 4: Unbalanced tree
    print("\n" + "-" * 50)
    print("Test Case 4: Unbalanced Tree")
    print("-" * 50)
    
    # Create an unbalanced tree
    #        1
    #       /
    #      2
    #     /
    #    3
    unbalanced = TreeNode(1)
    unbalanced.left = TreeNode(2)
    unbalanced.left.left = TreeNode(3)
    
    print("Original Tree:")
    print_tree(unbalanced)
    
    mirrored_unbalanced = mirror_binary_tree(unbalanced)
    print("\nMirrored Tree:")
    print_tree(mirrored_unbalanced)
    
    # Expected:
    #        1
    #         \
    #          2
    #           \
    #            3
    print("\nVerification:")
    is_correct = (
        mirrored_unbalanced.data == 1 and
        mirrored_unbalanced.right is not None and
        mirrored_unbalanced.right.data == 2 and
        mirrored_unbalanced.right.right is not None and
        mirrored_unbalanced.right.right.data == 3 and
        mirrored_unbalanced.left is None
    )
    print(f"Mirror correct: {is_correct}")
