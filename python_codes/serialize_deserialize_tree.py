# Definition of a binary tree node
#
# class TreeNode:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None

import re
from binary_tree import TreeNode, BinaryTree, display_tree


def serialize_helper(root, result):
    """
    Helper function for serialization. Traverses the tree in pre-order and appends node values to the result list.

    Args:
        root (TreeNode): The current node in the binary tree.
        result (list): The list to store serialized node values.
    """
    if not root:
        result.append("None")
        return
    result.append(str(root.data))
    serialize_helper(root.left, result)
    serialize_helper(root.right, result)


def serialize(root):
    """
    Serializes a binary tree into a comma-separated string representation.

    Args:
        root (TreeNode): The root node of the binary tree.

    Returns:
        str: A comma-separated string of node values where 'None' represents null nodes.
    """
    result = []
    serialize_helper(root, result)
    return ",".join(result)


def deserialize_helper(data):
    """
    Helper function for deserialization. Recursively builds the binary tree from a list of node values.

    Args:
        data (list): The list of node values.

    Returns:
        TreeNode: The root node of the deserialized binary tree.
    """
    if not data:
        return None
    value = data.pop(0)
    if value is None:
        return None
    node = TreeNode(value)
    node.left = deserialize_helper(data)
    node.right = deserialize_helper(data)
    return node


def deserialize(stream):
    """
    Deserializes a string representation of a binary tree back into a tree structure.

    Args:
        stream (str): A comma-separated string of node values where 'None' represents null nodes

    Returns:
        TreeNode: The root node of the deserialized binary tree, or None if the stream is empty
    """
    # Return None for empty streams
    if not stream:
        return None

    # Split the input string by commas to get individual node values
    stream = re.split(r",", stream)

    # Convert string values to integers or None as appropriate
    stream = [int(x) if x != "None" else None for x in stream]

    # Call the recursive helper function to build the tree
    node = deserialize_helper(stream)
    return node


# Driver code
def main():
    """
    Main function to demonstrate serialization and deserialization of binary trees.
    """
    global m
    input_trees = [
        [
            TreeNode(100),
            TreeNode(50),
            TreeNode(200),
            TreeNode(25),
            TreeNode(75),
            TreeNode(350),
        ],
        [
            TreeNode(100),
            TreeNode(200),
            TreeNode(75),
            TreeNode(50),
            TreeNode(25),
            TreeNode(350),
        ],
        [
            TreeNode(200),
            TreeNode(350),
            TreeNode(100),
            TreeNode(75),
            TreeNode(50),
            TreeNode(200),
            TreeNode(25),
        ],
        [
            TreeNode(25),
            TreeNode(50),
            TreeNode(75),
            TreeNode(100),
            TreeNode(200),
            TreeNode(350),
        ],
        [
            TreeNode(350),
            TreeNode(75),
            TreeNode(25),
            TreeNode(200),
            TreeNode(50),
            TreeNode(100),
        ],
        [
            TreeNode(1),
            None,
            TreeNode(2),
            None,
            TreeNode(3),
            None,
            TreeNode(4),
            None,
            TreeNode(5),
        ],
    ]

    indx = 1
    for i in input_trees:
        tree = BinaryTree(i)

        print(indx, ".\tBinary Tree:", sep="")
        indx += 1
        if tree.root is None:
            display_tree(None)
        else:
            display_tree(tree.root)

        # Serialization
        ostream = serialize(tree.root)
        print("\n\tSerialized integer list:")
        print("\t" + str(ostream))

        deserialized_root = deserialize(ostream)
        print("\n\tDeserialized binary tree:")
        display_tree(deserialized_root)
        print("-" * 100)


if __name__ == "__main__":
    main()
