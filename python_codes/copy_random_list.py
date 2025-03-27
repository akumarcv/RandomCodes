import collections


class Node:
    def __init__(self, x: int, next: "Node" = None, random: "Node" = None):
        self.val = int(x)
        self.next = next
        self.random = random


def copy_list(head):
    if head is None:
        return head

    new_list = None

    visited = {}
    queue = []
    queue.append(head)
    visited[head] = Node(head.val)

    while queue:
        current = queue.pop(0)
        if current.next and current.next not in visited:
            visited[current.next] = Node(current.next.val)
            queue.append(current.next)
        if current.next:
            visited[current].next = visited[current.next]

        if current.random and current.random not in visited:
            visited[current.random] = Node(current.random.val)
            queue.append(current.random)
        if current.random:
            visited[current].random = visited[current.random]

    return visited[head]


def copy_list_twopass(head):
    if head is None:
        return

    hmap = {None: None}

    curr = head

    while curr:
        hmap[curr] = Node(curr.val)
        curr = curr.next

    curr = head
    while curr:
        hmap[curr].next = hmap[curr.next]
        hmap[curr].random = hmap[curr.random]
        curr = curr.next
    return hmap[head]


# ...existing code...


def print_list(head, list_name=""):
    """Helper function to print linked list with random pointers"""
    if not head:
        return "[]"

    result = []
    node_map = {}
    current = head
    index = 0

    # First pass: Map nodes to indices
    while current:
        node_map[current] = index
        current = current.next
        index += 1

    # Second pass: Create readable representation
    current = head
    while current:
        random_index = node_map[current.random] if current.random else None
        result.append(f"[{current.val},{random_index}]")
        current = current.next

    print(f"{list_name}: {' -> '.join(result)}")


# Driver code
if __name__ == "__main__":
    # Test Case 1: [[7,null],[13,0],[11,4],[10,2],[1,0]]
    head1 = Node(7)
    node2 = Node(13)
    node3 = Node(11)
    node4 = Node(10)
    node5 = Node(1)

    head1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5

    head1.random = None
    node2.random = head1
    node3.random = node5
    node4.random = node3
    node5.random = head1

    print("\nTest Case 1:")
    print_list(head1, "Original")
    copied1 = copy_list_twopass(head1)
    print_list(copied1, "Copied")

    # Test Case 2: [[1,1],[2,1]]
    head2 = Node(1)
    node2 = Node(2)

    head2.next = node2
    head2.random = node2
    node2.random = node2

    print("\nTest Case 2:")
    print_list(head2, "Original")
    copied2 = copy_list_twopass(head2)
    print_list(copied2, "Copied")

    # Test Case 3: [[3,null],[3,0],[3,null]]
    head3 = Node(3)
    node2 = Node(3)
    node3 = Node(3)

    head3.next = node2
    node2.next = node3
    head3.random = None
    node2.random = head3
    node3.random = None

    print("\nTest Case 3:")
    print_list(head3, "Original")
    copied3 = copy_list_twopass(head3)
    print_list(copied3, "Copied")
