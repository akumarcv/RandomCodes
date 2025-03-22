import math
from linked_list import (
    LinkedList,
    reverse_linked_list_k,
    traverse_linked_list,
    print_list_with_forward_arrow,
)
from linked_list_node import LinkedListNode


def reverse_k_groups(head, k):

    dummy = LinkedListNode(0)
    dummy.next = head
    ptr = dummy

    while ptr != None:

        tracker = ptr

        for i in range(k):

            if tracker == None:
                break

            tracker = tracker.next

        if tracker == None:
            break

        previous, current = reverse_linked_list_k(ptr.next, k)

        last_node_of_reversed_group = ptr.next
        last_node_of_reversed_group.next = current
        ptr.next = previous
        ptr = last_node_of_reversed_group

    return dummy.next


def main():
    input_list = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [3, 4, 5, 6, 2, 8, 7, 7],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7],
        [1],
    ]
    k = [3, 2, 1, 7, 1]

    for i in range(len(input_list)):
        input_linked_list = LinkedList()
        input_linked_list.create_linked_list(input_list[i])

        print(i + 1, ".\tLinked list: ", end=" ")
        print_list_with_forward_arrow(input_linked_list.head)
        print("\n")
        print("\tReversed linked list: ", end=" ")
        result = reverse_k_groups(input_linked_list.head, k[i])
        print_list_with_forward_arrow(result)
        print("\n")
        print("-" * 100)


if __name__ == "__main__":
    main()
