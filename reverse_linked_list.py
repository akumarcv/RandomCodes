from linked_list_node import LinkedListNode
from linked_list import LinkedList

def reverse(head):

    if head.next is None:
        return head
    
    prev = None
    curr = head 
    