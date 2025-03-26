class ListNode:
    def __init__(self, k, v):
        self.k = k
        self.v = v
        self.prev = None
        self.next = None


class LRUCache:

    def __init__(self, capacity: int):
        self.cache = {}
        self.capacity = capacity
        self.head = None
        self.tail = None

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self.remove(node)
            self.add(node)
            return node.v
        else:
            return -1

    def add(self, node):
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node
        if not self.tail:
            self.tail = node
        self.cache[node.k] = node

    def remove(self, node=None):
        if not node:
            node = self.tail
        if not node:
            return
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node == self.head:
            self.head = node.next
        if node == self.tail:
            self.tail = node.prev

        del self.cache[node.k]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        elif len(self.cache) == self.capacity:
            self.remove()

        node = ListNode(key, value)
        self.add(node)


# Driver code to test the LRUCache
if __name__ == "__main__":
    lru_cache = LRUCache(2)

    lru_cache.put(1, 1)
    lru_cache.put(2, 2)
    print(lru_cache.get(1))  # returns 1
    lru_cache.put(3, 3)  # evicts key 2
    print(lru_cache.get(2))  # returns -1 (not found)
    lru_cache.put(4, 4)  # evicts key 1
    print(lru_cache.get(1))  # returns -1 (not found)
    print(lru_cache.get(3))  # returns 3
    print(lru_cache.get(4))  # returns 4
