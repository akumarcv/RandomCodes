class ListNode:
    """
    Node class for doubly linked list used in LRU Cache.
    Stores key-value pairs and maintains prev/next references.
    """
    def __init__(self, k, v):
        """
        Initialize a new node with key and value.
        
        Args:
            k: Key for cache entry
            v: Value to store
        """
        self.k = k          # Key for cache lookup
        self.v = v          # Value stored in cache
        self.prev = None    # Reference to previous node
        self.next = None    # Reference to next node


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation using hashmap and doubly linked list.
    Provides O(1) get and put operations with fixed capacity.
    """
    def __init__(self, capacity: int):
        """
        Initialize LRU cache with given capacity.
        
        Args:
            capacity: Maximum number of key-value pairs cache can hold
        """
        self.cache = {}      # HashMap for O(1) key lookup
        self.capacity = capacity
        self.head = None     # Most recently used item
        self.tail = None     # Least recently used item

    def get(self, key: int) -> int:
        """
        Retrieve value for key and mark as most recently used.
        
        Args:
            key: Key to look up
            
        Returns:
            int: Value if key exists, -1 otherwise
            
        Time Complexity: O(1)
        """
        if key in self.cache:
            node = self.cache[key]
            self.remove(node)    # Remove from current position
            self.add(node)       # Add to front (most recent)
            return node.v
        return -1

    def add(self, node):
        """
        Add node to front of list (most recently used).
        
        Args:
            node: ListNode to add
            
        Time Complexity: O(1)
        """
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node
        if not self.tail:
            self.tail = node
        self.cache[node.k] = node

    def remove(self, node=None):
        """
        Remove node from list and cache.
        If no node specified, removes least recently used item.
        
        Args:
            node: Optional; specific node to remove
            
        Time Complexity: O(1)
        """
        if not node:
            node = self.tail    # Remove LRU item by default
        if not node:
            return
        # Update links for neighbors
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        # Update head/tail if needed
        if node == self.head:
            self.head = node.next
        if node == self.tail:
            self.tail = node.prev
        # Remove from cache
        del self.cache[node.k]

    def put(self, key: int, value: int) -> None:
        """
        Add or update key-value pair in cache.
        Removes least recently used item if cache is at capacity.
        
        Args:
            key: Key to store
            value: Value to store
            
        Time Complexity: O(1)
        """
        if key in self.cache:
            self.remove(self.cache[key])    # Remove existing entry
        elif len(self.cache) == self.capacity:
            self.remove()    # Remove LRU item if at capacity
        # Add new node
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
