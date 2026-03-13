# HOW TO RUN:
#   uv run python 03_data_structures/03_linked_list.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- LINKED LISTS ---
# A linked list is a chain of boxes (called "nodes"). Each box holds
# two things: a value, and a pointer (arrow) to the next box.
#
# Think of it like a treasure hunt: each clue tells you what's here
# AND where to find the next clue. You can only follow the chain
# from the start -- you can't jump to the middle directly.
#
#   [value | next] -> [value | next] -> [value | next] -> None
#
# Why does this matter?
# - Understanding how data is connected (not just stored side by side)
# - Linked lists are the building block for trees and graphs
# - In ML, computation graphs (how neural networks calculate things)
#   use similar node-and-pointer ideas


# === CONCEPT 1: What is a Node? ===
# A node is just a container that holds a value and a reference
# (pointer) to the next node.

print("=" * 50)
print("CONCEPT 1: Creating Nodes")
print("=" * 50)

class Node:
    def __init__(self, value):
        self.value = value  # the data stored in this node
        self.next = None    # pointer to the next node (None = end)

    def __str__(self):
        return f"Node({self.value})"


# Create some nodes
node1 = Node("A")
node2 = Node("B")
node3 = Node("C")

# Link them together manually
node1.next = node2
node2.next = node3

# Walk through the chain
current = node1
while current is not None:
    next_value = current.next.value if current.next else "None (end)"
    print(f"  {current} -> points to: {next_value}")
    current = current.next
# Output:
#   Node(A) -> points to: B
#   Node(B) -> points to: C
#   Node(C) -> points to: None (end)

print()


# === CONCEPT 2: Building a LinkedList class ===
# Managing nodes by hand is messy. Let's build a class that handles
# all the pointer management for us.

print("=" * 50)
print("CONCEPT 2: LinkedList class")
print("=" * 50)

class LinkedList:
    def __init__(self):
        self.head = None  # the first node in the chain
        self._size = 0

    def append(self, value):
        """Add a new node at the end of the list."""
        new_node = Node(value)
        self._size += 1

        if self.head is None:
            # List is empty, new node becomes the head
            self.head = new_node
            return

        # Walk to the last node
        current = self.head
        while current.next is not None:
            current = current.next

        # Link the last node to our new node
        current.next = new_node

    def prepend(self, value):
        """Add a new node at the beginning of the list."""
        new_node = Node(value)
        new_node.next = self.head  # point new node to old head
        self.head = new_node       # new node is now the head
        self._size += 1

    def display(self):
        """Print the entire list in a readable way."""
        if self.head is None:
            print("  (empty list)")
            return

        parts = []
        current = self.head
        while current is not None:
            parts.append(str(current.value))
            current = current.next
        print("  " + " -> ".join(parts) + " -> None")

    def size(self):
        """Return the number of nodes."""
        return self._size

    def search(self, target):
        """Check if a value exists in the list. Returns True/False."""
        current = self.head
        while current is not None:
            if current.value == target:
                return True
            current = current.next
        return False

    def delete(self, target):
        """Remove the first node that has the given value."""
        if self.head is None:
            return  # nothing to delete

        # Special case: the head is the one to delete
        if self.head.value == target:
            self.head = self.head.next
            self._size -= 1
            return

        # Walk through and find the node BEFORE the target
        current = self.head
        while current.next is not None:
            if current.next.value == target:
                # Skip over the target node
                current.next = current.next.next
                self._size -= 1
                return
            current = current.next

    def to_list(self):
        """Convert the linked list to a regular Python list."""
        result = []
        current = self.head
        while current is not None:
            result.append(current.value)
            current = current.next
        return result


# Let's use it!
my_list = LinkedList()
my_list.append(10)
my_list.append(20)
my_list.append(30)
print("After appending 10, 20, 30:")
my_list.display()
# Output: 10 -> 20 -> 30 -> None

my_list.prepend(5)
print("After prepending 5:")
my_list.display()
# Output: 5 -> 10 -> 20 -> 30 -> None

print(f"Size: {my_list.size()}")
# Output: Size: 4

print(f"Search for 20: {my_list.search(20)}")
# Output: Search for 20: True

print(f"Search for 99: {my_list.search(99)}")
# Output: Search for 99: False

my_list.delete(20)
print("After deleting 20:")
my_list.display()
# Output: 5 -> 10 -> 30 -> None

print(f"As a Python list: {my_list.to_list()}")
# Output: As a Python list: [5, 10, 30]

print()


# === CONCEPT 3: Inserting at a specific position ===

print("=" * 50)
print("CONCEPT 3: Insert at position")
print("=" * 50)

class LinkedListV2(LinkedList):
    """Extended version with insert_at method."""

    def insert_at(self, position, value):
        """Insert a value at a specific position (0-based)."""
        if position < 0 or position > self._size:
            raise IndexError(f"Position {position} is out of range (0 to {self._size})")

        if position == 0:
            self.prepend(value)
            return

        new_node = Node(value)
        current = self.head
        # Walk to the node BEFORE the insertion point
        for _ in range(position - 1):
            current = current.next

        # Insert: new_node points to what current used to point to
        new_node.next = current.next
        current.next = new_node
        self._size += 1


demo = LinkedListV2()
demo.append("A")
demo.append("C")
demo.append("D")
print("Before insert:")
demo.display()
# Output: A -> C -> D -> None

demo.insert_at(1, "B")
print("After inserting 'B' at position 1:")
demo.display()
# Output: A -> B -> C -> D -> None

print()


# === CONCEPT 4: List vs Linked List ===

print("=" * 50)
print("CONCEPT 4: Python list vs Linked List")
print("=" * 50)

print("""
  OPERATION          | Python List  | Linked List
  -------------------|--------------|-------------
  Access by index    | Fast         | Slow (must walk)
  Append to end      | Fast         | Slow (must walk)*
  Insert at start    | Slow         | Fast
  Insert in middle   | Slow         | Fast (if you're already there)
  Search             | Slow         | Slow (must walk)
  Memory             | Compact      | Extra (stores pointers too)

  * Can be made fast by keeping a "tail" pointer.

  When to use a linked list:
  - Lots of insertions/deletions at the beginning or middle
  - You don't need random access (jumping to index 50)

  In practice, Python lists are almost always better for everyday
  work. But linked lists teach you how POINTERS work, which is
  essential for understanding trees and graphs (coming next!).
""")


# === CONCEPT 5: Iterating with __iter__ (bonus) ===
# Let's make our linked list work with Python's for loop!

print("=" * 50)
print("CONCEPT 5: Making LinkedList work with for loops")
print("=" * 50)

class NiceLinkedList(LinkedListV2):
    def __iter__(self):
        """This lets us use 'for item in linked_list'."""
        current = self.head
        while current is not None:
            yield current.value  # yield is like return, but keeps going
            current = current.next

    def __len__(self):
        return self._size


nice = NiceLinkedList()
nice.append("Python")
nice.append("is")
nice.append("fun")

print("Using a for loop on our linked list:")
for word in nice:
    print(f"  {word}")
# Output:
#   Python
#   is
#   fun

print(f"Length: {len(nice)}")
# Output: Length: 3

# We can even convert to a list easily now
print(f"As list: {list(nice)}")
# Output: As list: ['Python', 'is', 'fun']

print()


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test LinkedList: append / prepend / size / search / delete / to_list
_ll = LinkedList()
assert _ll.size() == 0
assert _ll.search(10) == False

_ll.append(10)
_ll.append(20)
_ll.append(30)
assert _ll.size() == 3
assert _ll.to_list() == [10, 20, 30]
assert _ll.search(20) == True
assert _ll.search(99) == False

_ll.prepend(5)
assert _ll.size() == 4
assert _ll.to_list() == [5, 10, 20, 30]

_ll.delete(20)
assert _ll.size() == 3
assert _ll.to_list() == [5, 10, 30]
assert _ll.search(20) == False

# Test delete head
_ll.delete(5)
assert _ll.to_list() == [10, 30]

# Test LinkedListV2: insert_at
_ll2 = LinkedListV2()
_ll2.append("A")
_ll2.append("C")
_ll2.append("D")
_ll2.insert_at(1, "B")
assert _ll2.to_list() == ["A", "B", "C", "D"]

# Test NiceLinkedList: for-loop iteration and len()
_nl = NiceLinkedList()
_nl.append("x")
_nl.append("y")
_nl.append("z")
assert list(_nl) == ["x", "y", "z"]
assert len(_nl) == 3

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Reverse a linked list
#    Add a method reverse() to the LinkedList class that reverses
#    the list in place (without creating a new list).
#    Example: 1 -> 2 -> 3 becomes 3 -> 2 -> 1
#    Hint: use three pointers: previous, current, and next_node.
#    Walk through the list and flip each pointer backwards.

# Your code here:


# 2. Find the middle
#    Add a method find_middle() that returns the middle value.
#    For [1, 2, 3, 4, 5], the middle is 3.
#    For [1, 2, 3, 4], the middle is 2 (first of the two middles).
#    Hint: use two pointers - "slow" moves 1 step, "fast" moves 2.
#    When fast reaches the end, slow is at the middle.

# Your code here:


# 3. Remove duplicates
#    Add a method remove_duplicates() that removes duplicate values.
#    Example: 1 -> 2 -> 2 -> 3 -> 1 becomes 1 -> 2 -> 3
#    Hint: use a set to track values you've already seen.

# Your code here:
