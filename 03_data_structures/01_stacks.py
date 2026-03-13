# HOW TO RUN:
#   uv run python 03_data_structures/01_stacks.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- STACKS ---
# A stack is like a pile of plates: you add plates on top, and you
# remove plates from the top. The last thing you put in is the first
# thing you take out. This is called LIFO (Last In, First Out).
#
# Why does this matter?
# - Your browser's "Back" button uses a stack (it goes to the LAST page you visited)
# - The "Undo" feature in any editor uses a stack
# - Python itself uses a stack to keep track of function calls
# - In ML, understanding stacks helps with recursive algorithms and
#   understanding how neural network layers process data


# === CONCEPT 1: Stack using a plain Python list ===
# Python lists already have the two operations a stack needs:
#   - append() to add something on top (called "push")
#   - pop() to remove the top item (called "pop")

print("=" * 50)
print("CONCEPT 1: Stack using a Python list")
print("=" * 50)

stack = []

# Push items onto the stack
stack.append("pancake 1")
stack.append("pancake 2")
stack.append("pancake 3")
print("Stack after pushing 3 pancakes:", stack)
# Output: Stack after pushing 3 pancakes: ['pancake 1', 'pancake 2', 'pancake 3']

# Pop the top item off
top = stack.pop()
print("Popped:", top)
# Output: Popped: pancake 3

print("Stack after popping:", stack)
# Output: Stack after popping: ['pancake 1', 'pancake 2']

# Peek at the top without removing it
# (just look at the last element using index -1)
print("Top of stack (peek):", stack[-1])
# Output: Top of stack (peek): pancake 2

# Check if the stack is empty
print("Is stack empty?", len(stack) == 0)
# Output: Is stack empty? False

print()


# === CONCEPT 2: Building a Stack class ===
# Wrapping the stack in a class gives us cleaner code and prevents
# mistakes (like accidentally inserting in the middle of the list).

print("=" * 50)
print("CONCEPT 2: Stack as a class")
print("=" * 50)

class Stack:
    def __init__(self):
        self._items = []  # underscore means "don't touch this directly"

    def push(self, item):
        """Add an item to the top of the stack."""
        self._items.append(item)

    def pop(self):
        """Remove and return the top item. Raises an error if empty."""
        if self.is_empty():
            raise IndexError("Cannot pop from an empty stack!")
        return self._items.pop()

    def peek(self):
        """Look at the top item without removing it."""
        if self.is_empty():
            raise IndexError("Cannot peek at an empty stack!")
        return self._items[-1]

    def is_empty(self):
        """Check if the stack has no items."""
        return len(self._items) == 0

    def size(self):
        """Return how many items are in the stack."""
        return len(self._items)

    def __str__(self):
        """Show the stack contents when printed."""
        return f"Stack (top -> bottom): {list(reversed(self._items))}"


# Let's use our Stack class
my_stack = Stack()
my_stack.push("A")
my_stack.push("B")
my_stack.push("C")
print(my_stack)
# Output: Stack (top -> bottom): ['C', 'B', 'A']

print("Popped:", my_stack.pop())
# Output: Popped: C

print("Top item:", my_stack.peek())
# Output: Top item: B

print("Size:", my_stack.size())
# Output: Size: 2

print()


# === CONCEPT 3: Real use case - checking balanced brackets ===
# This is a classic problem: given a string like "{[()]}", check
# whether every opening bracket has a matching closing bracket.
# Stacks are perfect for this!

print("=" * 50)
print("CONCEPT 3: Balanced brackets checker")
print("=" * 50)

def is_balanced(text):
    """Check if brackets in the text are properly matched."""
    stack = Stack()
    # Each opening bracket maps to its closing bracket
    matching = {"(": ")", "[": "]", "{": "}"}

    for char in text:
        if char in matching:
            # It's an opening bracket, push it
            stack.push(char)
        elif char in matching.values():
            # It's a closing bracket
            if stack.is_empty():
                return False  # nothing to match with
            top = stack.pop()
            if matching[top] != char:
                return False  # wrong type of bracket

    # If stack is empty, everything matched
    return stack.is_empty()


# Test it out
test_cases = [
    "{[()]}",      # balanced
    "{[(])}",      # NOT balanced - wrong order
    "((()))",      # balanced
    "(()",          # NOT balanced - missing closing
    "",            # balanced (nothing to mismatch)
]

for test in test_cases:
    result = "balanced" if is_balanced(test) else "NOT balanced"
    print(f"  '{test}' -> {result}")
# Output:
#   '{[()]}' -> balanced
#   '{[(])}' -> NOT balanced
#   '(())' -> balanced  (note: actually '((()))' is balanced)
#   '(()' -> NOT balanced
#   '' -> balanced

print()


# === CONCEPT 4: Use case - reversing things ===
# Since a stack reverses order (LIFO), you can use it to reverse
# any sequence.

print("=" * 50)
print("CONCEPT 4: Reversing a string with a stack")
print("=" * 50)

def reverse_string(text):
    """Reverse a string using a stack."""
    stack = Stack()
    for char in text:
        stack.push(char)

    reversed_text = ""
    while not stack.is_empty():
        reversed_text += stack.pop()

    return reversed_text

original = "hello"
reversed_version = reverse_string(original)
print(f"Original: '{original}' -> Reversed: '{reversed_version}'")
# Output: Original: 'hello' -> Reversed: 'olleh'

print()


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test Stack class: push / pop / peek / is_empty / size
_s = Stack()
assert _s.is_empty() == True
assert _s.size() == 0
_s.push(1)
_s.push(2)
_s.push(3)
assert _s.size() == 3
assert _s.peek() == 3          # top is 3
assert _s.pop() == 3           # pop returns 3
assert _s.peek() == 2          # top is now 2
assert _s.size() == 2
assert _s.is_empty() == False

# Test is_balanced()
assert is_balanced("{[()]}") == True
assert is_balanced("{[(])}") == False
assert is_balanced("((()))") == True
assert is_balanced("(()") == False
assert is_balanced("") == True

# Test reverse_string()
assert reverse_string("hello") == "olleh"
assert reverse_string("a") == "a"
assert reverse_string("") == ""
assert reverse_string("abcd") == "dcba"

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Undo feature
#    Create a Stack-based text editor that supports:
#    - type_text(text): adds text to the current document
#    - undo(): removes the last typed text
#    - show(): prints the current document
#    Hint: push each typed text onto a stack, pop to undo.

# Your code here:


# 2. Reverse a list
#    Write a function reverse_list(items) that uses a Stack to
#    reverse a list. Test it with [1, 2, 3, 4, 5].
#    Expected output: [5, 4, 3, 2, 1]
#    Hint: push all items, then pop them into a new list.

# Your code here:


# 3. Min Stack (challenge!)
#    Create a MinStack class that works like a regular Stack but
#    also has a get_min() method that returns the smallest item
#    in the stack in O(1) time (without searching through everything).
#    Hint: use a second stack to keep track of minimums.

# Your code here:
