# HOW TO RUN:
#   uv run python 03_data_structures/02_queues.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- QUEUES ---
# A queue is like a line at a store: the first person who joins the
# line is the first person who gets served. This is called FIFO
# (First In, First Out).
#
# Why does this matter?
# - Handling requests on a web server (first come, first served)
# - Printing documents (print queue)
# - BFS (Breadth-First Search) in graphs uses a queue
# - In ML, data loading pipelines use queues to feed batches of
#   data to the model during training


from collections import deque  # Python's built-in double-ended queue


# === CONCEPT 1: Queue using a list (slow way) ===
# You CAN use a list as a queue, but removing from the front is slow
# because Python has to shift every other element over.

print("=" * 50)
print("CONCEPT 1: Queue using a list (not recommended)")
print("=" * 50)

queue = []

# Add to the back (enqueue)
queue.append("person A")
queue.append("person B")
queue.append("person C")
print("Queue:", queue)
# Output: Queue: ['person A', 'person B', 'person C']

# Remove from the front (dequeue) -- this is the slow part!
first = queue.pop(0)  # pop(0) removes from position 0
print("Served:", first)
# Output: Served: person A

print("Queue after serving:", queue)
# Output: Queue after serving: ['person B', 'person C']

print()


# === CONCEPT 2: Queue using collections.deque (the right way) ===
# deque (pronounced "deck") is designed for fast operations on both
# ends. Adding and removing from either end is fast.

print("=" * 50)
print("CONCEPT 2: Queue using deque (recommended)")
print("=" * 50)

queue = deque()

# Add to the back (enqueue)
queue.append("person A")
queue.append("person B")
queue.append("person C")
print("Queue:", list(queue))
# Output: Queue: ['person A', 'person B', 'person C']

# Remove from the front (dequeue) -- fast with deque!
first = queue.popleft()
print("Served:", first)
# Output: Served: person A

print("Queue after serving:", list(queue))
# Output: Queue after serving: ['person B', 'person C']

# Peek at the front without removing
print("Next in line:", queue[0])
# Output: Next in line: person B

# Check the size
print("People waiting:", len(queue))
# Output: People waiting: 2

print()


# === CONCEPT 3: Building a Queue class ===
# Wrapping deque in a class makes our code cleaner and more readable.

print("=" * 50)
print("CONCEPT 3: Queue as a class")
print("=" * 50)

class Queue:
    def __init__(self):
        self._items = deque()

    def enqueue(self, item):
        """Add an item to the back of the queue."""
        self._items.append(item)

    def dequeue(self):
        """Remove and return the front item."""
        if self.is_empty():
            raise IndexError("Cannot dequeue from an empty queue!")
        return self._items.popleft()

    def peek(self):
        """Look at the front item without removing it."""
        if self.is_empty():
            raise IndexError("Cannot peek at an empty queue!")
        return self._items[0]

    def is_empty(self):
        """Check if the queue has no items."""
        return len(self._items) == 0

    def size(self):
        """Return how many items are in the queue."""
        return len(self._items)

    def __str__(self):
        return f"Queue (front -> back): {list(self._items)}"


# Use it
printer_queue = Queue()
printer_queue.enqueue("report.pdf")
printer_queue.enqueue("photo.jpg")
printer_queue.enqueue("letter.docx")
print(printer_queue)
# Output: Queue (front -> back): ['report.pdf', 'photo.jpg', 'letter.docx']

print("Now printing:", printer_queue.dequeue())
# Output: Now printing: report.pdf

print("Next up:", printer_queue.peek())
# Output: Next up: photo.jpg

print(printer_queue)
# Output: Queue (front -> back): ['photo.jpg', 'letter.docx']

print()


# === CONCEPT 4: Real use case - Hot Potato game ===
# A group of people stand in a circle and pass a "hot potato".
# After a fixed number of passes, whoever is holding it is out.
# This repeats until one person remains.

print("=" * 50)
print("CONCEPT 4: Hot Potato simulation")
print("=" * 50)

def hot_potato(names, num_passes):
    """Simulate the hot potato game using a queue."""
    queue = Queue()
    for name in names:
        queue.enqueue(name)

    while queue.size() > 1:
        # Pass the potato num_passes times
        for _ in range(num_passes):
            # Person at front gets the potato, goes to back
            person = queue.dequeue()
            queue.enqueue(person)

        # Person at front is OUT
        eliminated = queue.dequeue()
        print(f"  {eliminated} is eliminated!")

    winner = queue.dequeue()
    return winner

players = ["Astha", "Raj", "Priya", "Amit", "Sneha"]
winner = hot_potato(players, 3)
print(f"Winner: {winner}")
# Output will vary based on num_passes, but with 3:
#   Amit is eliminated!
#   Astha is eliminated!
#   Sneha is eliminated!
#   Raj is eliminated!
#   Winner: Priya

print()


# === CONCEPT 5: Stack vs Queue - when to use which? ===

print("=" * 50)
print("CONCEPT 5: Stack vs Queue comparison")
print("=" * 50)

print("""
  STACK (LIFO)                    QUEUE (FIFO)
  -------------------------       -------------------------
  Last in, first out              First in, first out
  Like a pile of plates           Like a line at a store
  push / pop                      enqueue / dequeue
  Browser back button             Print queue
  Undo/Redo                       Task scheduling
  Function call tracking          BFS in graphs
  Depth-First Search (DFS)        Data loading in ML
""")


# === CONCEPT 6: Priority Queue (sneak peek) ===
# Sometimes you want a queue where important items jump ahead.
# Python has a built-in module for this.

print("=" * 50)
print("CONCEPT 6: Priority Queue (quick intro)")
print("=" * 50)

import heapq  # Python's priority queue is based on a "heap"

# A priority queue serves the SMALLEST item first
tasks = []
heapq.heappush(tasks, (3, "low priority task"))
heapq.heappush(tasks, (1, "urgent task"))
heapq.heappush(tasks, (2, "medium task"))

print("Processing tasks by priority:")
while tasks:
    priority, task = heapq.heappop(tasks)
    print(f"  Priority {priority}: {task}")
# Output:
#   Priority 1: urgent task
#   Priority 2: medium task
#   Priority 3: low priority task

print()


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test Queue class: enqueue / dequeue / peek / is_empty / size
_q = Queue()
assert _q.is_empty() == True
assert _q.size() == 0
_q.enqueue("first")
_q.enqueue("second")
_q.enqueue("third")
assert _q.size() == 3
assert _q.peek() == "first"        # front is "first"
assert _q.dequeue() == "first"     # removes "first"
assert _q.peek() == "second"       # front is now "second"
assert _q.size() == 2
assert _q.is_empty() == False
_q.dequeue()
_q.dequeue()
assert _q.is_empty() == True

# Test that hot_potato returns a string (the winner)
_winner = hot_potato(["A", "B", "C", "D"], 2)
assert isinstance(_winner, str)
assert _winner in ["A", "B", "C", "D"]

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Task Manager
#    Create a simple task manager using a Queue:
#    - add_task(task): adds a task to the queue
#    - do_next_task(): removes and prints the next task
#    - show_tasks(): prints all waiting tasks
#    Test with 4-5 tasks.
#    Hint: use the Queue class from Concept 3.

# Your code here:


# 2. Recent calls
#    Create a class RecentCalls that keeps track of only the last N
#    phone calls. When a new call comes in and the list is full,
#    the oldest call is automatically removed.
#    Hint: deque has a maxlen parameter: deque(maxlen=5)

# Your code here:


# 3. Interleave a queue (challenge!)
#    Given a queue [1, 2, 3, 4, 5, 6], rearrange it to
#    [1, 4, 2, 5, 3, 6] (interleave the first half with the second).
#    Hint: split into two halves, then alternate between them.

# Your code here:
