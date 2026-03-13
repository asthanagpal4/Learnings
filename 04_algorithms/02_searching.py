# HOW TO RUN:
#   uv run python 04_algorithms/02_searching.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- SEARCHING ALGORITHMS ---
# Searching means finding a specific item in a collection of data.
# In ML, you constantly search through data — finding specific rows,
# looking up values, finding nearest neighbors, etc.
#
# We'll learn two searching methods:
# 1. Linear search — check every item one by one (works on any list)
# 2. Binary search — jump to the middle and narrow down (needs a sorted list)

print("=" * 60)
print("SECTION 4.2 — SEARCHING ALGORITHMS")
print("=" * 60)


# === CONCEPT 1: LINEAR SEARCH ===
# The simplest way to find something: look at every item, one by one.
# Like looking for your keys by checking every pocket, every drawer...
#
# Time complexity: O(n) — if you have n items, you might check all of them.

print("\n=== LINEAR SEARCH ===\n")

def linear_search(items, target):
    """
    Search for target in items. Return the index if found, -1 if not found.
    """
    for i in range(len(items)):
        if items[i] == target:
            return i  # Found it! Return the position.
    return -1  # Checked everything, not found.

# Try it
numbers = [4, 2, 7, 1, 9, 3, 8, 5]

result = linear_search(numbers, 9)
print(f"Searching for 9 in {numbers}")
print(f"Found at index: {result}")
# Output: Found at index: 4

result = linear_search(numbers, 6)
print(f"\nSearching for 6 in {numbers}")
print(f"Found at index: {result}")
# Output: Found at index: -1  (not in the list)

# Linear search works on ANY list — sorted or not, any data type
names = ["Astha", "Bob", "Cara", "Dev", "Eve"]
result = linear_search(names, "Cara")
print(f"\nSearching for 'Cara' in {names}")
print(f"Found at index: {result}")
# Output: Found at index: 2

# Python has built-in ways to search:
print(f"\nUsing 'in' operator:  {'Cara' in names}")    # True
print(f"Using .index():       {names.index('Cara')}")  # 2
# Output:
# Using 'in' operator:  True
# Using .index():       2


# === CONCEPT 2: BINARY SEARCH ===
# Binary search is MUCH faster, but it only works on SORTED lists.
# Idea: Look at the middle item.
#   - If it's what you want, done!
#   - If your target is smaller, search the left half.
#   - If your target is bigger, search the right half.
# Each step cuts the remaining items in half.
#
# Time complexity: O(log n) — incredibly fast even for huge lists.
# A list of 1,000,000 items? Binary search checks at most ~20 items!

print("\n=== BINARY SEARCH ===\n")

def binary_search(sorted_items, target):
    """
    Search for target in a SORTED list.
    Return the index if found, -1 if not found.
    """
    low = 0
    high = len(sorted_items) - 1

    steps = 0  # Let's count how many steps it takes

    while low <= high:
        steps += 1
        mid = (low + high) // 2  # Find the middle position

        if sorted_items[mid] == target:
            print(f"  Found in {steps} steps!")
            return mid
        elif sorted_items[mid] < target:
            # Target is bigger — search the right half
            low = mid + 1
        else:
            # Target is smaller — search the left half
            high = mid - 1

    print(f"  Not found after {steps} steps.")
    return -1

# The list MUST be sorted for binary search to work
sorted_nums = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
print(f"Sorted list: {sorted_nums}")

print(f"\nSearching for 23:")
result = binary_search(sorted_nums, 23)
print(f"Found at index: {result}")
# Output:
#   Found in 2 steps!
# Found at index: 5

print(f"\nSearching for 72:")
result = binary_search(sorted_nums, 72)
print(f"Found at index: {result}")
# Output:
#   Found in 3 steps!
# Found at index: 8

print(f"\nSearching for 50:")
result = binary_search(sorted_nums, 50)
print(f"Found at index: {result}")
# Output:
#   Not found after 4 steps.
# Found at index: -1

# Let's trace through an example to see how it works:
print("\n--- Tracing binary search for 23 in [2, 5, 8, 12, 16, 23, 38, 56, 72, 91] ---")
print("Step 1: low=0, high=9, mid=4 -> list[4]=16 < 23 -> search right half")
print("Step 2: low=5, high=9, mid=7 -> list[7]=56 > 23 -> search left half")
print("Step 3: low=5, high=6, mid=5 -> list[5]=23 = 23 -> FOUND!")


# === CONCEPT 3: COMPARING LINEAR vs BINARY SEARCH ===
# Let's see the speed difference with a big list.

print("\n=== SPEED COMPARISON ===\n")

import time
import random

# Create a big sorted list
big_list = list(range(100000))  # [0, 1, 2, ..., 99999]
target = 99997  # Near the end — worst case for linear search

# Time linear search
start = time.time()
for _ in range(100):  # Run 100 times to get measurable time
    linear_search(big_list, target)
linear_time = time.time() - start

# Time binary search (suppressing the print statements)
def binary_search_quiet(sorted_items, target):
    """Binary search without print statements."""
    low = 0
    high = len(sorted_items) - 1
    while low <= high:
        mid = (low + high) // 2
        if sorted_items[mid] == target:
            return mid
        elif sorted_items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

start = time.time()
for _ in range(100):
    binary_search_quiet(big_list, target)
binary_time = time.time() - start

print(f"Searching for {target} in a list of {len(big_list)} items (100 runs):")
print(f"  Linear search: {linear_time:.4f} seconds")
print(f"  Binary search: {binary_time:.6f} seconds")
print(f"  Binary search is roughly {linear_time / max(binary_time, 0.000001):.0f}x faster!")

print("\nWhy? Linear search checks up to 100,000 items.")
print("Binary search checks at most 17 items (log2 of 100,000 is about 17).")


# === CONCEPT 4: WHEN TO USE WHICH ===

print("\n=== WHEN TO USE WHICH ===\n")

print("""
+------------------+---------------------+---------------------+
| Situation        | Use Linear Search   | Use Binary Search   |
+------------------+---------------------+---------------------+
| List is sorted?  | Works either way    | MUST be sorted      |
| List is small    | Fine                | Not worth it        |
| List is large    | Slow (O(n))         | Fast (O(log n))     |
| Search once      | Fine                | Sort first, then    |
|                  |                     | search (if worth it)|
| Search many      | Slow each time      | Sort once, search   |
| times            |                     | many times fast     |
+------------------+---------------------+---------------------+

In Python, you'll often use:
  - 'in' operator for simple checks: if x in my_list
  - dict/set lookup for O(1) speed: if x in my_set
  - bisect module for binary search on sorted lists
""")

# Python's bisect module for binary search
import bisect

sorted_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Find where 55 would be inserted to keep the list sorted
position = bisect.bisect_left(sorted_data, 55)
print(f"55 would go at index {position} in {sorted_data}")
# Output: 55 would go at index 5

# Check if a value exists using bisect
def bisect_search(sorted_list, target):
    """Use bisect module to check if target is in sorted_list."""
    pos = bisect.bisect_left(sorted_list, target)
    if pos < len(sorted_list) and sorted_list[pos] == target:
        return pos
    return -1

print(f"Is 40 in the list? Index: {bisect_search(sorted_data, 40)}")
print(f"Is 55 in the list? Index: {bisect_search(sorted_data, 55)}")
# Output:
# Is 40 in the list? Index: 3
# Is 55 in the list? Index: -1


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# linear_search correctness
numbers = [4, 2, 7, 1, 9, 3, 8, 5]
assert linear_search(numbers, 9) == 4
assert linear_search(numbers, 4) == 0
assert linear_search(numbers, 5) == 7
assert linear_search(numbers, 6) == -1   # not in list

names = ["Astha", "Bob", "Cara", "Dev", "Eve"]
assert linear_search(names, "Cara") == 2
assert linear_search(names, "Astha") == 0
assert linear_search(names, "Zara") == -1

# binary_search_quiet correctness (using the version without print statements)
sorted_nums = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
assert binary_search_quiet(sorted_nums, 23) == 5
assert binary_search_quiet(sorted_nums, 72) == 8
assert binary_search_quiet(sorted_nums, 2) == 0
assert binary_search_quiet(sorted_nums, 91) == 9
assert binary_search_quiet(sorted_nums, 50) == -1  # not in list
assert binary_search_quiet(sorted_nums, 1) == -1   # smaller than all

# bisect_search correctness
sorted_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
assert bisect_search(sorted_data, 40) == 3
assert bisect_search(sorted_data, 10) == 0
assert bisect_search(sorted_data, 100) == 9
assert bisect_search(sorted_data, 55) == -1  # not in list

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)

# 1. Write a function `find_all` that returns ALL indices where a value appears.
#    Example: find_all([1, 3, 5, 3, 7, 3], 3) -> [1, 3, 5]
#    Hint: Use a list to collect all matching indices.

# Your code here:


# 2. Write a function that finds the MINIMUM value in a list using linear search.
#    Don't use min() — write it yourself!
#    Hint: Keep track of the smallest value seen so far.
#    Test with [8, 3, 9, 1, 5] -> should return 1

# Your code here:


# 3. Modify binary_search to return how many steps it took (not just the index).
#    Return a tuple: (index, steps)
#    Test: binary_search_counted([1,2,3,4,5,6,7,8,9,10], 7) -> (6, some_number)
#    Hint: Add a counter variable, increment it each loop iteration.

# Your code here:


# 4. You have a sorted list of student scores and want to find how many students
#    scored above 75:
#    scores = [45, 52, 58, 63, 67, 72, 78, 82, 85, 91, 95, 99]
#    Hint: Use bisect.bisect_right(scores, 75) to find where 75 would go,
#    then subtract from len(scores).
#    Expected answer: 6 students scored above 75

# Your code here:


print("\nDone! Try the exercises above by writing code and re-running this file.")
