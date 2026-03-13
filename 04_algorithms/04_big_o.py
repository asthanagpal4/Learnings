# HOW TO RUN:
#   uv run python 04_algorithms/04_big_o.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- BIG-O NOTATION (ALGORITHM COMPLEXITY) ---
# When you write code, you want to know: "How fast is this?"
# Big-O notation is a way to describe how an algorithm's speed changes
# as the amount of data grows.
#
# Think of it like this:
#   - If you have 10 items and it takes 1 second...
#   - What happens with 100 items? 1,000? 1,000,000?
#
# In ML, you deal with HUGE datasets (millions of rows), so understanding
# Big-O helps you choose the right approach and avoid slow code.

import time
import random

print("=" * 60)
print("SECTION 4.4 — BIG-O NOTATION")
print("=" * 60)


# === CONCEPT 1: O(1) — CONSTANT TIME ===
# No matter how big the data is, it takes the same amount of time.
# Examples: accessing a list by index, dict lookup, getting list length.

print("\n=== O(1) — CONSTANT TIME ===\n")
print("O(1) means: 'Same speed regardless of data size.'\n")

# Accessing an element by index is O(1)
for size in [1000, 10000, 100000, 1000000]:
    big_list = list(range(size))

    start = time.time()
    for _ in range(100000):
        _ = big_list[size // 2]  # Access the middle element
    elapsed = time.time() - start

    print(f"  List size {size:>10,}: Access middle element (100k times) = {elapsed:.4f}s")

print("\n  Notice: The time barely changes! That's O(1).")


# === CONCEPT 2: O(n) — LINEAR TIME ===
# Time grows proportionally to the data size.
# Double the data? Double the time.
# Examples: looping through a list, linear search, summing all items.

print("\n=== O(n) — LINEAR TIME ===\n")
print("O(n) means: 'Time grows proportionally to data size.'\n")

def sum_all(items):
    """Add up all items — must look at each one."""
    total = 0
    for item in items:
        total += item
    return total

for size in [10000, 20000, 40000, 80000]:
    data = list(range(size))

    start = time.time()
    for _ in range(100):
        sum_all(data)
    elapsed = time.time() - start

    print(f"  List size {size:>10,}: Sum all (100 times) = {elapsed:.4f}s")

print("\n  Notice: Double the size, roughly double the time. That's O(n).")


# === CONCEPT 3: O(n^2) — QUADRATIC TIME ===
# Time grows with the SQUARE of the data size.
# Double the data? FOUR times slower.
# Examples: bubble sort, checking all pairs, nested loops.

print("\n=== O(n^2) — QUADRATIC TIME ===\n")
print("O(n^2) means: 'Nested loops over data. Gets slow FAST.'\n")

def has_duplicates_slow(items):
    """Check for duplicates by comparing every pair — O(n^2)."""
    count = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            count += 1
            if items[i] == items[j]:
                return True, count
    return False, count

for size in [1000, 2000, 4000, 8000]:
    data = list(range(size))  # No duplicates, so it checks ALL pairs

    start = time.time()
    _, comparisons = has_duplicates_slow(data)
    elapsed = time.time() - start

    print(f"  List size {size:>6,}: {comparisons:>12,} comparisons, took {elapsed:.4f}s")

print("\n  Notice: Double the size, roughly 4x the comparisons. That's O(n^2).")
print("  A smarter way (using a set) can do this in O(n):")

def has_duplicates_fast(items):
    """Check for duplicates using a set — O(n)."""
    seen = set()
    for item in items:
        if item in seen:
            return True
        seen.add(item)
    return False

start = time.time()
has_duplicates_fast(list(range(8000)))
elapsed = time.time() - start
print(f"  O(n) version on 8,000 items: {elapsed:.4f}s  (much faster!)")


# === CONCEPT 4: O(log n) — LOGARITHMIC TIME ===
# Time grows very slowly even as data gets huge.
# Each step cuts the remaining work in half.
# Examples: binary search, balanced tree lookup.

print("\n=== O(log n) — LOGARITHMIC TIME ===\n")
print("O(log n) means: 'Cuts the problem in half each step. Very fast!'\n")

def binary_search(sorted_list, target):
    """Binary search — O(log n)."""
    low, high = 0, len(sorted_list) - 1
    steps = 0
    while low <= high:
        steps += 1
        mid = (low + high) // 2
        if sorted_list[mid] == target:
            return steps
        elif sorted_list[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return steps

import math

for size in [100, 1000, 10000, 100000, 1000000]:
    data = list(range(size))
    target = size - 3  # Near the end

    steps = binary_search(data, target)
    theoretical = math.ceil(math.log2(size))

    print(f"  List size {size:>10,}: Binary search took {steps:>3} steps "
          f"(log2 = {theoretical})")

print("\n  Notice: 1 million items, but only ~20 steps! That's O(log n).")


# === CONCEPT 5: O(n log n) — LINEARITHMIC TIME ===
# A bit more than O(n) but MUCH less than O(n^2).
# Examples: merge sort, Python's sorted(), most efficient sorting.

print("\n=== O(n log n) — LINEARITHMIC TIME ===\n")
print("O(n log n) means: 'Efficient sorting speed.'\n")

for size in [10000, 20000, 40000, 80000]:
    data = [random.randint(1, 100000) for _ in range(size)]

    start = time.time()
    sorted(data)
    elapsed = time.time() - start

    print(f"  List size {size:>10,}: sorted() took {elapsed:.4f}s")

print("\n  Notice: It grows, but not as fast as O(n^2) would.")


# === CONCEPT 6: THE BIG PICTURE ===

print("\n=== THE BIG PICTURE ===\n")

print("""
Here's how different Big-O complexities compare as data grows:

  n        O(1)    O(log n)   O(n)      O(n log n)   O(n^2)      O(2^n)
  -------- ------- ---------- --------- ------------ ------------ -----------
  10       1       3          10        33           100          1,024
  100      1       7          100       664          10,000       HUGE
  1,000    1       10         1,000     9,966        1,000,000    IMPOSSIBLE
  10,000   1       13         10,000    132,877      100,000,000  IMPOSSIBLE
  100,000  1       17         100,000   1,660,964    WAY TOO BIG  IMPOSSIBLE

Key takeaways:
  - O(1) and O(log n) are FAST — use dicts/sets and binary search
  - O(n) is usually fine — a simple loop through data
  - O(n log n) is the best you can do for general sorting
  - O(n^2) gets slow quickly — watch out for nested loops
  - O(2^n) is basically impossible for large n — avoid at all costs
""")


# === CONCEPT 7: COMMON OPERATIONS AND THEIR BIG-O ===

print("=== COMMON PYTHON OPERATIONS ===\n")

print("""
  Operation                          Big-O      Notes
  ---------------------------------- ---------- ---------------------------
  list[i]  (access by index)         O(1)       Instant
  list.append(x)                     O(1)       Adding to the end is fast
  list.insert(0, x)                  O(n)       Shifting everything over!
  x in list                          O(n)       Checks one by one
  x in set                           O(1)       Hash table lookup — fast!
  x in dict                          O(1)       Hash table lookup — fast!
  dict[key]                          O(1)       Instant
  sorted(list)                       O(n log n) Efficient sorting
  list.pop()  (last item)            O(1)       Fast
  list.pop(0) (first item)           O(n)       Shifts everything!
  for x in list                      O(n)       Touch each item once
  nested for loops                   O(n^2)     Every pair — gets slow!
""")


# === CONCEPT 8: LET'S ACTUALLY MEASURE AND COMPARE ===

print("=== MEASURING: LIST vs SET for 'in' OPERATOR ===\n")

sizes = [1000, 5000, 10000, 50000]

for size in sizes:
    data_list = list(range(size))
    data_set = set(range(size))
    target = size - 1  # Worst case for list (last element)

    # Measure list lookup
    start = time.time()
    for _ in range(1000):
        _ = target in data_list
    list_time = time.time() - start

    # Measure set lookup
    start = time.time()
    for _ in range(1000):
        _ = target in data_set
    set_time = time.time() - start

    print(f"  Size {size:>6,}: list 'in' = {list_time:.4f}s, "
          f"set 'in' = {set_time:.6f}s, "
          f"set is ~{list_time / max(set_time, 0.000001):.0f}x faster")

print("\n  This is why sets and dicts are so important!")
print("  In ML, when you need to check membership, use a set!")


# === CONCEPT 9: PRACTICAL TIPS ===

print("\n=== PRACTICAL TIPS FOR WRITING FAST CODE ===\n")

print("""
1. USE THE RIGHT DATA STRUCTURE
   - Need fast lookups? Use a dict or set (O(1)) instead of a list (O(n))
   - Need sorted data with fast search? Keep it sorted, use binary search

2. AVOID NESTED LOOPS WHEN POSSIBLE
   - Two nested loops = O(n^2). With 10,000 items, that's 100 million operations!
   - Often you can use a dict/set to avoid the inner loop

3. PYTHON'S BUILT-IN FUNCTIONS ARE FAST
   - sum(), min(), max(), sorted() are written in C — use them!
   - List comprehensions are faster than manual for loops

4. FOR ML SPECIFICALLY:
   - Use NumPy for number crunching (we'll cover this in 05_numpy_intro.py)
   - Vectorized operations are MUCH faster than Python loops
   - Pandas for data manipulation is built on NumPy
""")

# Quick demo: comprehension vs loop
print("Speed comparison: loop vs comprehension vs built-in\n")

big_data = list(range(100000))

# Loop
start = time.time()
total = 0
for x in big_data:
    total += x * 2
loop_time = time.time() - start

# Comprehension
start = time.time()
total = sum(x * 2 for x in big_data)
comp_time = time.time() - start

# Built-in with map
start = time.time()
total = sum(map(lambda x: x * 2, big_data))
map_time = time.time() - start

print(f"  Manual loop:       {loop_time:.4f}s")
print(f"  Generator + sum(): {comp_time:.4f}s")
print(f"  map() + sum():     {map_time:.4f}s")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# sum_all correctness (O(n) function)
assert sum_all([]) == 0
assert sum_all([1, 2, 3, 4, 5]) == 15
assert sum_all([100]) == 100
assert sum_all(list(range(101))) == 5050   # 0+1+2+...+100

# has_duplicates_slow correctness
assert has_duplicates_slow([1, 2, 3, 4, 5]) == (False, 10)   # no duplicates, n*(n-1)/2 comparisons for n=5
assert has_duplicates_slow([1, 2, 2, 4]) [0] == True         # has duplicate

# has_duplicates_fast correctness
assert has_duplicates_fast([1, 2, 3, 4, 5]) == False
assert has_duplicates_fast([1, 2, 2, 4]) == True
assert has_duplicates_fast([]) == False
assert has_duplicates_fast([7]) == False

# binary_search step count (O(log n) function)
# A list of 1000 elements should take at most 10 steps (log2(1000) < 10)
data_1000 = list(range(1000))
steps = binary_search(data_1000, 999)
assert steps <= 10, f"Expected at most 10 steps, got {steps}"

# binary search on specific values
sorted_small = list(range(10))   # [0,1,...,9]
assert binary_search(sorted_small, 0) <= 4
assert binary_search(sorted_small, 9) <= 4

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)

# 1. What is the Big-O of this function? (Write your answer as a comment)
#    def mystery(items):
#        for item in items:
#            print(item)
#    Hint: How many times does the loop run relative to the length of items?

# Your answer here:


# 2. What is the Big-O of this function?
#    def mystery2(items):
#        for i in items:
#            for j in items:
#                print(i, j)
#    Hint: For each item, you loop through ALL items again.

# Your answer here:


# 3. Write two versions of a function that checks if a list has any duplicates:
#    a) A SLOW version using nested loops — O(n^2)
#    b) A FAST version using a set — O(n)
#    Then time both on a list of 10,000 items and print the results.
#    Hint: For the fast version, add items to a set and check before adding.

# Your code here:


# 4. You have a list of 1,000,000 numbers and need to check if specific values
#    exist. Write code that:
#    a) Creates a list of 1,000,000 random numbers
#    b) Converts it to a set
#    c) Times how long it takes to check 1000 lookups in the list vs the set
#    Hint: Use random.randint() and the time module.

# Your code here:


print("\nDone! Try the exercises above by writing code and re-running this file.")
