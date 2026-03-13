# HOW TO RUN:
#   uv run python 04_algorithms/01_sorting.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- SORTING ALGORITHMS ---
# Sorting means putting items in order (smallest to largest, A to Z, etc.).
# It's one of the most common things you'll do in programming and in ML
# (for example, ranking predictions by confidence).
#
# We'll learn two sorting methods from scratch, then see Python's built-in sort.
# We'll also introduce "time complexity" — a way to talk about how fast an algorithm is.

print("=" * 60)
print("SECTION 4.1 — SORTING ALGORITHMS")
print("=" * 60)


# === CONCEPT 1: BUBBLE SORT ===
# Bubble sort is the simplest sorting algorithm.
# Idea: Walk through the list, compare neighbors, swap if out of order.
#        Repeat until no swaps are needed.
# It's called "bubble" sort because bigger values "bubble up" to the end.

print("\n=== BUBBLE SORT ===\n")

def bubble_sort(items):
    """Sort a list in ascending order using bubble sort."""
    # Make a copy so we don't change the original list
    result = items.copy()
    n = len(result)

    for i in range(n):
        # Track if we made any swaps this pass
        swapped = False

        # Compare each pair of neighbors
        # (we subtract i because the last i items are already sorted)
        for j in range(n - 1 - i):
            if result[j] > result[j + 1]:
                # Swap them!
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True

        # If no swaps happened, the list is already sorted — we can stop early
        if not swapped:
            break

    return result

# Let's try it
numbers = [64, 34, 25, 12, 22, 11, 90]
print(f"Before sorting: {numbers}")
sorted_numbers = bubble_sort(numbers)
print(f"After sorting:  {sorted_numbers}")
# Output: After sorting:  [11, 12, 22, 25, 34, 64, 90]

# Let's watch it step by step on a small list
print("\nStep-by-step bubble sort on [5, 3, 1, 4, 2]:")
demo = [5, 3, 1, 4, 2]
n = len(demo)
for i in range(n):
    for j in range(n - 1 - i):
        if demo[j] > demo[j + 1]:
            demo[j], demo[j + 1] = demo[j + 1], demo[j]
    print(f"  After pass {i + 1}: {demo}")
# Output:
#   After pass 1: [3, 1, 4, 2, 5]
#   After pass 2: [1, 3, 2, 4, 5]
#   After pass 3: [1, 2, 3, 4, 5]
#   After pass 4: [1, 2, 3, 4, 5]
#   After pass 5: [1, 2, 3, 4, 5]


# === CONCEPT 2: MERGE SORT ===
# Merge sort is much faster than bubble sort on large lists.
# Idea: Split the list in half, sort each half, then merge them back together.
# This is a "divide and conquer" strategy — break a big problem into smaller ones.

print("\n=== MERGE SORT ===\n")

def merge_sort(items):
    """Sort a list in ascending order using merge sort."""
    # Base case: a list with 0 or 1 items is already sorted
    if len(items) <= 1:
        return items

    # Split the list in half
    mid = len(items) // 2
    left_half = items[:mid]
    right_half = items[mid:]

    # Sort each half (this is the recursive part — the function calls itself!)
    left_sorted = merge_sort(left_half)
    right_sorted = merge_sort(right_half)

    # Merge the two sorted halves together
    return merge(left_sorted, right_sorted)

def merge(left, right):
    """Merge two sorted lists into one sorted list."""
    result = []
    i = 0  # pointer for left list
    j = 0  # pointer for right list

    # Compare elements from both lists and pick the smaller one
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Add any remaining elements (one list might be longer)
    result.extend(left[i:])
    result.extend(right[j:])

    return result

numbers2 = [38, 27, 43, 3, 9, 82, 10]
print(f"Before sorting: {numbers2}")
sorted_numbers2 = merge_sort(numbers2)
print(f"After sorting:  {sorted_numbers2}")
# Output: After sorting:  [3, 9, 10, 27, 38, 43, 82]


# === CONCEPT 3: PYTHON'S BUILT-IN SORT ===
# Python has built-in sorting that is very fast and well-tested.
# In real code, you'll almost always use this instead of writing your own.
# But understanding HOW sorting works helps you think about algorithms.

print("\n=== PYTHON'S BUILT-IN SORT ===\n")

# sorted() returns a NEW sorted list (original is unchanged)
fruits = ["banana", "apple", "cherry", "date"]
print(f"Original:        {fruits}")
print(f"sorted():        {sorted(fruits)}")
print(f"Original after:  {fruits}")  # unchanged!
# Output:
# Original:        ['banana', 'apple', 'cherry', 'date']
# sorted():        ['apple', 'banana', 'cherry', 'date']
# Original after:  ['banana', 'apple', 'cherry', 'date']

# .sort() modifies the list in place (no return value)
fruits.sort()
print(f"After .sort():   {fruits}")
# Output: After .sort():   ['apple', 'banana', 'cherry', 'date']

# Reverse sort
nums = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"\nAscending:  {sorted(nums)}")
print(f"Descending: {sorted(nums, reverse=True)}")
# Output:
# Ascending:  [1, 1, 2, 3, 4, 5, 6, 9]
# Descending: [9, 6, 5, 4, 3, 2, 1, 1]

# Sort by a custom rule using key=
# Example: sort words by their length
words = ["elephant", "cat", "do", "butterfly", "ant"]
print(f"\nBy length:    {sorted(words, key=len)}")
# Output: By length:    ['do', 'cat', 'ant', 'elephant', 'butterfly']

# Sort a list of tuples by the second element
students = [("Astha", 85), ("Bob", 72), ("Cara", 91), ("Dev", 68)]
by_score = sorted(students, key=lambda s: s[1])
print(f"By score:     {by_score}")
# Output: By score:     [('Dev', 68), ('Bob', 72), ('Astha', 85), ('Cara', 91)]

by_score_desc = sorted(students, key=lambda s: s[1], reverse=True)
print(f"Top scorers:  {by_score_desc}")
# Output: Top scorers:  [('Cara', 91), ('Astha', 85), ('Bob', 72), ('Dev', 68)]


# === CONCEPT 4: TIME COMPLEXITY INTRO ===
# "How fast is this algorithm?" depends on how many items you have.
#
# Bubble sort: In the worst case, it does about n*n comparisons.
#   We say it's O(n^2) — "order n squared".
#   Double the list size? It takes about 4x longer.
#
# Merge sort: It does about n * log(n) comparisons.
#   We say it's O(n log n) — much faster for big lists.
#
# Python's sorted() uses a clever algorithm called Timsort,
#   which is also O(n log n), and it's optimized for real-world data.
#
# We'll explore Big-O in detail in file 04_big_o.py.

print("\n=== TIME COMPLEXITY PREVIEW ===\n")

import time

# Let's compare bubble sort and Python's sorted on different list sizes
import random

for size in [100, 500, 1000, 2000]:
    test_list = [random.randint(1, 10000) for _ in range(size)]

    # Time bubble sort
    start = time.time()
    bubble_sort(test_list)
    bubble_time = time.time() - start

    # Time Python's sorted
    start = time.time()
    sorted(test_list)
    python_time = time.time() - start

    print(f"List size {size:>5}: Bubble sort = {bubble_time:.4f}s, "
          f"Python sorted = {python_time:.6f}s")

print("\nNotice: As the list gets bigger, bubble sort slows down MUCH faster!")
print("That's the difference between O(n^2) and O(n log n).")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# bubble_sort correctness
assert bubble_sort([64, 34, 25, 12, 22, 11, 90]) == [11, 12, 22, 25, 34, 64, 90]
assert bubble_sort([5, 3, 1, 4, 2]) == [1, 2, 3, 4, 5]
assert bubble_sort([1]) == [1]
assert bubble_sort([]) == []
assert bubble_sort([3, 3, 1]) == [1, 3, 3]  # handles duplicates

# merge_sort correctness
assert merge_sort([38, 27, 43, 3, 9, 82, 10]) == [3, 9, 10, 27, 38, 43, 82]
assert merge_sort([5, 3, 1, 4, 2]) == [1, 2, 3, 4, 5]
assert merge_sort([1]) == [1]
assert merge_sort([]) == []
assert merge_sort([3, 3, 1]) == [1, 3, 3]

# both sorts agree with Python's built-in sorted()
test_cases = [[9, 4, 7, 2, 8, 1], [100, 50, 75], [-3, 0, 5, -1]]
for tc in test_cases:
    expected = sorted(tc)
    assert bubble_sort(tc) == expected
    assert merge_sort(tc) == expected

# Python built-in sort behaviour
assert sorted(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]
students = [("Astha", 85), ("Bob", 72), ("Cara", 91), ("Dev", 68)]
assert sorted(students, key=lambda s: s[1]) == [("Dev", 68), ("Bob", 72), ("Astha", 85), ("Cara", 91)]

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)

# 1. Write a function that sorts a list in DESCENDING order using bubble sort.
#    Hint: Just change the comparison from > to <.
#    Test with: [3, 7, 1, 9, 4] -> should give [9, 7, 4, 3, 1]

# Your code here:


# 2. Sort this list of dictionaries by the "age" key:
#    people = [{"name": "Astha", "age": 25}, {"name": "Bob", "age": 20},
#              {"name": "Cara", "age": 30}]
#    Hint: Use sorted() with key=lambda
#    Expected: [{'name': 'Bob', 'age': 20}, {'name': 'Astha', 'age': 25},
#               {'name': 'Cara', 'age': 30}]

# Your code here:


# 3. Write a function that counts how many swaps bubble sort makes.
#    Hint: Add a counter variable, increment it each time you swap.
#    Test with [5, 3, 1] — it should make 3 swaps.

# Your code here:


# 4. Sort a list of strings by their LAST character.
#    words = ["hello", "world", "python", "code"]
#    Hint: key=lambda w: w[-1]
#    Expected: ['code', 'world', 'hello', 'python']

# Your code here:


print("\nDone! Try the exercises above by writing code and re-running this file.")
