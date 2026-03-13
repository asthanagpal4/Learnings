# HOW TO RUN:
#   uv run python 02_programming/03_list_comprehensions.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- LIST COMPREHENSIONS ---
# Comprehensions are a shorter, more Pythonic way to create lists, dicts, and sets.
# Instead of writing a for-loop to build a new list, you do it in one line.
# In ML, you'll use these constantly to transform and filter data quickly.


# === CONCEPT 1: BASIC LIST COMPREHENSION ===
# Old way (for-loop):
#   result = []
#   for x in something:
#       result.append(do_something(x))
#
# New way (comprehension):
#   result = [do_something(x) for x in something]

print("=" * 50)
print("CONCEPT 1: Basic list comprehension")
print("=" * 50)

# Old way — doubling each number
doubles_old = []
for n in [1, 2, 3, 4, 5]:
    doubles_old.append(n * 2)
print(f"Old way: {doubles_old}")
# Output: Old way: [2, 4, 6, 8, 10]

# New way — same thing in one line!
doubles_new = [n * 2 for n in [1, 2, 3, 4, 5]]
print(f"New way: {doubles_new}")
# Output: New way: [2, 4, 6, 8, 10]

# More examples
squares = [x ** 2 for x in range(1, 6)]
print(f"Squares of 1-5: {squares}")
# Output: Squares of 1-5: [1, 4, 9, 16, 25]

names = ["alice", "bob", "charlie"]
upper_names = [name.upper() for name in names]
print(f"Uppercase names: {upper_names}")
# Output: Uppercase names: ['ALICE', 'BOB', 'CHARLIE']


# === CONCEPT 2: COMPREHENSION WITH A CONDITION (FILTERING) ===
# You can add "if" at the end to only include items that pass a test.
# Pattern: [expression for item in iterable if condition]

print("\n" + "=" * 50)
print("CONCEPT 2: Filtering with if")
print("=" * 50)

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

evens = [n for n in numbers if n % 2 == 0]
print(f"Even numbers: {evens}")
# Output: Even numbers: [2, 4, 6, 8, 10]

big_numbers = [n for n in numbers if n > 5]
print(f"Numbers greater than 5: {big_numbers}")
# Output: Numbers greater than 5: [6, 7, 8, 9, 10]

# Filter strings by length
words = ["cat", "elephant", "dog", "hippopotamus", "ant"]
long_words = [w for w in words if len(w) > 3]
print(f"Words longer than 3 chars: {long_words}")
# Output: Words longer than 3 chars: ['elephant', 'hippopotamus']


# === CONCEPT 3: COMPREHENSION WITH IF-ELSE (TRANSFORMING) ===
# If you want to transform (not filter), put the if-else BEFORE the "for".
# Pattern: [value_if_true if condition else value_if_false for item in iterable]

print("\n" + "=" * 50)
print("CONCEPT 3: if-else in comprehensions")
print("=" * 50)

numbers = [1, 2, 3, 4, 5, 6]
labels = ["even" if n % 2 == 0 else "odd" for n in numbers]
print(f"Labels: {labels}")
# Output: Labels: ['odd', 'even', 'odd', 'even', 'odd', 'even']

scores = [85, 42, 91, 38, 76, 55]
results = ["pass" if s >= 50 else "fail" for s in scores]
print(f"Results: {results}")
# Output: Results: ['pass', 'fail', 'pass', 'fail', 'pass', 'pass']


# === CONCEPT 4: DICTIONARY COMPREHENSIONS ===
# Same idea, but creates a dictionary instead of a list.
# Pattern: {key: value for item in iterable}

print("\n" + "=" * 50)
print("CONCEPT 4: Dictionary comprehensions")
print("=" * 50)

# Create a dict mapping numbers to their squares
square_dict = {n: n ** 2 for n in range(1, 6)}
print(f"Squares dict: {square_dict}")
# Output: Squares dict: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Create a dict from two lists using zip()
students = ["Alice", "Bob", "Charlie"]
grades = [85, 92, 78]
grade_book = {name: grade for name, grade in zip(students, grades)}
print(f"Grade book: {grade_book}")
# Output: Grade book: {'Alice': 85, 'Bob': 92, 'Charlie': 78}

# Filter a dictionary — only keep passing students
passing = {name: grade for name, grade in grade_book.items() if grade >= 80}
print(f"Passing students: {passing}")
# Output: Passing students: {'Alice': 85, 'Bob': 92}


# === CONCEPT 5: SET COMPREHENSIONS ===
# Same idea but creates a set (no duplicates, no order).
# Pattern: {expression for item in iterable}

print("\n" + "=" * 50)
print("CONCEPT 5: Set comprehensions")
print("=" * 50)

numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique_squares = {n ** 2 for n in numbers}
print(f"Unique squares: {unique_squares}")
# Output: Unique squares: {16, 1, 4, 9}

words = ["Hello", "hello", "HELLO", "world", "World"]
unique_lower = {w.lower() for w in words}
print(f"Unique lowercase words: {unique_lower}")
# Output: Unique lowercase words: {'hello', 'world'}


# === CONCEPT 6: NESTED COMPREHENSIONS ===
# You can put a comprehension inside another one.
# This is like a nested for-loop. Don't overdo it — readability matters!

print("\n" + "=" * 50)
print("CONCEPT 6: Nested comprehensions")
print("=" * 50)

# All combinations of two lists (like a multiplication table)
rows = [1, 2, 3]
cols = [10, 20, 30]
products = [r * c for r in rows for c in cols]
print(f"Products: {products}")
# Output: Products: [10, 20, 30, 20, 40, 60, 30, 60, 90]

# Flatten a list of lists
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(f"Flattened: {flat}")
# Output: Flattened: [1, 2, 3, 4, 5, 6, 7, 8, 9]


# === CONCEPT 7: PRACTICAL ML-STYLE EXAMPLES ===
# These patterns come up a lot when working with data.

print("\n" + "=" * 50)
print("CONCEPT 7: Practical examples")
print("=" * 50)

# Normalize scores to 0-1 range (common in ML!)
raw_scores = [60, 75, 80, 95, 50]
max_score = max(raw_scores)
normalized = [score / max_score for score in raw_scores]
print(f"Raw scores: {raw_scores}")
print(f"Normalized:  {[round(n, 2) for n in normalized]}")
# Output: Normalized:  [0.63, 0.79, 0.84, 1.0, 0.53]

# Clean messy data
messy_data = ["  Alice  ", "bob", "  CHARLIE", " diana "]
clean_data = [name.strip().title() for name in messy_data]
print(f"Cleaned names: {clean_data}")
# Output: Cleaned names: ['Alice', 'Bob', 'Charlie', 'Diana']

# Extract numbers from mixed data
mixed = ["age:25", "height:170", "weight:65", "name:Astha"]
numbers_only = {k: int(v) for item in mixed
                for k, v in [item.split(":")]
                if v.isdigit()}
print(f"Extracted numbers: {numbers_only}")
# Output: Extracted numbers: {'age': 25, 'height': 170, 'weight': 65}


# === CONCEPT 8: WHEN NOT TO USE COMPREHENSIONS ===
# Comprehensions are great, but don't force them.
# If the logic is complicated, a regular loop is easier to read.

print("\n" + "=" * 50)
print("CONCEPT 8: Keep it readable")
print("=" * 50)

# This is fine — simple and clear:
evens = [n for n in range(20) if n % 2 == 0]
print(f"Simple (good): {evens}")

# But if you need multiple steps, use a regular loop:
data = ["Alice:85", "Bob:fail", "Charlie:92", "Diana:bad"]
parsed = {}
for item in data:
    name, score_str = item.split(":")
    try:
        parsed[name] = int(score_str)
    except ValueError:
        parsed[name] = None  # mark as missing

print(f"Complex parsing (use a loop): {parsed}")
# Output: Complex parsing (use a loop): {'Alice': 85, 'Bob': None, 'Charlie': 92, 'Diana': None}


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test 1: Basic list comprehension — doubling numbers
assert doubles_new == [2, 4, 6, 8, 10], "doubles_new should be [2,4,6,8,10]"
assert squares == [1, 4, 9, 16, 25], "squares should be [1,4,9,16,25]"
assert upper_names == ["ALICE", "BOB", "CHARLIE"], "upper_names should be uppercase"

# Test 2: Filtering with if
assert evens == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18], "evens from range(20) should be correct"

_test_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_test_evens = [n for n in _test_numbers if n % 2 == 0]
assert _test_evens == [2, 4, 6, 8, 10], "Even numbers from 1-10 should be [2,4,6,8,10]"

_test_long = [w for w in ["cat", "elephant", "dog", "hippopotamus", "ant"] if len(w) > 3]
assert _test_long == ["elephant", "hippopotamus"], "Long words filter wrong"

# Test 3: if-else in comprehension (transforming, not filtering)
_test_labels = ["even" if n % 2 == 0 else "odd" for n in [1, 2, 3, 4, 5, 6]]
assert _test_labels == ["odd", "even", "odd", "even", "odd", "even"], "Labels wrong"

_test_results = ["pass" if s >= 50 else "fail" for s in [85, 42, 91, 38, 76, 55]]
assert _test_results == ["pass", "fail", "pass", "fail", "pass", "pass"], "Results wrong"

# Test 4: Dictionary comprehensions
_test_sq_dict = {n: n ** 2 for n in range(1, 6)}
assert _test_sq_dict == {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}, "Square dict wrong"

_test_passing = {name: g for name, g in {"Alice": 85, "Bob": 92, "Charlie": 78}.items() if g >= 80}
assert _test_passing == {"Alice": 85, "Bob": 92}, "Passing dict should only include >= 80"

# Test 5: Set comprehensions remove duplicates
_test_unique = {n ** 2 for n in [1, 2, 2, 3]}
assert _test_unique == {1, 4, 9}, "Unique squares wrong"

# Test 6: Nested comprehension — flatten a matrix
_test_flat = [num for row in [[1, 2], [3, 4], [5, 6]] for num in row]
assert _test_flat == [1, 2, 3, 4, 5, 6], "Flattened matrix wrong"

# Test 7: Normalization
_test_raw = [50, 100]
_test_max = max(_test_raw)
_test_norm = [s / _test_max for s in _test_raw]
assert _test_norm == [0.5, 1.0], "Normalized values wrong"

# Test 8: Clean messy data
_test_clean = [name.strip().title() for name in ["  alice  ", "BOB"]]
assert _test_clean == ["Alice", "Bob"], "Cleaned names wrong"

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Use a list comprehension to create a list of the first 10 cube numbers.
#    (1^3, 2^3, 3^3, ..., 10^3)
#    Expected: [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
#    Hint: [x ** 3 for x in ...]

# YOUR CODE HERE:


# 2. Given this list of temperatures in Celsius:
#    celsius = [0, 10, 20, 30, 40, 100]
#    Convert each to Fahrenheit using a list comprehension.
#    Formula: F = C * 9/5 + 32
#    Expected: [32.0, 50.0, 68.0, 86.0, 104.0, 212.0]

# YOUR CODE HERE:


# 3. Given this dictionary:
#    inventory = {"apples": 5, "bananas": 0, "cherries": 12, "dates": 0, "elderberries": 3}
#    Use a dict comprehension to create a new dict with only items that have stock > 0.
#    Expected: {'apples': 5, 'cherries': 12, 'elderberries': 3}

# YOUR CODE HERE:


# 4. Given a sentence, use a list comprehension to get the length of each word.
#    sentence = "the quick brown fox jumps over the lazy dog"
#    Expected: [3, 5, 5, 3, 5, 4, 3, 4, 3]
#    Hint: split the sentence first, then get len() of each word

# YOUR CODE HERE:
