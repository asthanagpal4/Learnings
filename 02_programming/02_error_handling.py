# HOW TO RUN:
#   uv run python 02_programming/02_error_handling.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- ERROR HANDLING ---
# Programs crash when something unexpected happens (bad input, missing file, etc.).
# Error handling lets your program deal with problems gracefully instead of crashing.
# In ML, you'll handle errors when loading corrupted data, network failures, etc.
# Python uses try/except blocks to catch and handle errors.


# === CONCEPT 1: WHAT HAPPENS WITHOUT ERROR HANDLING ===
# When Python hits an error, it stops the entire program.
# Let's see what kinds of errors exist (we'll catch them so the program continues).

print("=" * 50)
print("CONCEPT 1: Common error types")
print("=" * 50)

# TypeError — wrong type of data
# For example: "hello" + 5 would crash (can't add string and number)

# ValueError — right type, wrong value
# For example: int("abc") would crash (can't convert "abc" to a number)

# ZeroDivisionError — dividing by zero
# For example: 10 / 0 would crash

# FileNotFoundError — file doesn't exist
# For example: open("nonexistent.txt") would crash

# KeyError — key not found in dictionary
# For example: {"a": 1}["b"] would crash

# IndexError — list index out of range
# For example: [1, 2, 3][10] would crash

print("We'll learn to handle all of these!")


# === CONCEPT 2: TRY / EXCEPT ===
# "try" runs code that might fail.
# "except" runs only if an error happens — it catches the error.

print("\n" + "=" * 50)
print("CONCEPT 2: try / except")
print("=" * 50)

# Without error handling, this would crash the program:
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Oops! You can't divide by zero.")
# Output: Oops! You can't divide by zero.

# The program keeps running after the error is caught!
print("Program is still running!")
# Output: Program is still running!


# === CONCEPT 3: CATCHING SPECIFIC ERRORS ===
# Always catch the specific error you expect, not all errors.
# This makes debugging easier.

print("\n" + "=" * 50)
print("CONCEPT 3: Catching specific errors")
print("=" * 50)

# Example: Converting user input to a number
test_inputs = ["42", "hello", "3.14"]

for value in test_inputs:
    try:
        number = int(value)
        print(f"'{value}' converted to integer: {number}")
    except ValueError:
        print(f"'{value}' is not a valid integer!")

# Output:
# '42' converted to integer: 42
# 'hello' is not a valid integer!
# '3.14' is not a valid integer!


# === CONCEPT 4: CATCHING MULTIPLE ERROR TYPES ===
# You can have multiple except blocks for different errors.

print("\n" + "=" * 50)
print("CONCEPT 4: Multiple except blocks")
print("=" * 50)

def safe_divide(a, b):
    """Divides a by b, handling errors gracefully."""
    try:
        result = a / b
        print(f"{a} / {b} = {result}")
    except ZeroDivisionError:
        print(f"Cannot divide {a} by zero!")
    except TypeError:
        print(f"Cannot divide {a} by {b} — wrong types!")

safe_divide(10, 3)       # Output: 10 / 3 = 3.3333333333333335
safe_divide(10, 0)       # Output: Cannot divide 10 by zero!
safe_divide(10, "two")   # Output: Cannot divide 10 by two — wrong types!


# === CONCEPT 5: THE ERROR MESSAGE ===
# You can capture the error message using "as e".

print("\n" + "=" * 50)
print("CONCEPT 5: Getting the error message")
print("=" * 50)

try:
    numbers = [1, 2, 3]
    print(numbers[10])
except IndexError as e:
    print(f"Error caught: {e}")
# Output: Error caught: list index out of range

try:
    person = {"name": "Astha"}
    print(person["age"])
except KeyError as e:
    print(f"Error caught: missing key {e}")
# Output: Error caught: missing key 'age'


# === CONCEPT 6: TRY / EXCEPT / ELSE / FINALLY ===
# else — runs only if NO error happened
# finally — runs NO MATTER WHAT (error or not)

print("\n" + "=" * 50)
print("CONCEPT 6: else and finally")
print("=" * 50)

def convert_to_int(text):
    try:
        number = int(text)
    except ValueError:
        print(f"  '{text}' failed to convert")
    else:
        print(f"  '{text}' converted successfully to {number}")
    finally:
        print(f"  Done processing '{text}'")

convert_to_int("99")
# Output:
#   '99' converted successfully to 99
#   Done processing '99'

print()

convert_to_int("abc")
# Output:
#   'abc' failed to convert
#   Done processing 'abc'


# === CONCEPT 7: RAISING YOUR OWN ERRORS ===
# Sometimes YOU want to signal that something is wrong.
# Use "raise" to create an error on purpose.

print("\n" + "=" * 50)
print("CONCEPT 7: Raising errors")
print("=" * 50)

def calculate_average(scores):
    """Calculate average score. Raises error if list is empty."""
    if not scores:
        raise ValueError("Cannot calculate average of an empty list!")
    return sum(scores) / len(scores)

# This works fine
avg = calculate_average([85, 92, 78])
print(f"Average: {avg}")
# Output: Average: 85.0

# This would raise our custom error — let's catch it
try:
    avg = calculate_average([])
except ValueError as e:
    print(f"Error: {e}")
# Output: Error: Cannot calculate average of an empty list!


# === CONCEPT 8: DEFENSIVE PROGRAMMING ===
# Check for problems BEFORE they happen. Validate your data early.

print("\n" + "=" * 50)
print("CONCEPT 8: Defensive programming")
print("=" * 50)

def process_score(name, score):
    """Process a student's score with validation."""
    # Check types first
    if not isinstance(name, str):
        raise TypeError(f"Name must be a string, got {type(name).__name__}")
    if not isinstance(score, (int, float)):
        raise TypeError(f"Score must be a number, got {type(score).__name__}")

    # Check valid range
    if score < 0 or score > 100:
        raise ValueError(f"Score must be 0-100, got {score}")

    # If we get here, the data is good
    grade = "Pass" if score >= 50 else "Fail"
    print(f"  {name}: {score} -> {grade}")

# Good data
process_score("Astha", 88)
# Output:   Astha: 88 -> Pass

# Bad data — caught gracefully
test_cases = [
    ("Bob", 150),       # score too high
    ("Charlie", -5),    # score negative
    (123, 85),          # name is not a string
]

for name, score in test_cases:
    try:
        process_score(name, score)
    except (TypeError, ValueError) as e:
        print(f"  Rejected: {e}")

# Output:
#   Rejected: Score must be 0-100, got 150
#   Rejected: Score must be 0-100, got -5
#   Rejected: Name must be a string, got int


# === CONCEPT 9: A PRACTICAL PATTERN — SAFE FILE READING ===
# Combining file handling with error handling — very common in real code.

print("\n" + "=" * 50)
print("CONCEPT 9: Safe file reading pattern")
print("=" * 50)

import os

def read_scores(filename):
    """Read scores from a file safely."""
    if not os.path.exists(filename):
        print(f"  File '{filename}' not found!")
        return {}

    scores = {}
    with open(filename, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:       # skip empty lines
                continue
            try:
                parts = line.split(",")
                name = parts[0]
                score = int(parts[1])
                scores[name] = score
            except (ValueError, IndexError):
                print(f"  Warning: Could not parse line {line_num}: '{line}'")

    return scores

# Create a test file with some bad data
with open("test_scores.txt", "w") as f:
    f.write("Alice,85\n")
    f.write("Bob,ninety\n")      # bad score
    f.write("Charlie\n")         # missing score
    f.write("Diana,95\n")

result = read_scores("test_scores.txt")
print(f"  Successfully loaded: {result}")
# Output:
#   Warning: Could not parse line 2: 'Bob,ninety'
#   Warning: Could not parse line 3: 'Charlie'
#   Successfully loaded: {'Alice': 85, 'Diana': 95}

result2 = read_scores("nonexistent.txt")
# Output:   File 'nonexistent.txt' not found!

# Cleanup
os.remove("test_scores.txt")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test 1: safe_divide — normal division works
def _test_safe_divide_result(a, b):
    """Returns result or None if error."""
    try:
        return a / b
    except (ZeroDivisionError, TypeError):
        return None

assert _test_safe_divide_result(10, 2) == 5.0, "10/2 should be 5.0"
assert _test_safe_divide_result(10, 0) is None, "Divide by zero should return None"
assert _test_safe_divide_result(10, "x") is None, "Type error should return None"

# Test 2: calculate_average raises ValueError on empty list
_raised = False
try:
    calculate_average([])
except ValueError:
    _raised = True
assert _raised, "calculate_average([]) should raise ValueError"

# Test 3: calculate_average returns correct value on non-empty list
assert calculate_average([80, 90, 100]) == 90.0, "Average of [80,90,100] should be 90.0"
assert calculate_average([50]) == 50.0, "Average of single-item list should be that item"

# Test 4: process_score raises TypeError for non-string name
_raised = False
try:
    process_score(123, 80)
except TypeError:
    _raised = True
assert _raised, "process_score with integer name should raise TypeError"

# Test 5: process_score raises ValueError for out-of-range score
_raised = False
try:
    process_score("Alice", 150)
except ValueError:
    _raised = True
assert _raised, "process_score with score=150 should raise ValueError"

_raised = False
try:
    process_score("Alice", -5)
except ValueError:
    _raised = True
assert _raised, "process_score with score=-5 should raise ValueError"

# Test 6: read_scores returns empty dict for missing file
_result = read_scores("_nonexistent_file_xyz.txt")
assert _result == {}, "read_scores on missing file should return {}"

# Test 7: read_scores correctly parses a valid file
import os as _os
with open("_test_scores_tmp.txt", "w") as _f:
    _f.write("Alice,85\n")
    _f.write("Bob,92\n")
    _f.write("bad_line\n")

_parsed = read_scores("_test_scores_tmp.txt")
assert _parsed.get("Alice") == 85, "Alice's score should be 85"
assert _parsed.get("Bob") == 92, "Bob's score should be 92"
_os.remove("_test_scores_tmp.txt")

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Write a function safe_int(text) that:
#    - Tries to convert text to an integer
#    - Returns the integer if successful
#    - Returns None if it fails
#    Test it with: safe_int("42"), safe_int("hello"), safe_int("3.5")
#    Expected: 42, None, None
#    Hint: use try/except ValueError, and return None in the except block

# YOUR CODE HERE:


def safe_int(text: str) -> int | None:
    try:
        number = int(text)
        return number
    except ValueError:
        return None


print(safe_int("42"))

print(safe_int("hello"))

print(safe_int("3.5"))


# 2. Write a function safe_get(dictionary, key) that:
#    - Returns the value for the key if it exists
#    - Returns "Key not found" if the key doesn't exist
#    - Uses try/except (not the .get() method)
#    Test with: safe_get({"a": 1, "b": 2}, "a") and safe_get({"a": 1}, "c")
#    Expected: 1, "Key not found"
#    Hint: catch KeyError

# YOUR CODE HERE:
def safe_get(dictionary: dict, key: str) -> int | str:
    try:
        value = dictionary[key]
        return value
    except KeyError:
        return "Key not found"

print(safe_get({"a": 1, "b": 2}, "a"))
print(safe_get({"a": 1}, "c"))


# 3. Write a function divide_list(numbers, divisor) that:
#    - Divides every number in the list by divisor
#    - Returns a new list with the results
#    - Handles ZeroDivisionError (return empty list)
#    - Handles TypeError if non-numbers are in the list (skip them)
#    Test with: divide_list([10, 20, 30], 5) -> [2.0, 4.0, 6.0]
#    Test with: divide_list([10, 20], 0) -> []
#    Hint: check for zero divisor first, then use try/except inside the loop

# YOUR CODE HERE:
def divide_list(numbers: list, divisor: int | float) -> list[float]:
    if divisor == 0:
        return []
    result = []
    for number in numbers:
        try:
            result.append(number/divisor)
        except TypeError:
            continue 
    return result
print(divide_list([10, 20, 30], 5))
print(divide_list([10, 20], 0))


