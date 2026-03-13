# HOW TO RUN:
#   uv run python 04_algorithms/03_recursion.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- RECURSION ---
# Recursion is when a function calls ITSELF to solve a problem.
# It sounds strange, but it's a powerful technique.
# Think of it like Russian nesting dolls — each doll contains a smaller version
# of itself, until you reach the tiniest one (the "base case").
#
# In ML, recursion shows up in tree-based models (decision trees, random forests),
# parsing data structures, and divide-and-conquer algorithms.

print("=" * 60)
print("SECTION 4.3 — RECURSION")
print("=" * 60)


# === CONCEPT 1: BASE CASE AND RECURSIVE CASE ===
# Every recursive function needs TWO things:
#   1. BASE CASE — when to STOP (the simplest version of the problem)
#   2. RECURSIVE CASE — break the problem into a smaller version of itself
#
# Without a base case, the function calls itself forever (until Python crashes).

print("\n=== BASE CASE AND RECURSIVE CASE ===\n")

# Classic example: Countdown
def countdown(n):
    """Count down from n to 1, then say 'Go!'"""
    if n <= 0:           # BASE CASE: stop when we reach 0
        print("Go!")
        return
    print(n)
    countdown(n - 1)     # RECURSIVE CASE: call ourselves with n-1

print("Countdown from 5:")
countdown(5)
# Output:
# 5
# 4
# 3
# 2
# 1
# Go!


# === CONCEPT 2: FACTORIAL — A CLASSIC EXAMPLE ===
# Factorial of n (written n!) = n * (n-1) * (n-2) * ... * 1
# Example: 5! = 5 * 4 * 3 * 2 * 1 = 120
#
# Notice: 5! = 5 * 4!
#         4! = 4 * 3!
#         ...
#         1! = 1  (base case!)

print("\n=== FACTORIAL ===\n")

def factorial(n):
    """Calculate n! using recursion."""
    # Base case
    if n <= 1:
        return 1
    # Recursive case: n! = n * (n-1)!
    return n * factorial(n - 1)

# Let's test it
for i in range(1, 8):
    print(f"{i}! = {factorial(i)}")
# Output:
# 1! = 1
# 2! = 2
# 3! = 6
# 4! = 24
# 5! = 120
# 6! = 720
# 7! = 5040

# Let's trace through factorial(4) to understand what happens:
print("\nTracing factorial(4):")
print("  factorial(4) calls factorial(3)  -> waiting...")
print("    factorial(3) calls factorial(2)  -> waiting...")
print("      factorial(2) calls factorial(1)  -> waiting...")
print("        factorial(1) returns 1  (base case!)")
print("      factorial(2) returns 2 * 1 = 2")
print("    factorial(3) returns 3 * 2 = 6")
print("  factorial(4) returns 4 * 6 = 24")


# === CONCEPT 3: THE CALL STACK ===
# When a function calls itself, Python keeps track of each call in a "stack"
# (like a stack of plates — last in, first out).
# Each call waits for the one inside it to finish before it can complete.
#
# If you recurse too deeply, Python will raise a RecursionError.
# Python's default limit is about 1000 levels deep.

print("\n=== THE CALL STACK ===\n")

def factorial_verbose(n, depth=0):
    """Factorial that shows the call stack."""
    indent = "  " * depth
    print(f"{indent}factorial({n}) called")

    if n <= 1:
        print(f"{indent}factorial({n}) returns 1  <-- base case")
        return 1

    result = n * factorial_verbose(n - 1, depth + 1)
    print(f"{indent}factorial({n}) returns {result}")
    return result

print("Watching the call stack for factorial(5):")
answer = factorial_verbose(5)
print(f"\nFinal answer: {answer}")
# You'll see each call go deeper, then results come back up!

# What happens if there's no base case? Let's NOT run this, just look:
# def infinite_recursion(n):
#     return infinite_recursion(n)  # No base case! This crashes!
# infinite_recursion(1)  # -> RecursionError: maximum recursion depth exceeded

import sys
print(f"\nPython's recursion limit: {sys.getrecursionlimit()}")


# === CONCEPT 4: MORE EXAMPLES ===

print("\n=== MORE RECURSION EXAMPLES ===\n")

# Sum of a list
def recursive_sum(items):
    """Add up all items in a list, recursively."""
    if len(items) == 0:     # Base case: empty list sums to 0
        return 0
    return items[0] + recursive_sum(items[1:])  # First item + sum of the rest

print(f"Sum of [1, 2, 3, 4, 5] = {recursive_sum([1, 2, 3, 4, 5])}")
# Output: Sum of [1, 2, 3, 4, 5] = 15

# Reverse a string
def reverse_string(s):
    """Reverse a string using recursion."""
    if len(s) <= 1:         # Base case: empty or single character
        return s
    return reverse_string(s[1:]) + s[0]  # Reverse the rest, then add first char

print(f"Reverse of 'hello' = '{reverse_string('hello')}'")
# Output: Reverse of 'hello' = 'olleh'

# Check if a word is a palindrome (reads the same forwards and backwards)
def is_palindrome(s):
    """Check if a string is a palindrome."""
    if len(s) <= 1:             # Base case: 0 or 1 characters — it's a palindrome
        return True
    if s[0] != s[-1]:           # First and last don't match — not a palindrome
        return False
    return is_palindrome(s[1:-1])  # Check the middle part

for word in ["racecar", "hello", "madam", "python"]:
    print(f"  '{word}' is a palindrome? {is_palindrome(word)}")
# Output:
#   'racecar' is a palindrome? True
#   'hello' is a palindrome? False
#   'madam' is a palindrome? True
#   'python' is a palindrome? False


# === CONCEPT 5: TREE RECURSION (FIBONACCI) ===
# Some problems make TWO recursive calls — this creates a "tree" of calls.
# The Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
# Each number is the sum of the two before it.
# fib(n) = fib(n-1) + fib(n-2)

print("\n=== TREE RECURSION — FIBONACCI ===\n")

def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0          # Base case 1
    if n == 1:
        return 1          # Base case 2
    return fibonacci(n - 1) + fibonacci(n - 2)  # TWO recursive calls!

print("Fibonacci sequence:")
for i in range(12):
    print(f"  fib({i}) = {fibonacci(i)}")
# Output:
#   fib(0) = 0
#   fib(1) = 1
#   fib(2) = 1
#   fib(3) = 2
#   fib(4) = 3
#   fib(5) = 5
#   fib(6) = 8
#   fib(7) = 13
#   fib(8) = 21
#   fib(9) = 34
#   fib(10) = 55
#   fib(11) = 89

# WARNING: Tree recursion is SLOW for big n because it recalculates things.
# fib(5) calculates fib(3) twice, fib(2) three times, etc.
# This is O(2^n) — very slow!

# A faster way: use a loop (called "iterative" approach)
def fibonacci_fast(n):
    """Calculate Fibonacci using a loop — much faster!"""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

import time

print("\nSpeed comparison for fib(30):")
start = time.time()
result1 = fibonacci(30)
slow_time = time.time() - start
print(f"  Recursive: fib(30) = {result1}, took {slow_time:.4f}s")

start = time.time()
result2 = fibonacci_fast(30)
fast_time = time.time() - start
print(f"  Iterative: fib(30) = {result2}, took {fast_time:.6f}s")
print(f"  The loop version is roughly {slow_time / max(fast_time, 0.000001):.0f}x faster!")


# === CONCEPT 6: RECURSION vs LOOPS ===
# Many things can be done with EITHER recursion or loops.
# General advice:
#   - Use LOOPS when the problem is straightforward (summing, counting)
#   - Use RECURSION when the problem naturally splits into subproblems
#     (trees, nested structures, divide-and-conquer algorithms)
#   - In Python, loops are usually faster (Python isn't optimized for recursion)

print("\n=== RECURSION vs LOOPS ===\n")

# Same task two ways: power function (x raised to n)

def power_recursive(x, n):
    """Calculate x^n using recursion."""
    if n == 0:
        return 1
    return x * power_recursive(x, n - 1)

def power_loop(x, n):
    """Calculate x^n using a loop."""
    result = 1
    for _ in range(n):
        result *= x
    return result

print(f"2^10 (recursive) = {power_recursive(2, 10)}")
print(f"2^10 (loop)      = {power_loop(2, 10)}")
print(f"2^10 (built-in)  = {2 ** 10}")
# All output: 1024


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# factorial correctness
assert factorial(1) == 1
assert factorial(2) == 2
assert factorial(3) == 6
assert factorial(4) == 24
assert factorial(5) == 120
assert factorial(7) == 5040

# recursive_sum correctness
assert recursive_sum([]) == 0
assert recursive_sum([1]) == 1
assert recursive_sum([1, 2, 3, 4, 5]) == 15
assert recursive_sum([10, -3, 7]) == 14

# reverse_string correctness
assert reverse_string("hello") == "olleh"
assert reverse_string("a") == "a"
assert reverse_string("") == ""
assert reverse_string("racecar") == "racecar"

# is_palindrome correctness
assert is_palindrome("racecar") == True
assert is_palindrome("madam") == True
assert is_palindrome("a") == True
assert is_palindrome("") == True
assert is_palindrome("hello") == False
assert is_palindrome("python") == False

# fibonacci correctness
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(2) == 1
assert fibonacci(5) == 5
assert fibonacci(10) == 55
assert fibonacci(11) == 89

# fibonacci_fast matches slow version
for i in range(12):
    assert fibonacci_fast(i) == fibonacci(i)

# power functions
assert power_recursive(2, 10) == 1024
assert power_recursive(3, 3) == 27
assert power_recursive(5, 0) == 1
assert power_loop(2, 10) == 1024
assert power_loop(3, 3) == 27
assert power_loop(5, 0) == 1

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)

# 1. Write a recursive function to calculate the sum of digits of a number.
#    Example: sum_digits(1234) -> 1 + 2 + 3 + 4 = 10
#    Hint: n % 10 gives the last digit, n // 10 removes the last digit.
#    Base case: when n is 0, return 0.

# Your code here:


# 2. Write a recursive function to count how many items are in a nested list.
#    Example: count_items([1, [2, 3], [4, [5, 6]]]) -> 6
#    Hint: If an item is a list (isinstance(item, list)), recurse into it.
#    Base case: if the item is NOT a list, count it as 1.

# Your code here:


# 3. Write a recursive function to find the maximum value in a list.
#    Example: find_max([3, 7, 2, 9, 1]) -> 9
#    Hint: Compare first element with the max of the rest.
#    Base case: a list with one item — that item is the max.

# Your code here:


# 4. Write a recursive function to flatten a nested list.
#    Example: flatten([1, [2, 3], [4, [5, 6]]]) -> [1, 2, 3, 4, 5, 6]
#    Hint: Similar to exercise 2, but build a list instead of counting.

# Your code here:


print("\nDone! Try the exercises above by writing code and re-running this file.")
