# HOW TO RUN:
#   uv run python 02_programming/06_modules.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- MODULES & IMPORTS ---
# A module is just a Python file that contains functions, classes, or variables.
# Importing lets you use code from other files or from Python's built-in library.
# In ML, you'll import modules constantly: numpy, pandas, torch, etc.
# This lesson shows how importing works and how to organize your own code.


# === CONCEPT 1: IMPORTING BUILT-IN MODULES ===
# Python comes with many useful modules. You just need to import them.

print("=" * 50)
print("CONCEPT 1: Importing built-in modules")
print("=" * 50)

# Import the entire module
import math

print(f"Pi: {math.pi}")                      # Output: Pi: 3.141592653589793
print(f"Square root of 16: {math.sqrt(16)}") # Output: Square root of 16: 4.0
print(f"2 to the power 10: {math.pow(2, 10)}")  # Output: 2 to the power 10: 1024.0


# === CONCEPT 2: DIFFERENT WAYS TO IMPORT ===
# There are several ways to import, each with pros and cons.

print("\n" + "=" * 50)
print("CONCEPT 2: Different import styles")
print("=" * 50)

# Style 1: import the whole module
import random
print(f"Random number: {random.randint(1, 100)}")

# Style 2: import specific things from a module
from datetime import datetime, date
now = datetime.now()
print(f"Current time: {now.strftime('%Y-%m-%d %H:%M')}")

today = date.today()
print(f"Today's date: {today}")

# Style 3: import with a nickname (alias)
# This is VERY common in ML!
import statistics as stats
data = [85, 92, 78, 95, 88]
print(f"Mean: {stats.mean(data)}")       # Output: Mean: 87.6
print(f"Median: {stats.median(data)}")   # Output: Median: 88

# Note: In ML, you'll write things like:
#   import numpy as np
#   import pandas as pd
#   import torch


# === CONCEPT 3: USEFUL BUILT-IN MODULES ===
# Let's explore some modules you'll use often.

print("\n" + "=" * 50)
print("CONCEPT 3: Useful built-in modules")
print("=" * 50)

# --- os: working with files and directories ---
import os
print(f"Current directory: {os.getcwd()}")
print(f"Files here: {os.listdir('.')[:5]}...")  # first 5 files

# --- json: reading and writing JSON data ---
import json

# Convert a Python dictionary to JSON (text format)
person = {"name": "Astha", "age": 25, "skills": ["Python", "ML"]}
json_text = json.dumps(person, indent=2)  # "dumps" = dump to string
print(f"\nJSON format:\n{json_text}")

# Convert JSON text back to a Python dictionary
parsed = json.loads(json_text)  # "loads" = load from string
print(f"Back to dict: {parsed['name']}")
# Output: Back to dict: Astha

# --- collections: special data structures ---
from collections import Counter

words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
word_counts = Counter(words)
print(f"\nWord counts: {word_counts}")
# Output: Word counts: Counter({'apple': 3, 'banana': 2, 'cherry': 1})
print(f"Most common: {word_counts.most_common(2)}")
# Output: Most common: [('apple', 3), ('banana', 2)]


# === CONCEPT 4: THE RANDOM MODULE ===
# Super useful for ML: shuffling data, random sampling, etc.

print("\n" + "=" * 50)
print("CONCEPT 4: The random module")
print("=" * 50)

import random

# Set a seed for reproducibility (same "random" numbers every time)
random.seed(42)

# Random integer between 1 and 10
print(f"Random int: {random.randint(1, 10)}")

# Random float between 0 and 1
print(f"Random float: {round(random.random(), 4)}")

# Pick a random item from a list
colors = ["red", "blue", "green", "yellow"]
print(f"Random color: {random.choice(colors)}")

# Shuffle a list (changes it in place)
numbers = [1, 2, 3, 4, 5]
random.shuffle(numbers)
print(f"Shuffled: {numbers}")

# Pick multiple random items
sample = random.sample(range(1, 50), 5)
print(f"5 random numbers from 1-49: {sample}")


# === CONCEPT 5: THE __name__ VARIABLE ===
# Every Python file has a special variable called __name__.
# If you RUN the file directly: __name__ is "__main__"
# If you IMPORT the file from another file: __name__ is the filename
# This lets you write code that only runs when the file is executed directly.

print("\n" + "=" * 50)
print("CONCEPT 5: __name__ == '__main__'")
print("=" * 50)

print(f"This file's __name__ is: {__name__}")
# Output: This file's __name__ is: __main__
# (Because we're running this file directly)

# The common pattern:
# if __name__ == "__main__":
#     # This code only runs when you execute the file directly
#     # It does NOT run when someone imports this file
#     run_tests()
#     main()

# Why this matters:
# When you create a module with useful functions, you might also want
# to test those functions. The if __name__ block lets you include tests
# that only run when you execute the file, not when you import it.

# Example of how it works:
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

if __name__ == "__main__":
    # These tests run when you execute this file
    # But they won't run if another file does: import 06_modules
    print(f"Testing greet: {greet('Astha')}")
    print(f"Testing add: {add(3, 4)}")
# Output:
# Testing greet: Hello, Astha!
# Testing add: 7


# === CONCEPT 6: CREATING YOUR OWN MODULE ===
# Any .py file is a module! If you create "helpers.py" with functions,
# another file can do "import helpers" and use them.

print("\n" + "=" * 50)
print("CONCEPT 6: How modules work in practice")
print("=" * 50)

# Let's simulate creating and using a module.
# In real life, you'd put this in a separate file.

# --- Imagine this is in a file called "math_utils.py" ---
# def square(x):
#     return x ** 2
#
# def cube(x):
#     return x ** 3
#
# PI = 3.14159
# --- End of math_utils.py ---

# Then in your main file, you'd write:
# import math_utils
# print(math_utils.square(5))      -> 25
# print(math_utils.PI)             -> 3.14159

# Or:
# from math_utils import square, cube
# print(square(5))                 -> 25

# For now, let's demonstrate with the built-in approach:
print("Module organization tips:")
print("  1. Put related functions in one file")
print("  2. Give the file a clear, descriptive name")
print("  3. Use if __name__ == '__main__' for testing")
print("  4. Import only what you need")


# === CONCEPT 7: COMMON IMPORT PATTERNS IN ML ===
# Here's a preview of imports you'll see in ML code.
# (You can't run these yet without installing the packages,
#  but it's good to see the patterns.)

print("\n" + "=" * 50)
print("CONCEPT 7: ML import patterns (preview)")
print("=" * 50)

# These are the most common ML imports you'll encounter:
ml_imports = """
# Data handling
import numpy as np              # numerical computing
import pandas as pd             # data tables (like spreadsheets)

# Visualization
import matplotlib.pyplot as plt # plotting graphs

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Deep Learning
import torch                    # PyTorch
import torch.nn as nn           # neural network layers
from torch.utils.data import DataLoader

# Notice the patterns:
# - "import X as Y" gives short names for long module names
# - "from X import Y" imports specific things you need
"""
print(ml_imports)


# === CONCEPT 8: AVOIDING COMMON MISTAKES ===

print("=" * 50)
print("CONCEPT 8: Common mistakes to avoid")
print("=" * 50)

# Mistake 1: "from module import *" — imports EVERYTHING
# Don't do this! It pollutes your namespace and you can't tell
# where things came from.
# BAD:  from math import *
# GOOD: from math import sqrt, pi

# Mistake 2: Naming your file the same as a module
# If you name your file "random.py" and try to import random,
# Python will import YOUR file instead of the built-in one!
# Don't name files: random.py, math.py, os.py, etc.

# Mistake 3: Circular imports
# If file A imports file B, and file B imports file A, things break.
# Solution: reorganize your code so the dependency goes one way.

print("Key rules:")
print("  1. Never do 'from X import *'")
print("  2. Never name your file the same as a built-in module")
print("  3. Avoid circular imports")
print("  4. Import at the top of your file")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test 1: math module basics
import math as _math
assert round(_math.pi, 5) == 3.14159, "math.pi should start with 3.14159"
assert _math.sqrt(25) == 5.0, "sqrt(25) should be 5.0"
assert _math.pow(2, 8) == 256.0, "2^8 should be 256.0"

# Test 2: greet and add functions defined in this file
assert greet("Astha") == "Hello, Astha!", "greet() output wrong"
assert greet("World") == "Hello, World!", "greet() output wrong"
assert add(3, 4) == 7, "add(3, 4) should return 7"
assert add(0, 0) == 0, "add(0, 0) should return 0"
assert add(-1, 1) == 0, "add(-1, 1) should return 0"

# Test 3: json module — round-trip encode/decode
import json as _json
_data = {"name": "Astha", "score": 95}
_encoded = _json.dumps(_data)
_decoded = _json.loads(_encoded)
assert _decoded["name"] == "Astha", "JSON round-trip name wrong"
assert _decoded["score"] == 95, "JSON round-trip score wrong"

# Test 4: statistics module — mean and median
import statistics as _stats
_nums = [10, 20, 30, 40, 50]
assert _stats.mean(_nums) == 30, "Mean of [10,20,30,40,50] should be 30"
assert _stats.median(_nums) == 30, "Median of [10,20,30,40,50] should be 30"

# Test 5: collections.Counter — counts items correctly
from collections import Counter as _Counter
_c = _Counter(["a", "b", "a", "a", "b"])
assert _c["a"] == 3, "Counter should count 'a' as 3"
assert _c["b"] == 2, "Counter should count 'b' as 2"
assert _c.most_common(1)[0][0] == "a", "Most common should be 'a'"

# Test 6: __name__ is '__main__' when run directly
assert __name__ == "__main__", "__name__ should be '__main__' when running directly"

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Use the math module to:
#    - Print the value of e (Euler's number)
#    - Calculate the ceiling of 4.3 (round up)
#    - Calculate the floor of 4.7 (round down)
#    - Calculate log base 2 of 256
#    Expected: e=2.718..., ceil=5, floor=4, log2=8.0
#    Hint: math.e, math.ceil(), math.floor(), math.log2()

# YOUR CODE HERE:


# 2. Use the random module (with seed=100) to:
#    - Generate a list of 10 random integers between 1 and 100
#    - Print the list, its min, max, and average
#    Hint: use a list comprehension with random.randint()

# YOUR CODE HERE:


# 3. Use the collections.Counter to:
#    - Count the characters in "mississippi"
#    - Print the 3 most common characters
#    Expected most common: [('s', 4), ('i', 4), ('p', 2)]
#    Hint: Counter("mississippi")

# YOUR CODE HERE:


# 4. Use the json module to:
#    - Create a dictionary with your name, favorite_language, and a list of hobbies
#    - Convert it to a JSON string and print it nicely (with indent=2)
#    - Convert it back to a dictionary and print just the hobbies list
#    Hint: json.dumps() and json.loads()

# YOUR CODE HERE:
