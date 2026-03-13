# HOW TO RUN:
#   uv run python 02_programming/01_file_handling.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- FILE HANDLING ---
# Files let you save data permanently (not just in memory).
# When your program ends, variables disappear. Files stay.
# In ML, you'll load datasets from files (CSV, text, JSON) all the time.
# This lesson teaches you how to read from and write to files using Python.


# === CONCEPT 1: WRITING TO A FILE ===
# open() creates or opens a file.
# "w" means "write mode" — it creates a new file (or erases an old one).
# Always close the file when you're done, or use "with" (shown below).

print("=" * 50)
print("CONCEPT 1: Writing to a file")
print("=" * 50)

# Let's create a small sample file to use in this lesson
with open("sample_data.txt", "w") as f:
    f.write("Alice,85\n")
    f.write("Bob,92\n")
    f.write("Charlie,78\n")
    f.write("Diana,95\n")
    f.write("Eve,88\n")

print("Created sample_data.txt with student scores!")
# The file now contains:
# Alice,85
# Bob,92
# Charlie,78
# Diana,95
# Eve,88


# === CONCEPT 2: READING AN ENTIRE FILE ===
# "r" means "read mode" — the file must already exist.
# .read() gives you the whole file as one big string.

print("\n" + "=" * 50)
print("CONCEPT 2: Reading an entire file")
print("=" * 50)

with open("sample_data.txt", "r") as f:
    content = f.read()

print("Entire file content:")
print(content)
# Output:
# Alice,85
# Bob,92
# Charlie,78
# Diana,95
# Eve,88


# === CONCEPT 3: READING LINE BY LINE ===
# .readlines() gives you a list where each item is one line.
# Each line has a "\n" (newline) at the end — use .strip() to remove it.

print("=" * 50)
print("CONCEPT 3: Reading line by line")
print("=" * 50)

with open("sample_data.txt", "r") as f:
    lines = f.readlines()

print("Raw lines (notice the \\n at the end):")
print(lines)
# Output: ['Alice,85\n', 'Bob,92\n', 'Charlie,78\n', 'Diana,95\n', 'Eve,88\n']

print("\nCleaned lines:")
for line in lines:
    clean = line.strip()      # removes the newline character
    print(clean)
# Output:
# Alice,85
# Bob,92
# Charlie,78
# Diana,95
# Eve,88


# === CONCEPT 4: PARSING FILE DATA ===
# "Parsing" means breaking raw text into useful pieces.
# Since our file has "name,score" on each line, we can split by comma.

print("\n" + "=" * 50)
print("CONCEPT 4: Parsing file data")
print("=" * 50)

scores = {}  # empty dictionary to store name -> score

with open("sample_data.txt", "r") as f:
    for line in f:                        # you can loop over a file directly!
        line = line.strip()               # remove newline
        parts = line.split(",")           # split "Alice,85" into ["Alice", "85"]
        name = parts[0]
        score = int(parts[1])             # convert "85" to the number 85
        scores[name] = score

print("Parsed scores dictionary:")
print(scores)
# Output: {'Alice': 85, 'Bob': 92, 'Charlie': 78, 'Diana': 95, 'Eve': 88}

print(f"Bob's score: {scores['Bob']}")
# Output: Bob's score: 92


# === CONCEPT 5: APPENDING TO A FILE ===
# "a" means "append mode" — adds to the end without erasing what's there.

print("\n" + "=" * 50)
print("CONCEPT 5: Appending to a file")
print("=" * 50)

with open("sample_data.txt", "a") as f:
    f.write("Frank,91\n")

print("Added Frank to the file!")

# Let's verify by reading it back
with open("sample_data.txt", "r") as f:
    print(f.read())
# Output now includes Frank at the end


# === CONCEPT 6: WRITING MULTIPLE LINES ===
# .writelines() writes a list of strings. You must add "\n" yourself.

print("=" * 50)
print("CONCEPT 6: Writing multiple lines")
print("=" * 50)

new_students = ["Grace,87\n", "Hank,73\n", "Ivy,96\n"]

with open("more_students.txt", "w") as f:
    f.writelines(new_students)

print("Created more_students.txt")

with open("more_students.txt", "r") as f:
    print(f.read())
# Output:
# Grace,87
# Hank,73
# Ivy,96


# === CONCEPT 7: CHECKING IF A FILE EXISTS ===
# Before reading a file, it's smart to check if it exists.
# We use the "os" module for this.

print("=" * 50)
print("CONCEPT 7: Checking if a file exists")
print("=" * 50)

import os

print(f"sample_data.txt exists? {os.path.exists('sample_data.txt')}")
# Output: sample_data.txt exists? True

print(f"ghost_file.txt exists? {os.path.exists('ghost_file.txt')}")
# Output: ghost_file.txt exists? False


# === CLEANUP ===
# Remove the files we created so we don't leave clutter
os.remove("sample_data.txt")
os.remove("more_students.txt")
print("\nCleaned up temporary files.")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

import os as _os

# Test 1: Write a file and verify its contents can be read back
with open("_test_file.txt", "w") as _f:
    _f.write("TestName,99\n")
    _f.write("OtherName,55\n")

with open("_test_file.txt", "r") as _f:
    _test_content = _f.read()

assert "TestName,99" in _test_content, "File write/read failed"
assert "OtherName,55" in _test_content, "File write/read failed"

# Test 2: readlines() returns correct number of lines
with open("_test_file.txt", "r") as _f:
    _test_lines = _f.readlines()

assert len(_test_lines) == 2, "readlines() should return 2 lines"
assert _test_lines[0].strip() == "TestName,99", "First line wrong"

# Test 3: Parsing name,score format into a dict works correctly
_test_scores = {}
with open("_test_file.txt", "r") as _f:
    for _line in _f:
        _parts = _line.strip().split(",")
        _test_scores[_parts[0]] = int(_parts[1])

assert _test_scores["TestName"] == 99, "Parsed score should be 99"
assert _test_scores["OtherName"] == 55, "Parsed score should be 55"

# Test 4: Append mode adds to the file without erasing it
with open("_test_file.txt", "a") as _f:
    _f.write("NewPerson,77\n")

with open("_test_file.txt", "r") as _f:
    _appended = _f.read()

assert "TestName,99" in _appended, "Original content should still be there"
assert "NewPerson,77" in _appended, "Appended content should be present"

# Test 5: os.path.exists() returns True for existing file, False for missing
assert _os.path.exists("_test_file.txt") == True, "File should exist"
assert _os.path.exists("_ghost_file_xyz.txt") == False, "Ghost file should not exist"

# Cleanup test file
_os.remove("_test_file.txt")
assert _os.path.exists("_test_file.txt") == False, "File should be deleted"

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Create a file called "my_favorites.txt" with 3 lines:
#    your favorite food, color, and animal (one per line).
#    Then read the file and print each line with a number:
#      1. pizza
#      2. blue
#      3. cat
#    Hint: use enumerate() with start=1 in your loop

# YOUR CODE HERE:


# 2. Create a file "numbers.txt" with the numbers 1 through 10 (one per line).
#    Then read the file, convert each line to an integer, and print the sum.
#    Expected output: Sum = 55
#    Hint: int(line.strip()) converts a line to a number

# YOUR CODE HERE:


# 3. Create a file "shopping.txt" with some items. Then write a program that:
#    - Reads the file
#    - Asks the user for a new item (use input())
#    - Appends that item to the file
#    - Reads and prints the updated file
#    Hint: use "a" mode to append

# YOUR CODE HERE:
