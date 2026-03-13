# HOW TO RUN:
#   uv run python 01_foundations/02_loops_and_lists.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- LOOPS AND LISTS ---
# Now that you know lists and loops separately, let's combine them.
# Looping through lists is something you'll do ALL the time.
# In ML, you loop through datasets, predictions, training steps, etc.


# === LOOPING THROUGH A LIST WITH FOR ===
# The simplest way: "for item in list"

fruits = ["apple", "banana", "cherry"]

print("My fruits:")
for fruit in fruits:
    print(" -", fruit)
# Output:
# My fruits:
#  - apple
#  - banana
#  - cherry

# You can name the loop variable anything, but pick something meaningful
scores = [85, 92, 78, 95]
total = 0
for score in scores:
    total = total + score

print("Total:", total)        # Output: Total: 350
print("Average:", total / len(scores))  # Output: Average: 87.5


# === ENUMERATE: Get both the index AND the item ===
# Sometimes you need to know the position number too.
# enumerate() gives you both.

colors = ["red", "green", "blue", "yellow"]

print("Colors with their positions:")
for index, color in enumerate(colors):
    print(f"  Position {index}: {color}")
# Output:
# Colors with their positions:
#   Position 0: red
#   Position 1: green
#   Position 2: blue
#   Position 3: yellow

# This is much cleaner than doing it manually with a counter variable!


# === BUILDING A NEW LIST FROM AN OLD ONE ===
# A very common pattern: start empty, loop, append results

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Get only the even numbers
evens = []
for num in numbers:
    if num % 2 == 0:    # % gives remainder; even numbers have remainder 0
        evens.append(num)

print("Even numbers:", evens)
# Output: Even numbers: [2, 4, 6, 8, 10]

# Double every number
doubled = []
for num in numbers:
    doubled.append(num * 2)

print("Doubled:", doubled)
# Output: Doubled: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


# === FINDING THINGS IN A LIST ===

temperatures = [22, 25, 19, 30, 28, 17, 35, 21]

# Find the maximum temperature (without using max())
highest = temperatures[0]   # Start by assuming the first one is highest
for temp in temperatures:
    if temp > highest:
        highest = temp

print("Highest temperature:", highest)
# Output: Highest temperature: 35

# Of course, Python has a built-in for this:
print("Using max():", max(temperatures))
# Output: Using max(): 35
print("Using min():", min(temperatures))
# Output: Using min(): 17


# === WHILE LOOPS WITH LISTS ===
# Sometimes you want to keep going until a list is empty

tasks = ["email", "code", "lunch", "meeting"]
print("\nCompleting tasks:")
while len(tasks) > 0:
    current_task = tasks.pop(0)   # Remove and get the first item
    print(f"  Done: {current_task}")

print("All tasks completed!")
# Output:
# Completing tasks:
#   Done: email
#   Done: code
#   Done: lunch
#   Done: meeting
# All tasks completed!


# === NESTED LOOPS (Loop inside a loop) ===
# When you have a list of lists, you need a loop inside a loop

# A grid of numbers (like a small spreadsheet or a matrix in ML!)
grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print("Grid contents:")
for row in grid:
    for item in row:
        print(item, end=" ")   # end=" " keeps printing on the same line
    print()   # Move to next line after each row
# Output:
# Grid contents:
# 1 2 3
# 4 5 6
# 7 8 9

# Accessing a specific item in a nested list
print("Middle item:", grid[1][1])   # Row 1, Column 1
# Output: Middle item: 5


# === COMMON PATTERNS SUMMARY ===

data = [10, 20, 30, 40, 50]

# Pattern 1: Sum
total = 0
for num in data:
    total += num       # Same as total = total + num
print("Sum:", total)   # Output: Sum: 150

# Pattern 2: Count items that match a condition
big_count = 0
for num in data:
    if num >= 30:
        big_count += 1
print("Numbers >= 30:", big_count)   # Output: Numbers >= 30: 3

# Pattern 3: Transform each item
labels = []
for num in data:
    if num >= 30:
        labels.append("high")
    else:
        labels.append("low")
print("Labels:", labels)
# Output: Labels: ['low', 'low', 'high', 'high', 'high']
# (This is similar to labeling data in ML!)


# === RANGE WITH LISTS ===
# range() generates a sequence of numbers -- great for loops

# Print numbers 0 through 4
for i in range(5):
    print(i, end=" ")
print()   # Output: 0 1 2 3 4

# Use range to access list items by index
names = ["Alice", "Bob", "Charlie"]
for i in range(len(names)):
    print(f"Person {i + 1}: {names[i]}")
# Output:
# Person 1: Alice
# Person 2: Bob
# Person 3: Charlie

# But usually enumerate is cleaner for this -- prefer enumerate!


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test: looping total and average (scores = [85, 92, 78, 95], total was 350 before being overwritten)
# By this point, total = 150 (from common patterns, sum of data=[10,20,30,40,50])
assert evens == [2, 4, 6, 8, 10], "evens list is wrong"
assert doubled == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], "doubled list is wrong"

# Test: finding things
assert highest == 35, "highest temperature should be 35"
assert max(temperatures) == 35, "max() should return 35"
assert min(temperatures) == 17, "min() should return 17"

# Test: nested list / grid
assert grid[1][1] == 5, "middle item of grid should be 5"
assert grid[0] == [1, 2, 3], "first row of grid is wrong"

# Test: common patterns (data = [10, 20, 30, 40, 50])
assert data == [10, 20, 30, 40, 50], "data list should be [10, 20, 30, 40, 50]"
assert total == 150, "sum of data [10,20,30,40,50] should be 150"
assert big_count == 3, "numbers >= 30 should be 3"
assert labels == ["low", "low", "high", "high", "high"], "labels list is wrong"

print("\nAll tests passed!")

# === EXERCISES ===

# 1. Given the list of temperatures below, loop through them and
#    print whether each one is "hot" (above 30) or "cold" (30 or below).
#    Expected output:
#    25 -> cold
#    35 -> hot
#    20 -> cold
#    40 -> hot
#    30 -> cold
#    Hint: use a for loop with an if/else inside

temps = [25, 35, 20, 40, 30]

for temp in temps:
    if temp > 30:
        print(f"{temp} -> hot")
    else:
        print(f"{temp} -> cold")




# 2. Start with the list of words below. Build a new list that contains
#    only the words that have more than 4 letters.
#    Expected output: ['python', 'learning', 'great']
#    Hint: len("hello") gives you 5

words = ["I", "love", "python", "and", "learning", "is", "great"]

new_list = []
for word in words:
    if len(word) > 4:
        new_list.append(word)
print(new_list)


# 3. Use enumerate to print each item with its position,
#    but starting the count from 1 instead of 0.
#    Expected output:
#    1. math
#    2. science
#    3. history
#    4. art
#    Hint: enumerate(list, start=1) starts counting from 1

subjects = ["math", "science", "history", "art"]


for index, subject in enumerate(subjects, start=1):
    print(f"{index}. {subject}")


# 4. Given the nested list (matrix) below, calculate the sum of ALL numbers.
#    Expected output: 45
#    Hint: you need two loops, one inside the other

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

total = 0
for row in matrix:
    for item in row:
        total += item
print(total)


# 5. (Challenge) Given a list of numbers, create TWO new lists:
#    one with the positive numbers, one with the negative numbers.
#    Expected output:
#    Positive: [5, 3, 8, 1]
#    Negative: [-2, -7, -4]
#    Hint: start with two empty lists, loop once, use if/else

mixed_numbers = [5, -2, 3, -7, 8, -4, 1]

positive = []
negative = []
for num in mixed_numbers:
    if num >= 0:
        positive.append(num)
    else:
        negative.append(num)
print(f"Positive: {positive}")
print(f"Negative: {negative}")
