# HOW TO RUN:
#   uv run python 01_foundations/01_lists.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- LISTS ---
# A list is a collection of items stored in order.
# Think of it like a shopping list or a to-do list.
# Lists are one of the most important tools in Python.
# In ML, you will use lists (and their powerful cousin, arrays) everywhere
# to store data points, features, labels, and more.


# === CREATING LISTS ===
# You make a list using square brackets []
# Items are separated by commas

fruits = ["apple", "banana", "cherry"]
print("My fruits:", fruits)
# Output: My fruits: ['apple', 'banana', 'cherry']

# A list can hold numbers too
scores = [85, 92, 78, 95, 88]
print("Test scores:", scores)
# Output: Test scores: [85, 92, 78, 95, 88]

# A list can even mix different types (but this is less common)
mixed = ["Astha", 25, True, 3.14]
print("Mixed list:", mixed)
# Output: Mixed list: ['Astha', 25, True, 3.14]

# An empty list -- you'll use this a lot to collect results
empty = []
print("Empty list:", empty)
# Output: Empty list: []


# === INDEXING (Accessing items) ===
# Each item in a list has a position number called an "index"
# IMPORTANT: Python starts counting from 0, not 1!
#
#   Index:    0        1        2
#   Item:  "apple"  "banana"  "cherry"

fruits = ["apple", "banana", "cherry"]

print(fruits[0])   # Output: apple       (first item)
print(fruits[1])   # Output: banana      (second item)
print(fruits[2])   # Output: cherry      (third item)

# Negative indexing -- count from the end
print(fruits[-1])  # Output: cherry      (last item)
print(fruits[-2])  # Output: banana      (second to last)


# === SLICING (Getting a portion of a list) ===
# Slicing lets you grab multiple items at once
# Syntax: list[start:stop]
# start is included, stop is NOT included

numbers = [10, 20, 30, 40, 50, 60, 70]

print(numbers[1:4])   # Output: [20, 30, 40]   (index 1, 2, 3)
print(numbers[:3])    # Output: [10, 20, 30]   (from beginning to index 2)
print(numbers[4:])    # Output: [50, 60, 70]   (from index 4 to end)
print(numbers[::2])   # Output: [10, 30, 50, 70]  (every 2nd item)
print(numbers[::-1])  # Output: [70, 60, 50, 40, 30, 20, 10]  (reversed!)

# Try it yourself: What does numbers[2:5] give you?
# Think about it first, then uncomment the line below to check.
print(numbers[2:5]) 


# === MUTABILITY (Lists can be changed!) ===
# Unlike some other data types, you can change a list after creating it.
# This is called "mutability" -- the list is "mutable".

colors = ["red", "green", "blue"]
print("Before:", colors)
# Output: Before: ['red', 'green', 'blue']

colors[1] = "yellow"   # Change the second item
print("After:", colors)
# Output: After: ['red', 'yellow', 'blue']


# === LIST METHODS (Built-in actions you can do with lists) ===

# --- append: Add one item to the end ---
animals = ["cat", "dog"]
animals.append("fish")
print("After append:", animals)
# Output: After append: ['cat', 'dog', 'fish']

# --- insert: Add an item at a specific position ---
animals.insert(1, "bird")   # Insert "bird" at index 1
print("After insert:", animals)
# Output: After insert: ['cat', 'bird', 'dog', 'fish']

# --- remove: Remove the first occurrence of an item ---
animals.remove("dog")
print("After remove:", animals)
# Output: After remove: ['cat', 'bird', 'fish']

# --- pop: Remove and return an item by index (default is last) ---
last_animal = animals.pop()
print("Popped:", last_animal)
# Output: Popped: fish
print("After pop:", animals)
# Output: After pop: ['cat', 'bird']

# --- sort: Sort the list in place ---
numbers = [42, 7, 15, 3, 99]
numbers.sort()
print("Sorted:", numbers)
# Output: Sorted: [3, 7, 15, 42, 99]

# --- reverse: Reverse the list in place ---
numbers.reverse()
print("Reversed:", numbers)
# Output: Reversed: [99, 42, 15, 7, 3]

# --- len: How many items? (This is a function, not a method) ---
print("Length:", len(numbers))
# Output: Length: 5

# --- in: Check if something is in the list ---
print("Is 42 in the list?", 42 in numbers)
# Output: Is 42 in the list? True
print("Is 100 in the list?", 100 in numbers)
# Output: Is 100 in the list? False

# --- count: How many times does an item appear? ---
letters = ["a", "b", "a", "c", "a"]
print("Count of 'a':", letters.count("a"))
# Output: Count of 'a': 3

# --- extend: Add multiple items from another list ---
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list1.extend(list2)
print("Extended:", list1)
# Output: Extended: [1, 2, 3, 4, 5, 6]


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test: basic list creation
assert fruits == ["apple", "banana", "cherry"], "fruits list is wrong"
assert scores == [85, 92, 78, 95, 88], "scores list is wrong"
assert len(mixed) == 4, "mixed list should have 4 items"
assert empty == [], "empty list should be []"

# Test: indexing
assert fruits[0] == "apple", "first fruit should be apple"
assert fruits[-1] == "cherry", "last fruit should be cherry"

# Test: slicing (numbers at this point is [99, 42, 15, 7, 3] after sort+reverse)
assert numbers[1:4] == [42, 15, 7], "slice [1:4] is wrong"
assert numbers[:3] == [99, 42, 15], "slice [:3] is wrong"

# Test: list methods
assert animals == ["cat", "bird"], "animals list after operations is wrong"
assert last_animal == "fish", "popped item should be fish"
assert numbers == [99, 42, 15, 7, 3], "numbers after sort+reverse is wrong"
assert len(numbers) == 5, "numbers should have 5 items"
assert 42 in numbers, "42 should be in numbers"
assert 100 not in numbers, "100 should not be in numbers"
assert letters.count("a") == 3, "count of 'a' should be 3"
assert list1 == [1, 2, 3, 4, 5, 6], "extended list1 is wrong"

print("\nAll tests passed!")

# === EXERCISES ===
# Try these on your own below! Run the file to see if your code works.

# 1. Create a list called "my_hobbies" with at least 3 hobbies.
#    Print the list and print the first hobby.
#    Hint: my_hobbies = ["reading", ...]

my_hobbies = ["dancing", "singing", "swimming"]
print(my_hobbies)
print(my_hobbies[0])


# 2. Given the list below, use slicing to get only the middle 3 items.
#    Expected output: [30, 40, 50]
#    Hint: think about which index to start and stop at
data = [10, 20, 30, 40, 50, 60, 70]

print(data[2:5])


# 3. Start with an empty list called "shopping".
#    Add "milk", "bread", and "eggs" using append.
#    Then remove "bread".
#    Print the final list.
#    Expected output: ['milk', 'eggs']

shopping = []
shopping.append("milk")
shopping.append("bread")
shopping.append("eggs")
print(shopping)

shopping.remove("bread")

print(shopping)


# 4. Given the list below, sort it and print the smallest and largest numbers.
#    Hint: after sorting, the smallest is at index 0, the largest at index -1
grades = [78, 95, 62, 88, 71, 100, 55]

grades.sort()
print("sorted:", grades)
print(grades[0])
print(grades[-1])


# 5. (Challenge) Create two lists of your favorite foods and your friend's
#    favorite foods. Combine them into one big list using extend.
#    Then check if "pizza" is in the combined list.

my_fav = ["momos", "golgappe", "idli"]
puchku_fav = ["kadhi", "podidosa", "cheesecake"]

my_fav.extend(puchku_fav)

print("Is pizza in combined list:", 'pizza' in my_fav)
