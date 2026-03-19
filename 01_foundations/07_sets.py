# HOW TO RUN:
#   uv run python 01_foundations/07_sets.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- SETS ---
# A set is a collection of UNIQUE items (no duplicates allowed).
# Sets are unordered -- they don't remember which item came first.
#
# Why sets matter:
# - Removing duplicates from data (very common in data cleaning)
# - Fast membership testing (checking if something is in the set)
# - Mathematical operations like union and intersection
# - In ML, useful for comparing vocabularies, unique labels, etc.


# === CREATING SETS ===
# Use curly braces {} (like dicts, but without key:value pairs)
# Or use set() to convert from a list

fruits = {"apple", "banana", "cherry"}
print("Fruits:", fruits)
# Output: Fruits: {'cherry', 'apple', 'banana'}
# Note: the ORDER might be different each time! Sets are unordered.

# Creating from a list (and removing duplicates automatically!)
numbers_list = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
numbers_set = set(numbers_list)
print("Unique numbers:", numbers_set)
# Output: Unique numbers: {1, 2, 3, 4}

# IMPORTANT: an empty set must be created with set(), not {}
# Because {} creates an empty dictionary!
empty_set = set()
empty_dict = {}
print(type(empty_set))    # Output: <class 'set'>
print(type(empty_dict))   # Output: <class 'dict'>


# === REMOVING DUPLICATES (practical use) ===

names = ["Astha", "Raj", "Priya", "Astha", "Raj", "Astha"]
unique_names = list(set(names))   # Convert to set (removes dups), back to list
print("All names:", names)
print("Unique names:", unique_names)
# Output:
# All names: ['Astha', 'Raj', 'Priya', 'Astha', 'Raj', 'Astha']
# Unique names: ['Raj', 'Astha', 'Priya']   (order may vary)

print(f"Had {len(names)} names, {len(unique_names)} are unique")
# Output: Had 6 names, 3 are unique


# === ADDING AND REMOVING ITEMS ===

colors = {"red", "green", "blue"}

# Add an item
colors.add("yellow")
print("After add:", colors)
# Output: After add: {'yellow', 'red', 'green', 'blue'}  (order may vary)

# Add a duplicate -- nothing happens! (no error, just ignored)
colors.add("red")
print("After adding duplicate:", colors)
# Output: After adding duplicate: {'yellow', 'red', 'green', 'blue'}

# Remove an item
colors.remove("green")     # Crashes if item doesn't exist
print("After remove:", colors)

colors.discard("purple")   # Safe -- no crash if item doesn't exist
print("After discard:", colors)


# === CHECKING MEMBERSHIP (Very fast!) ===
# The "in" keyword works with sets and is MUCH faster than lists
# for large collections.

big_set = set(range(1000000))   # A set with numbers 0 to 999999

print(999999 in big_set)   # Output: True   (this is super fast)
print(1000001 in big_set)  # Output: False


# === SET OPERATIONS ===
# These come from math (set theory) and are very useful!

python_students = {"Astha", "Raj", "Priya", "Neha"}
ml_students = {"Raj", "Neha", "Amit", "Sara"}

# --- Union: ALL students from BOTH groups (no duplicates) ---
all_students = python_students | ml_students    # or: .union()
print("All students:", all_students)
# Output: All students: {'Astha', 'Raj', 'Priya', 'Neha', 'Amit', 'Sara'}

# --- Intersection: Students in BOTH groups ---
both = python_students & ml_students    # or: .intersection()
print("In both:", both)
# Output: In both: {'Raj', 'Neha'}

# --- Difference: In Python but NOT in ML ---
only_python = python_students - ml_students    # or: .difference()
print("Only Python:", only_python)
# Output: Only Python: {'Astha', 'Priya'}

only_ml = ml_students - python_students
print("Only ML:", only_ml)
# Output: Only ML: {'Amit', 'Sara'}

# --- Symmetric Difference: In one OR the other, but NOT both ---
exclusive = python_students ^ ml_students    # or: .symmetric_difference()
print("Exclusive:", exclusive)
# Output: Exclusive: {'Astha', 'Priya', 'Amit', 'Sara'}


# === SUBSET AND SUPERSET ===

small = {1, 2, 3}
big = {1, 2, 3, 4, 5}

print("Is small a subset of big?", small.issubset(big))
# Output: Is small a subset of big? True

print("Is big a superset of small?", big.issuperset(small))
# Output: Is big a superset of small? True

print("Do they share any items?", not small.isdisjoint(big))
# Output: Do they share any items? True


# === LOOPING THROUGH SETS ===
# You can loop, but remember: order is not guaranteed

languages = {"Python", "JavaScript", "Rust", "Go"}
print("Languages I know:")
for lang in languages:
    print(f"  - {lang}")
# Output (order may vary):
# Languages I know:
#   - Go
#   - Python
#   - Rust
#   - JavaScript


# === WHEN TO USE SETS VS LISTS VS TUPLES ===

print("\n--- When to use what? ---")

# LIST: Ordered, allows duplicates, mutable
#   Use when order matters and you might have repeats
#   Example: a sequence of events, data points in order

# TUPLE: Ordered, allows duplicates, immutable
#   Use when data shouldn't change
#   Example: coordinates (x, y), function return values

# SET: Unordered, NO duplicates, mutable
#   Use when you need uniqueness or fast lookups
#   Example: unique words in a text, tags, categories

# DICTIONARY: Key-value pairs, keys are unique
#   Use when you want to look up values by a name/key
#   Example: word counts, configuration settings

print("List:  ordered, duplicates OK, changeable")
print("Tuple: ordered, duplicates OK, NOT changeable")
print("Set:   unordered, NO duplicates, changeable")
print("Dict:  key-value pairs, keys unique, changeable")


# === FROZEN SETS (bonus) ===
# A frozenset is an immutable set -- you can't add or remove items.
# You'd use it when you need a set that can be used as a dictionary key.

frozen = frozenset([1, 2, 3])
print("\nFrozen set:", frozen)
# Output: Frozen set: frozenset({1, 2, 3})
# frozen.add(4)   # This would CRASH! Can't change a frozenset.


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test: creating sets
assert numbers_set == {1, 2, 3, 4}, "unique numbers from list should be {1, 2, 3, 4}"
assert type(empty_set) == set, "empty_set should be type set"
assert type(empty_dict) == dict, "empty_dict should be type dict"

# Test: removing duplicates
assert len(unique_names) == 3, "there should be 3 unique names"
assert set(unique_names) == {"Astha", "Raj", "Priya"}, "unique names set is wrong"

# Test: adding and removing (colors set after operations)
# After add("yellow"), add("red" duplicate), remove("green"), discard("purple"):
assert "yellow" in colors, "yellow should be in colors after add"
assert "green" not in colors, "green should have been removed from colors"
assert "red" in colors, "red should still be in colors"
assert "purple" not in colors, "purple was never in colors"

# Test: membership
assert 999999 in big_set, "999999 should be in big_set"
assert 1000001 not in big_set, "1000001 should not be in big_set"

# Test: set operations
assert python_students | ml_students == {"Astha", "Raj", "Priya", "Neha", "Amit", "Sara"}, "union is wrong"
assert python_students & ml_students == {"Raj", "Neha"}, "intersection is wrong"
assert python_students - ml_students == {"Astha", "Priya"}, "difference (python - ml) is wrong"
assert ml_students - python_students == {"Amit", "Sara"}, "difference (ml - python) is wrong"
assert python_students ^ ml_students == {"Astha", "Priya", "Amit", "Sara"}, "symmetric difference is wrong"

# Test: subset and superset
assert small.issubset(big), "small should be a subset of big"
assert big.issuperset(small), "big should be a superset of small"
assert not small.isdisjoint(big), "small and big share items, not disjoint"

# Test: frozenset
assert frozen == frozenset({1, 2, 3}), "frozen should equal frozenset({1, 2, 3})"
assert type(frozen) == frozenset, "frozen should be of type frozenset"

print("\nAll tests passed!")

# === EXERCISES ===

# 1. Given the list below, find how many unique numbers there are.
#    Expected output: 5 unique numbers
#    Hint: convert to a set, then use len()

numbers = [4, 7, 2, 4, 9, 7, 1, 4, 2, 9]


numbers_set = set(numbers)
unique_numbers = len(numbers_set)
print(f"{unique_numbers} unique numbers")


# 2. Given two lists of friends below, find:
#    a) All friends (combined, no duplicates)
#    b) Friends you have in common
#    c) Friends only you have (not in the other list)
#    Hint: convert lists to sets first, then use set operations

my_friends = ["Alice", "Bob", "Charlie", "Diana"]
your_friends = ["Bob", "Diana", "Eve", "Frank"]


my_friends_set = set(my_friends)
your_friends_set = set(your_friends)
combined_friends = my_friends_set | your_friends_set
print(combined_friends)
common_friends = my_friends_set & your_friends_set
print(common_friends)
only_my_friends = my_friends_set - your_friends_set
print(only_my_friends)



# 3. Given a sentence, find all the unique words in it.
#    Expected output for the sentence below: 5 unique words
#    Hint: split the sentence, convert to set

sentence = "the cat and the dog and the bird"

words = sentence.split()
sentence_set = set(words)
unique_words = len(sentence_set)
print(f"{unique_words} unique words")






# 4. Write code to check if all items in list_a appear in list_b.
#    Expected output: True (because 1, 2, 3 are all in list_b)
#    Hint: use issubset()

list_a = [1, 2, 3]
list_b = [1, 2, 3, 4, 5, 6]

set_a = set(list_a)
set_b = set(list_b)
print(set_a.issubset(set_b))


# 5. (Challenge) Given a list of email addresses, find the duplicates.
#    Expected output: {'b@mail.com'}
#    Hint: compare the set of all emails with the count of each email,
#    or loop and check if you've seen it before

emails = ["a@mail.com", "b@mail.com", "c@mail.com", "b@mail.com", "d@mail.com"]


seen = set()
duplicate = set()
for email in emails:
    if email in seen:
        duplicate.add(email)
    else:
        seen.add(email)

print(duplicate)

