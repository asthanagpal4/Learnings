# HOW TO RUN:
#   uv run python 01_foundations/06_dictionaries.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- DICTIONARIES ---
# A dictionary stores data as key-value pairs.
# Think of it like a real dictionary: you look up a WORD (the key)
# to find its DEFINITION (the value).
#
# Dictionaries are one of Python's most powerful data structures.
# In ML, they're used for: configuration settings, word-to-number mappings
# (vocabularies), storing results, JSON data, and much more.


# === CREATING DICTIONARIES ===
# Use curly braces {} with key: value pairs

person = {
    "name": "Astha",
    "age": 25,
    "city": "Delhi"
}
print(person)
# Output: {'name': 'Astha', 'age': 25, 'city': 'Delhi'}

# An empty dictionary
empty = {}

# Keys can be strings, numbers, or tuples (anything immutable)
# Values can be ANYTHING


# === ACCESSING VALUES ===
# Use the key in square brackets to get the value

print(person["name"])    # Output: Astha
print(person["age"])     # Output: 25

# If the key doesn't exist, you get an error!
# print(person["email"])   # This would CRASH! KeyError!

# Safer way: use .get() -- returns None if key doesn't exist
print(person.get("email"))          # Output: None
print(person.get("email", "N/A"))   # Output: N/A  (custom default)


# === ADDING AND CHANGING VALUES ===
# Dictionaries are mutable -- you can change them!

person["email"] = "astha@example.com"   # Add new key-value pair
print(person)
# Output: {'name': 'Astha', 'age': 25, 'city': 'Delhi', 'email': 'astha@example.com'}

person["age"] = 26    # Change an existing value
print("Updated age:", person["age"])
# Output: Updated age: 26


# === REMOVING ITEMS ===

# del removes a key-value pair
del person["email"]
print("After del:", person)
# Output: After del: {'name': 'Astha', 'age': 26, 'city': 'Delhi'}

# pop removes and returns the value
age = person.pop("age")
print("Popped age:", age)
print("After pop:", person)
# Output:
# Popped age: 26
# After pop: {'name': 'Astha', 'city': 'Delhi'}


# === CHECKING IF A KEY EXISTS ===

person = {"name": "Astha", "age": 25, "city": "Delhi"}

print("name" in person)     # Output: True
print("email" in person)    # Output: False

# Use this to avoid errors
if "age" in person:
    print(f"Age is {person['age']}")
# Output: Age is 25


# === LOOPING THROUGH DICTIONARIES ===

scores = {"math": 95, "science": 88, "english": 72, "history": 84}

# Loop through keys (default)
print("Subjects:")
for subject in scores:
    print(f"  {subject}")
# Output:
# Subjects:
#   math
#   science
#   english
#   history

# Loop through values
print("Scores:")
for score in scores.values():
    print(f"  {score}")
# Output:
# Scores:
#   95
#   88
#   72
#   84

# Loop through both keys and values (most useful!)
print("Report card:")
for subject, score in scores.items():
    print(f"  {subject}: {score}")
# Output:
# Report card:
#   math: 95
#   science: 88
#   english: 72
#   history: 84


# === USEFUL DICTIONARY METHODS ===

info = {"name": "Astha", "age": 25}

# Get all keys, values, or items
print("Keys:", list(info.keys()))      # Output: Keys: ['name', 'age']
print("Values:", list(info.values()))  # Output: Values: ['Astha', 25]
print("Items:", list(info.items()))    # Output: Items: [('name', 'Astha'), ('age', 25)]

# Number of key-value pairs
print("Length:", len(info))   # Output: Length: 2

# Update with another dictionary (merge)
extra = {"city": "Delhi", "hobby": "reading"}
info.update(extra)
print("Updated:", info)
# Output: Updated: {'name': 'Astha', 'age': 25, 'city': 'Delhi', 'hobby': 'reading'}


# === DICTIONARIES WITH LISTS AS VALUES ===
# Values can be lists -- this is very common!

grades = {
    "Astha": [85, 90, 78],
    "Raj": [92, 88, 95],
    "Priya": [70, 75, 80]
}

# Access a specific student's grades
print("Astha's grades:", grades["Astha"])
# Output: Astha's grades: [85, 90, 78]

# Calculate average for each student
print("\nAverages:")
for student, student_grades in grades.items():
    avg = sum(student_grades) / len(student_grades)
    print(f"  {student}: {avg:.1f}")
# Output:
# Averages:
#   Astha: 84.3
#   Raj: 91.7
#   Priya: 75.0


# === NESTED DICTIONARIES ===
# A dictionary inside a dictionary

users = {
    "user1": {
        "name": "Astha",
        "age": 25,
        "skills": ["Python", "ML"]
    },
    "user2": {
        "name": "Raj",
        "age": 28,
        "skills": ["Java", "SQL"]
    }
}

print("User 1 name:", users["user1"]["name"])
# Output: User 1 name: Astha

print("User 1 skills:", users["user1"]["skills"])
# Output: User 1 skills: ['Python', 'ML']


# === COUNTING WITH DICTIONARIES ===
# One of the most practical uses -- counting how often things appear.
# This pattern is fundamental in NLP (Natural Language Processing)!

sentence = "the cat sat on the mat the cat"
words = sentence.split()

word_count = {}
for word in words:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

print("Word frequencies:", word_count)
# Output: Word frequencies: {'the': 3, 'cat': 2, 'sat': 1, 'on': 1, 'mat': 1}

# Shorter way using .get()
word_count2 = {}
for word in words:
    word_count2[word] = word_count2.get(word, 0) + 1

print("Same result:", word_count2)
# Output: Same result: {'the': 3, 'cat': 2, 'sat': 1, 'on': 1, 'mat': 1}


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test: accessing values
assert person["name"] == "Astha", "person['name'] should be 'Astha'"
assert person["age"] == 25, "person['age'] should be 25"
assert person.get("email") is None, "person.get('email') should be None"
assert person.get("email", "N/A") == "N/A", "person.get('email', 'N/A') should be 'N/A'"

# Test: checking key existence
assert "name" in person, "'name' should be in person"
assert "email" not in person, "'email' should not be in person"

# Test: dict methods
assert list(info.keys()) == ["name", "age", "city", "hobby"], "info.keys() is wrong"
assert len(info) == 4, "info should have 4 keys after update"
assert info["city"] == "Delhi", "info['city'] should be 'Delhi'"
assert info["hobby"] == "reading", "info['hobby'] should be 'reading'"

# Test: dict with list values
assert grades["Astha"] == [85, 90, 78], "Astha's grades are wrong"
assert abs(sum(grades["Raj"]) / len(grades["Raj"]) - 91.67) < 0.01, "Raj's average should be ~91.67"

# Test: nested dict
assert users["user1"]["name"] == "Astha", "user1 name should be 'Astha'"
assert users["user1"]["skills"] == ["Python", "ML"], "user1 skills are wrong"
assert users["user2"]["age"] == 28, "user2 age should be 28"

# Test: counting with dict
assert word_count["the"] == 3, "count of 'the' should be 3"
assert word_count["cat"] == 2, "count of 'cat' should be 2"
assert word_count["sat"] == 1, "count of 'sat' should be 1"
assert word_count2 == word_count, "word_count2 should equal word_count"

print("\nAll tests passed!")

# === EXERCISES ===

# 1. Create a dictionary called "movie" with keys: "title", "year", "rating".
#    Fill in your favorite movie's info. Print each value.
#    Example output:
#    Title: Inception
#    Year: 2010
#    Rating: 8.8

# Write your code here:


# 2. Given the dictionary below, add a new key "country" with value "India",
#    change the "age" to 26, and remove the "hobby" key.
#    Expected output: {'name': 'Astha', 'age': 26, 'city': 'Delhi', 'country': 'India'}

profile = {"name": "Astha", "age": 25, "city": "Delhi", "hobby": "reading"}

# Write your code here:


# 3. Given the scores dictionary below, loop through it and print
#    only the subjects where the score is above 85.
#    Expected output:
#    math: 95
#    science: 88
#    Hint: use .items() and an if statement

scores = {"math": 95, "science": 88, "english": 72, "history": 84}

# Write your code here:


# 4. Count how many times each letter appears in the word below.
#    Expected output (order may vary):
#    {'m': 1, 'i': 4, 's': 4, 'p': 2}
#    Hint: same counting pattern as the word counter example above

word = "mississippi"

# Write your code here:


# 5. (Challenge) Given a list of (item, price) tuples, create a dictionary
#    from them, then find the most expensive item.
#    Expected output: laptop costs 80000
#    Hint: dict() can convert a list of tuples to a dictionary

products = [("phone", 15000), ("laptop", 80000), ("tablet", 30000), ("watch", 5000)]

# Write your code here:
