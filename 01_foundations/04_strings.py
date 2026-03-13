# HOW TO RUN:
#   uv run python 01_foundations/04_strings.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- STRINGS ---
# Strings are sequences of characters (letters, numbers, symbols).
# You've been using them already with print()!
# Strings are EVERYWHERE in programming, especially in ML for
# processing text data (like training data for language models).


# === CREATING STRINGS ===
# You can use single quotes or double quotes -- both work the same

name = "Astha"
greeting = 'Hello!'
print(name, greeting)
# Output: Astha Hello!

# For text with quotes inside, use the other type of quote
sentence = "It's a beautiful day"
quote = 'She said "hello"'
print(sentence)   # Output: It's a beautiful day
print(quote)      # Output: She said "hello"

# Multi-line strings use triple quotes
poem = """Roses are red,
Violets are blue,
Python is fun,
And so are you."""
print(poem)
# Output:
# Roses are red,
# Violets are blue,
# Python is fun,
# And so are you.


# === STRINGS ARE SEQUENCES (like lists!) ===
# You can index and slice strings just like lists

word = "Python"
print(word[0])     # Output: P
print(word[-1])    # Output: n
print(word[0:3])   # Output: Pyt
print(word[::-1])  # Output: nohtyP  (reversed!)

print(len(word))   # Output: 6


# === IMMUTABILITY: Strings CANNOT be changed ===
# This is a big difference from lists!
# Lists are mutable (changeable), strings are immutable (fixed).

name = "Astha"
# name[0] = "a"   # This would CRASH with an error!

# Instead, you create a NEW string
new_name = "a" + name[1:]
print(new_name)    # Output: astha

# Compare with lists:
my_list = [1, 2, 3]
my_list[0] = 99    # This works fine! Lists are mutable.
print(my_list)     # Output: [99, 2, 3]


# === USEFUL STRING METHODS ===
# Methods are actions you can do with strings.
# They always return a NEW string (because strings can't be changed).

text = "Hello, World!"

# --- Changing case ---
print(text.upper())       # Output: HELLO, WORLD!
print(text.lower())       # Output: hello, world!
print(text.title())       # Output: Hello, World!
print("astha".capitalize())  # Output: Astha

# --- Checking content ---
print("hello".isalpha())    # Output: True   (only letters?)
print("hello123".isalpha()) # Output: False
print("12345".isdigit())    # Output: True   (only digits?)
print(text.startswith("Hello"))  # Output: True
print(text.endswith("!"))        # Output: True

# --- Finding and replacing ---
sentence = "I love Python and Python loves me"

print(sentence.find("Python"))       # Output: 7   (position of first "Python")
print(sentence.count("Python"))      # Output: 2   (how many times?)
print(sentence.replace("Python", "ML"))
# Output: I love ML and ML loves me


# === STRIP: Remove extra whitespace ===
# Super useful for cleaning messy data!

messy = "   hello world   "
print(f"Before: '{messy}'")
print(f"After strip: '{messy.strip()}'")
print(f"After lstrip: '{messy.lstrip()}'")   # Left side only
print(f"After rstrip: '{messy.rstrip()}'")   # Right side only
# Output:
# Before: '   hello world   '
# After strip: 'hello world'
# After lstrip: 'hello world   '
# After rstrip: '   hello world'


# === SPLIT AND JOIN ===
# split() breaks a string into a list of parts
# join() combines a list into a single string
# These two are EXTREMELY useful for text processing in NLP/ML!

# --- split ---
sentence = "Machine Learning is amazing"
words = sentence.split()   # Split on spaces (default)
print(words)
# Output: ['Machine', 'Learning', 'is', 'amazing']

csv_data = "apple,banana,cherry"
items = csv_data.split(",")   # Split on commas
print(items)
# Output: ['apple', 'banana', 'cherry']

# --- join ---
words = ["I", "love", "Python"]
sentence = " ".join(words)    # Join with spaces
print(sentence)
# Output: I love Python

csv_line = ",".join(["name", "age", "city"])
print(csv_line)
# Output: name,age,city


# === STRING FORMATTING ===
# There are several ways to build strings with variables inside them.

name = "Astha"
age = 25
score = 87.6543

# Method 1: f-strings (the best and most modern way!)
print(f"Name: {name}, Age: {age}")
# Output: Name: Astha, Age: 25

# You can put expressions inside the curly braces
print(f"In 5 years, you'll be {age + 5}")
# Output: In 5 years, you'll be 30

# Format numbers with f-strings
print(f"Score: {score:.2f}")    # 2 decimal places
# Output: Score: 87.65

print(f"Score: {score:.0f}")    # No decimal places
# Output: Score: 88

# Method 2: .format() (older but still used)
print("Hello, {}!".format(name))
# Output: Hello, Astha!

# Method 3: % formatting (oldest, you'll see in older code)
print("Hello, %s!" % name)
# Output: Hello, Astha!

# Stick with f-strings -- they're the cleanest!


# === LOOPING THROUGH STRINGS ===

word = "Hello"
for letter in word:
    print(letter, end="-")
print()
# Output: H-e-l-l-o-

# Check if a character is in a string
print("H" in "Hello")    # Output: True
print("x" in "Hello")    # Output: False


# === STRING VS LIST: MUTABILITY COMPARISON ===

print("\n--- Mutability Comparison ---")

# String (immutable) -- cannot change in place
my_string = "hello"
# my_string[0] = "H"   # ERROR! Can't do this.
my_string = "H" + my_string[1:]   # Must create new string
print("New string:", my_string)
# Output: New string: Hello

# List (mutable) -- can change in place
my_list = ["h", "e", "l", "l", "o"]
my_list[0] = "H"   # This works!
print("Changed list:", my_list)
# Output: Changed list: ['H', 'e', 'l', 'l', 'o']

# Convert between them:
letters = list("hello")     # String to list of characters
print(letters)              # Output: ['h', 'e', 'l', 'l', 'o']

word = "".join(letters)     # List back to string
print(word)                 # Output: hello


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test: basic string properties
assert name == "Astha", "name should be 'Astha'"
assert len("Python") == 6, "len of 'Python' should be 6"
assert "Python"[0] == "P", "first letter of 'Python' should be P"
assert "Python"[-1] == "n", "last letter of 'Python' should be n"
assert "Python"[0:3] == "Pyt", "slice [0:3] of 'Python' should be 'Pyt'"
assert "Python"[::-1] == "nohtyP", "reversed 'Python' should be 'nohtyP'"

# Test: string methods (case)
assert "Hello, World!".upper() == "HELLO, WORLD!", "upper() is wrong"
assert "Hello, World!".lower() == "hello, world!", "lower() is wrong"
assert "hello".capitalize() == "Hello", "capitalize() is wrong"

# Test: checking content
assert "hello".isalpha() == True, "isalpha() should be True for 'hello'"
assert "hello123".isalpha() == False, "isalpha() should be False for 'hello123'"
assert "12345".isdigit() == True, "isdigit() should be True for '12345'"
assert "Hello, World!".startswith("Hello") == True, "startswith('Hello') should be True"
assert "Hello, World!".endswith("!") == True, "endswith('!') should be True"

# Test: find, count, replace
assert "I love Python and Python loves me".find("Python") == 7, "find('Python') should be 7"
assert "I love Python and Python loves me".count("Python") == 2, "count('Python') should be 2"
assert "I love Python and Python loves me".replace("Python", "ML") == "I love ML and ML loves me", "replace is wrong"

# Test: strip
assert "   hello world   ".strip() == "hello world", "strip() is wrong"
assert "   hello world   ".lstrip() == "hello world   ", "lstrip() is wrong"
assert "   hello world   ".rstrip() == "   hello world", "rstrip() is wrong"

# Test: split and join
assert "Machine Learning is amazing".split() == ["Machine", "Learning", "is", "amazing"], "split() is wrong"
assert "apple,banana,cherry".split(",") == ["apple", "banana", "cherry"], "split(',') is wrong"
assert " ".join(["I", "love", "Python"]) == "I love Python", "join with space is wrong"
assert ",".join(["name", "age", "city"]) == "name,age,city", "join with comma is wrong"

# Test: immutability workaround
assert "a" + "Astha"[1:] == "astha", "string concatenation for lowercase first char is wrong"

# Test: convert between string and list
assert list("hello") == ["h", "e", "l", "l", "o"], "list('hello') is wrong"
assert "".join(["h", "e", "l", "l", "o"]) == "hello", "join to rebuild string is wrong"

print("\nAll tests passed!")

# === EXERCISES ===

# 1. Given the string below, print it in all uppercase, all lowercase,
#    and title case (first letter of each word capitalized).
#    Expected output:
#    MACHINE LEARNING IS FUN
#    machine learning is fun
#    Machine Learning Is Fun

text = "machine learning is fun"

print(text.upper())
print(text.lower())
print(text.capitalize())


# 2. Given the sentence below, split it into words, then count
#    how many words it has.
#    Expected output: 6
#    Hint: split it, then use len() on the result

sentence = "I want to learn machine learning"

words = sentence.split()
print(len(words))



# 3. Given the list of words below, join them with " -> " between each word.
#    Expected output: start -> learn -> practice -> master
#    Hint: use " -> ".join(...)

steps = ["start", "learn", "practice", "master"]

new_steps = " -> ".join(steps)
print(new_steps)


# 4. Write code that takes the messy string below, strips whitespace,
#    converts to lowercase, and replaces "hate" with "love".
#    Expected output: 'i love python'

messy_input = "   I Hate Python   "

proper_input = messy_input.strip()
lower_input = proper_input.lower()
final = lower_input.replace("hate", "love")
print(final)


# 5. (Challenge) Given a sentence, count how many vowels (a, e, i, o, u)
#    it contains. Ignore uppercase/lowercase.
#    Expected output for "Hello World": 3
#    Hint: convert to lowercase first, then loop through each character

test_sentence = "Hello World"

updated_sentence = test_sentence.lower()
count = 0
for character in updated_sentence:
    if character == "a" or character == "e" or character == "i" or character == "o" or character == "u":
        count += 1
print(count)
