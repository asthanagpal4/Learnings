# HOW TO RUN:
#   uv run python 03_data_structures/06_hash_tables.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- HASH TABLES ---
# A hash table is the engine behind Python's dict. When you write
# my_dict["name"] = "Astha", Python doesn't search through every
# item to find "name" -- it uses a clever trick to jump DIRECTLY
# to the right spot. This makes lookups super fast.
#
# How it works (simplified):
# 1. Take the key (e.g., "name")
# 2. Run it through a "hash function" that converts it to a number
# 3. Use that number to pick a slot in an array
# 4. Store the value in that slot
#
# Why does this matter?
# - Python dicts are hash tables -- you use them every day!
# - Understanding this helps you write faster code
# - In ML, feature hashing is used to handle large vocabularies
# - Embedding tables in neural networks work on similar principles


# === CONCEPT 1: What is a hash function? ===
# A hash function takes any input and converts it into a fixed-size
# number. Python has a built-in hash() function.

print("=" * 50)
print("CONCEPT 1: Hash functions")
print("=" * 50)

# Python's built-in hash function
print("hash('Astha')  =", hash("Astha"))
print("hash('Raj')    =", hash("Raj"))
print("hash(42)       =", hash(42))
print("hash(3.14)     =", hash(3.14))

# The same input ALWAYS gives the same hash (within one program run)
print("\nhash('hello') == hash('hello'):", hash("hello") == hash("hello"))
# Output: True

# Different inputs (usually) give different hashes
print("hash('hello') == hash('world'):", hash("hello") == hash("world"))
# Output: False

# How we use the hash to pick a slot:
# If we have 10 slots, we use: hash(key) % 10
table_size = 10
print(f"\nSlot for 'Astha': {hash('Astha') % table_size}")
print(f"Slot for 'Raj':   {hash('Raj') % table_size}")
print(f"Slot for 'Priya': {hash('Priya') % table_size}")

print()


# === CONCEPT 2: Building a simple hash table from scratch ===
# Let's build our own mini version of Python's dict!

print("=" * 50)
print("CONCEPT 2: Building a hash table from scratch")
print("=" * 50)

class HashTable:
    def __init__(self, size=10):
        self.size = size
        # Create a list of empty "buckets"
        # Each bucket will hold a list of (key, value) pairs
        self.buckets = [[] for _ in range(size)]

    def _hash(self, key):
        """Convert a key into a bucket index."""
        return hash(key) % self.size

    def set(self, key, value):
        """Store a key-value pair."""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]

        # Check if key already exists in this bucket
        for i, (existing_key, existing_value) in enumerate(bucket):
            if existing_key == key:
                # Update existing key
                bucket[i] = (key, value)
                return

        # Key doesn't exist yet, add it
        bucket.append((key, value))

    def get(self, key):
        """Retrieve a value by its key."""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]

        for existing_key, value in bucket:
            if existing_key == key:
                return value

        raise KeyError(f"Key '{key}' not found")

    def delete(self, key):
        """Remove a key-value pair."""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]

        for i, (existing_key, value) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                return

        raise KeyError(f"Key '{key}' not found")

    def contains(self, key):
        """Check if a key exists."""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]

        for existing_key, value in bucket:
            if existing_key == key:
                return True
        return False

    def display(self):
        """Show the internal structure."""
        print("  Hash Table internals:")
        for i, bucket in enumerate(self.buckets):
            if bucket:
                print(f"    Bucket {i}: {bucket}")
            else:
                print(f"    Bucket {i}: (empty)")


# Let's use our hash table
ht = HashTable(size=5)
ht.set("name", "Astha")
ht.set("age", 25)
ht.set("city", "Delhi")
ht.set("hobby", "coding")

print("Our hash table:")
ht.display()

print(f"\nGet 'name': {ht.get('name')}")
# Output: Get 'name': Astha

print(f"Contains 'age': {ht.contains('age')}")
# Output: Contains 'age': True

print(f"Contains 'email': {ht.contains('email')}")
# Output: Contains 'email': False

# Update a value
ht.set("age", 26)
print(f"Updated 'age': {ht.get('age')}")
# Output: Updated 'age': 26

print()


# === CONCEPT 3: Collisions ===
# A "collision" happens when two different keys hash to the same
# bucket. For example, hash("Astha") % 5 and hash("Raj") % 5
# might both give bucket 3.
#
# Our hash table handles this using "chaining" -- each bucket
# holds a LIST of (key, value) pairs. So multiple items can
# share the same bucket.

print("=" * 50)
print("CONCEPT 3: Collisions and how to handle them")
print("=" * 50)

# Let's force collisions by using a tiny hash table
tiny = HashTable(size=3)  # only 3 buckets for many items!

items = [
    ("apple", 1), ("banana", 2), ("cherry", 3),
    ("date", 4), ("elderberry", 5), ("fig", 6)
]

for key, value in items:
    tiny.set(key, value)
    bucket_index = tiny._hash(key)
    print(f"  '{key}' -> bucket {bucket_index}")

print("\nWith only 3 buckets, collisions are guaranteed:")
tiny.display()

# But lookups still work correctly!
print(f"\nGet 'cherry': {tiny.get('cherry')}")
print(f"Get 'fig': {tiny.get('fig')}")
# Each bucket is a list, so Python searches within the bucket

print()

# The two main strategies for handling collisions:
print("Two strategies for handling collisions:")
print("""
  1. Chaining (what we built above)
     - Each bucket holds a list of items
     - If two keys land in the same bucket, both go in the list
     - Simple but uses extra memory

  2. Open Addressing (what Python actually uses for dicts)
     - If a bucket is taken, look at the next bucket
     - Keep looking until you find an empty one
     - More complex but faster in practice
""")


# === CONCEPT 4: Why hash tables are fast ===

print("=" * 50)
print("CONCEPT 4: Speed comparison")
print("=" * 50)

import time

# Let's compare: searching in a list vs searching in a dict
big_list = list(range(1_000_000))
big_dict = {i: True for i in range(1_000_000)}

# Search for something near the end
target = 999_999

# List search (slow -- has to check every element)
start = time.time()
for _ in range(100):
    _ = target in big_list
list_time = time.time() - start

# Dict search (fast -- jumps directly to the right spot)
start = time.time()
for _ in range(100):
    _ = target in big_dict
dict_time = time.time() - start

print(f"Searching for {target} (100 times):")
print(f"  List: {list_time:.4f} seconds")
print(f"  Dict: {dict_time:.6f} seconds")
print(f"  Dict is roughly {list_time / max(dict_time, 0.000001):.0f}x faster!")

print()

print("Speed of operations (Big-O notation):")
print("""
  OPERATION     | List        | Hash Table (dict)
  --------------|-------------|------------------
  Search        | O(n) slow   | O(1) instant*
  Insert        | O(1) fast   | O(1) instant*
  Delete        | O(n) slow   | O(1) instant*
  Access by key | N/A         | O(1) instant*

  * "instant" means it takes the same time whether you have
    10 items or 10 million items. Amazing, right?

  The catch: hash tables use more memory than lists.
  This is a classic tradeoff: speed vs memory.
""")


# === CONCEPT 5: What can and can't be a dict key? ===

print("=" * 50)
print("CONCEPT 5: What can be a dict key?")
print("=" * 50)

# Only "hashable" things can be dict keys
# Hashable = immutable (can't be changed)

# These work as keys:
valid_keys = {
    "string_key": 1,       # strings are hashable
    42: 2,                  # numbers are hashable
    (1, 2, 3): 3,           # tuples are hashable
    True: 4,                # booleans are hashable
    3.14: 5,                # floats are hashable
}
print("Valid dict keys:", list(valid_keys.keys()))

# These do NOT work as keys:
print("\nThings that CANNOT be dict keys:")
print("  - Lists (because you can change them)")
print("  - Dicts (because you can change them)")
print("  - Sets  (because you can change them)")
print()

# Why? If a key could change after being stored, Python wouldn't
# be able to find it anymore -- the hash would be different!
print("Why? Because if a key changes, its hash changes,")
print("and Python can't find it in the table anymore.")

print()


# === CONCEPT 6: Hash tables in ML ===

print("=" * 50)
print("CONCEPT 6: Hash tables in Machine Learning")
print("=" * 50)

print("""
  Hash tables show up in ML more than you might think:

  1. Feature Hashing (Hashing Trick)
     - Instead of creating a huge table of all possible words,
       hash each word to a fixed-size array
     - Used in spam filters and text classification

  2. Embedding Tables
     - Neural networks map words/items to vectors using lookups
     - This is basically a hash table: word -> vector of numbers

  3. Caching / Memoization
     - Store results of expensive computations
     - If the same input comes again, return cached result
     - Used to speed up model training and inference

  4. Counting and Frequency
     - Count how many times each word appears (bag of words)
     - Count label frequencies for class balancing
""")

# Quick example: word frequency counter
text = "the cat sat on the mat the cat"
word_counts = {}
for word in text.split():
    word_counts[word] = word_counts.get(word, 0) + 1

print("Word frequencies:")
for word, count in word_counts.items():
    print(f"  '{word}': {count}")
# Output:
#   'the': 3
#   'cat': 2
#   'sat': 1
#   'on': 1
#   'mat': 1

print()


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test HashTable: set / get / contains / delete
_ht = HashTable(size=7)
assert _ht.contains("x") == False

_ht.set("name", "Astha")
_ht.set("age", 25)
_ht.set("city", "Delhi")
assert _ht.get("name") == "Astha"
assert _ht.get("age") == 25
assert _ht.get("city") == "Delhi"
assert _ht.contains("name") == True
assert _ht.contains("email") == False

# Test update (set same key again)
_ht.set("age", 26)
assert _ht.get("age") == 26

# Test delete
_ht.delete("city")
assert _ht.contains("city") == False

# Test KeyError on missing key
_key_error_raised = False
try:
    _ht.get("city")
except KeyError:
    _key_error_raised = True
assert _key_error_raised == True

# Test with collisions (tiny table)
_tiny = HashTable(size=2)
_tiny.set("a", 1)
_tiny.set("b", 2)
_tiny.set("c", 3)
assert _tiny.get("a") == 1
assert _tiny.get("b") == 2
assert _tiny.get("c") == 3

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Two Sum
#    Given a list of numbers and a target sum, find two numbers
#    that add up to the target. Return their indices.
#    Example: numbers = [2, 7, 11, 15], target = 9 -> (0, 1)
#    Because numbers[0] + numbers[1] = 2 + 7 = 9
#    Hint: for each number, check if (target - number) is in a dict.

# Your code here:


# 2. First unique character
#    Given a string, find the first character that appears only once.
#    Example: "aabbcdef" -> 'c'
#    Hint: count all characters first (using a dict), then find
#    the first one with count 1.

# Your code here:


# 3. Group anagrams
#    Given a list of words, group together words that are anagrams
#    of each other (same letters, different order).
#    Example: ["eat", "tea", "tan", "ate", "nat", "bat"]
#    -> [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
#    Hint: sort the letters of each word to create a key.
#    "eat" sorted = "aet", "tea" sorted = "aet" -- same key!

# Your code here:
