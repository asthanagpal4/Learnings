# HOW TO RUN:
#   uv run python 07_nlp_tokenization/01_text_processing.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE: 01_text_processing.py
# TOPIC: Text Processing — How computers turn text into numbers
#
# KEY IDEA:
#   Computers don't understand letters or words. They only understand numbers.
#   Before any ML model can read a sentence, that sentence must become a list
#   of numbers. This file shows the first steps of that journey.
# =============================================================================

# --- PART 1: COMPUTERS SEE NUMBERS, NOT LETTERS ---

# Every character on your keyboard has a number assigned to it.
# This numbering system is called ASCII (American Standard Code for
# Information Interchange). It was invented in the 1960s.
#
# For example:
#   'A' = 65
#   'a' = 97
#   '0' = 48
#   ' ' (space) = 32

print("=" * 60)
print("PART 1: Characters are secretly numbers!")
print("=" * 60)

# ord() converts a character to its number
print("ord('A') =", ord('A'))
print("ord('a') =", ord('a'))
print("ord('z') =", ord('z'))
print("ord('0') =", ord('0'))
print("ord(' ') =", ord(' '))

# chr() does the opposite: number -> character
print()
print("chr(65)  =", chr(65))
print("chr(97)  =", chr(97))
print("chr(72)  =", chr(72))
print("chr(101) =", chr(101))

# You can spell out a word using ord():
word = "Hello"
print()
print(f"The word '{word}' as numbers:")
numbers = [ord(c) for c in word]
print(numbers)

# And convert back:
recovered = "".join([chr(n) for n in numbers])
print(f"Numbers back to text: '{recovered}'")


# --- PART 2: UTF-8 ENCODING ---

print()
print("=" * 60)
print("PART 2: UTF-8 — handling ALL languages")
print("=" * 60)

# ASCII only covers English letters and basic symbols (128 characters).
# UTF-8 is an extension that can represent every character in every
# language — over 140,000 characters!
#
# In Python, strings are already UTF-8 internally.
# The encode() method converts a string to raw bytes (numbers 0-255).

text = "Hello"
encoded = text.encode('utf-8')
print(f"'{text}' encoded to bytes: {list(encoded)}")

# Non-English characters use more than 1 byte:
text2 = "cafe"
encoded2 = text2.encode('utf-8')
print(f"'{text2}' encoded: {list(encoded2)}")

# The key insight: every piece of text can become a list of numbers.
# That list of numbers is what gets fed into an ML model.


# --- PART 3: BASIC TEXT CLEANING ---

print()
print("=" * 60)
print("PART 3: Cleaning text before processing")
print("=" * 60)

# Real-world text is messy. Before processing, we usually clean it:
#   1. Lowercase everything (so "The" and "the" are the same word)
#   2. Remove punctuation (commas, periods, etc. are usually noise)
#   3. Split into words (tokenize)

import re  # 're' is the Regular Expressions module, built into Python

raw_text = "Hello, World! This is a Test. Python is GREAT for ML."

print("Original text:")
print(raw_text)

# Step 1: Lowercase
lower_text = raw_text.lower()
print()
print("After lowercasing:")
print(lower_text)

# Step 2: Remove punctuation
# re.sub(pattern, replacement, text) replaces all matches of 'pattern' with 'replacement'
# '[^a-z0-9\s]' means "anything that is NOT a-z, 0-9, or whitespace"
clean_text = re.sub(r'[^a-z0-9\s]', '', lower_text)
print()
print("After removing punctuation:")
print(clean_text)

# Step 3: Split into words
words = clean_text.split()
print()
print("After splitting into words:")
print(words)
print(f"Number of words: {len(words)}")


# =============================================================================
# FIRST PRINCIPLES: COMPLEXITY ANALYSIS OF VOCABULARY BUILDING
# =============================================================================
#
# What is the computational cost of building a vocabulary from text?
#
# STEP 1: Scanning the corpus to find all words
#   - We read every character/word in the text exactly once
#   - If n = total number of words (or characters) in the corpus:
#     TIME: O(n) — linear scan
#
# STEP 2: Finding unique words (using a set)
#   - Inserting each word into a set: O(1) average per word, O(n) total
#   - After scanning, we have v unique words (the vocabulary)
#
# STEP 3: Sorting the vocabulary
#   - Sorting v unique words: O(v log v)
#
# TOTAL TIME COMPLEXITY: O(n + v log v)
#   - For large corpora, n >> v log v, so it is dominated by O(n)
#   - Example: Wikipedia has ~3 billion words, ~3 million unique words
#     n = 3,000,000,000    v = 3,000,000
#     n >> v log v = 3,000,000 * ~22 = 66,000,000
#     So the corpus scan dominates!
#
# SPACE COMPLEXITY:
#   - O(n) for storing the corpus in memory
#   - O(v) for storing the vocabulary (the set of unique words)
#   - Total: O(n + v), dominated by O(n) since v <= n
#
# =============================================================================

# --- PART 4: BUILDING A VOCABULARY ---

print()
print("=" * 60)
print("PART 4: Building a vocabulary (word -> number mapping)")
print("=" * 60)

# A "vocabulary" is a dictionary that maps each unique word to a unique number.
# This is how ML models represent words.
#
# Example:
#   "hello" -> 0
#   "world" -> 1
#   "python" -> 2
#   ...

sample_text = """
the cat sat on the mat the cat ate the rat
the dog ran on the mat the dog saw the cat
"""

# Clean the text
words_sample = sample_text.lower().split()
print("Words in our sample text:")
print(words_sample)
print(f"Total words (with repeats): {len(words_sample)}")

# Get unique words
unique_words = sorted(set(words_sample))  # sorted() keeps order consistent
print()
print("Unique words (vocabulary):")
print(unique_words)
print(f"Vocabulary size: {len(unique_words)}")

# Build word-to-integer mapping (word2id)
word2id = {word: idx for idx, word in enumerate(unique_words)}
print()
print("Word to ID mapping:")
for word, idx in word2id.items():
    print(f"  '{word}' -> {idx}")

# Build integer-to-word mapping (id2word) — needed for decoding
id2word = {idx: word for word, idx in word2id.items()}

# Encode a sentence as a list of integers
sentence = "the cat sat on the mat"
encoded_sentence = [word2id[w] for w in sentence.split()]
print()
print(f"Encoding '{sentence}':")
print(f"  -> {encoded_sentence}")

# Decode back to words
decoded_sentence = " ".join([id2word[i] for i in encoded_sentence])
print(f"Decoding back: '{decoded_sentence}'")


# --- PART 5: WORD FREQUENCY COUNTING ---

print()
print("=" * 60)
print("PART 5: Counting word frequencies")
print("=" * 60)

from collections import Counter  # Counter makes counting easy!

# A Counter is a special dictionary that automatically counts things.
word_counts = Counter(words_sample)
print("Word counts in our sample text:")
for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
    bar = "#" * count
    print(f"  '{word:8s}': {count}  {bar}")

# Most common words:
print()
print("5 most common words:")
for word, count in word_counts.most_common(5):
    print(f"  '{word}' appears {count} times")

# This information is useful! In ML:
#   - Very common words ("the", "a", "is") are often less informative
#   - Very rare words might be noise or typos


# --- PART 6: CONNECTION TO ML ---

print()
print("=" * 60)
print("PART 6: Why this matters for ML")
print("=" * 60)

print("""
Here is the pipeline from raw text to ML-ready input:

  Raw Text  ->  Clean  ->  Tokenize  ->  Vocabulary  ->  Numbers

  "Hello!"  ->  "hello" ->  ["hello"]  ->  {"hello":0}  ->  [0]

The numbers are what the model actually sees. Every word (or character,
or subword piece) gets a unique integer ID.

Key facts:
  - GPT-2 uses a vocabulary of 50,257 tokens
  - LLaMA uses a vocabulary of 32,000 tokens
  - Each input to these models is a list of integers
  - The model learns which integers tend to follow other integers

This is why "predicting the next token" is the core task of language models.
The model sees a list of integers and tries to guess the next integer.
""")

# Demonstrate the full pipeline on one sentence:
test_sentence = "the dog saw the cat"
print(f"Full pipeline demo with: '{test_sentence}'")

cleaned = test_sentence.lower()
tokens = cleaned.split()
print(f"  After cleaning & splitting: {tokens}")

# Handle unknown words gracefully
ids = []
for token in tokens:
    if token in word2id:
        ids.append(word2id[token])
    else:
        ids.append(-1)  # -1 means "unknown"
print(f"  As integer IDs:             {ids}")
print(f"  (using vocab built earlier)")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# ord() and chr() are inverses of each other
assert ord('A') == 65
assert ord('a') == 97
assert chr(65) == 'A'
assert chr(97) == 'a'

# Encoding a word to numbers and back recovers the original word
word_test = "Hello"
nums = [ord(c) for c in word_test]
assert nums == [72, 101, 108, 108, 111]
assert "".join([chr(n) for n in nums]) == word_test

# UTF-8 encoding: "Hello" encodes to the same bytes as its ASCII codes
assert list("Hello".encode('utf-8')) == [72, 101, 108, 108, 111]

# Text cleaning: lowercase + remove punctuation
import re as _re
_raw = "Hello, World! This is a Test."
_lower = _raw.lower()
assert _lower == "hello, world! this is a test."
_clean = _re.sub(r'[^a-z0-9\s]', '', _lower)
assert ',' not in _clean
assert '!' not in _clean
assert 'hello' in _clean.split()

# Vocabulary building from sample_text
assert 'the' in word2id
assert 'cat' in word2id
assert 'dog' in word2id
# Vocabulary is the set of unique words — no duplicates
assert len(word2id) == len(unique_words)
# id2word is the reverse mapping
for _w, _i in word2id.items():
    assert id2word[_i] == _w

# Encoding and decoding a sentence is a roundtrip
_sent = "the cat sat on the mat"
_enc = [word2id[w] for w in _sent.split()]
_dec = " ".join([id2word[i] for i in _enc])
assert _dec == _sent

# Word frequency counting: 'the' is the most common word in sample_text
assert word_counts.most_common(1)[0][0] == 'the'
assert word_counts['the'] >= 4   # appears many times in sample_text

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print()
print("=" * 60)
print("EXERCISES — try these yourself!")
print("=" * 60)
print("""
Exercise 1:
  Use ord() and chr() to encode and decode your own name.
  - Convert each letter of your name to its ASCII number
  - Then convert those numbers back to letters
  Hint: [ord(c) for c in "YourName"]

Exercise 2:
  Take this sentence and build a vocabulary from it:
    "I love Python and Python loves me"
  - How many unique words are there?
  - What is the word2id mapping?
  - Encode the sentence as integers
  Hint: use set() to find unique words, then enumerate()

Exercise 3:
  Use collections.Counter to count letter frequencies (not word
  frequencies) in a sentence of your choice.
  - Which letter appears most often?
  - Try it on a long paragraph
  Hint: Counter("your sentence here") counts characters directly

Exercise 4 (challenge):
  Modify the text cleaning function to also:
  - Remove numbers (digits)
  - Remove extra spaces (more than one space in a row)
  Hint: re.sub(r'\\d', '', text) removes digits
        re.sub(r'\\s+', ' ', text) collapses multiple spaces

Exercise 5 (first-principles complexity analysis):
  If a corpus has 1 million words and 50,000 unique words,
  what is the complexity of building the vocabulary?

  Work through it:
    n = 1,000,000 (total words)
    v = 50,000    (unique words)

    Step 1: Scan corpus -> O(n) = O(1,000,000)
    Step 2: Build set   -> O(n) = O(1,000,000)
    Step 3: Sort vocab  -> O(v log v) = O(50,000 * log2(50,000))
                         = O(50,000 * ~15.6) = O(~780,000)
    Total: O(n + v log v) = O(1,000,000 + 780,000) = O(1,780,000)

    The corpus scan dominates. As the corpus grows, sorting the
    vocabulary becomes a smaller and smaller fraction of the work.
""")
