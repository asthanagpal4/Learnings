# HOW TO RUN:
#   uv run python 07_nlp_tokenization/03_word_tokenization.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE: 03_word_tokenization.py
# TOPIC: Tokenization — How do we split text into pieces for ML?
#
# KEY IDEA:
#   Before feeding text to an ML model, we must split it into "tokens".
#   A token is the basic unit the model works with — like a word, a character,
#   or a piece of a word.
#
#   There are three main approaches:
#     1. Word tokenization  — split by spaces ("hello world" -> ["hello", "world"])
#     2. Character tokenization — split by character ("hi" -> ["h", "i"])
#     3. Subword tokenization — split into word-pieces (the modern approach!)
#
#   This file explores options 1 and 2, and explains WHY modern models
#   use option 3 (subword tokenization, specifically BPE).
# =============================================================================

import re
from collections import Counter

# =============================================================================
# OUR TEST CORPUS
# =============================================================================

CORPUS = """
Machine learning is a type of artificial intelligence that allows computers to
learn from data without being explicitly programmed. Deep learning is a subset
of machine learning that uses neural networks with many layers. Natural language
processing helps computers understand human language. Tokenization is the process
of splitting text into smaller units called tokens. The tokenizer is trained on
a large corpus of text to learn which token splits are most useful.
Running, runs, runner, ran — all related to the concept of running.
Unbelievable, unbelievably, believable, believe, believer, believed.
""".strip()

print("=" * 60)
print("TOKENIZATION: Three approaches compared")
print("=" * 60)


# =============================================================================
# APPROACH 1: WORD TOKENIZATION
# =============================================================================

print()
print("=" * 60)
print("APPROACH 1: Word Tokenization")
print("=" * 60)

# The simplest idea: split text into words by splitting on spaces and newlines.

def word_tokenize(text):
    """
    Simple word tokenizer:
      1. Lowercase everything
      2. Remove punctuation
      3. Split by whitespace
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip()   # collapse multiple spaces
    words = text.split()
    return words

word_tokens = word_tokenize(CORPUS)
word_vocab = sorted(set(word_tokens))

print(f"Total tokens (with repeats): {len(word_tokens)}")
print(f"Unique tokens (vocabulary size): {len(word_vocab)}")
print()
print("Sample vocabulary (first 30 words alphabetically):")
print(word_vocab[:30])


# =============================================================================
# FIRST PRINCIPLES: ZIPF'S LAW AND HEAPS' LAW
# =============================================================================
#
# ZIPF'S LAW:
#   The k-th most frequent word in a language has frequency proportional to 1/k.
#   If the most common word appears 1,000,000 times, then:
#     2nd most common: ~500,000 times  (1/2)
#     3rd most common: ~333,333 times  (1/3)
#     10th most common: ~100,000 times (1/10)
#     100th most common: ~10,000 times (1/100)
#
#   This means a TINY number of words account for MOST of the text,
#   while a HUGE number of words appear only once or twice.
#
# HEAPS' LAW (derived from Zipf's law):
#   The vocabulary size V grows SUBLINEARLY with corpus size n:
#     V is proportional to n^beta,  where beta is roughly 0.5-0.7
#
#   DERIVATION of the implication:
#     If V = k * n^beta (where k is a constant, beta ~ 0.5):
#       Doubling corpus: V_new = k * (2n)^0.5 = k * sqrt(2) * n^0.5
#                                              = sqrt(2) * V_old
#                                              = 1.41 * V_old
#       So doubling the corpus increases vocabulary by only 41%!
#
#       Quadrupling corpus: V_new = k * (4n)^0.5 = 2 * V_old
#       So 4x corpus -> only 2x vocabulary.
#
#   This is why vocabulary grows very slowly compared to corpus size.
#
# OOV (OUT-OF-VOCABULARY) RATE ANALYSIS:
#   With vocabulary of size V built from corpus of size n:
#   - OOV rate on new text decreases as V increases
#   - But it NEVER reaches 0 for word-level tokenization
#   - Why? Because new words are always being created (proper nouns,
#     neologisms, typos, code, URLs, foreign words...)
#   - This is a fundamental limitation of word-level tokenization
#   - Subword tokenization (BPE) solves this by falling back to characters
#
# =============================================================================

# --- Problem 1: HUGE VOCABULARY ---

print()
print("--- Problem 1: Vocabulary grows with the language ---")
print("""
For English Wikipedia alone:
  - Unique words: ~3 million
  - With proper nouns, abbreviations, code snippets: millions more

Each word needs its own row in the model's embedding table.
A vocabulary of 1 million words = 1 million rows to store and train.
This is too expensive!

Also, rare words appear so few times that the model can't learn
anything useful about them.
""")

# Show word frequency distribution
word_counts = Counter(word_tokens)
# Count how many words appear only once
once = sum(1 for count in word_counts.values() if count == 1)
twice = sum(1 for count in word_counts.values() if count == 2)
often = sum(1 for count in word_counts.values() if count >= 5)

print(f"Words appearing only once:   {once}  ({100*once/len(word_vocab):.0f}% of vocab)")
print(f"Words appearing twice:       {twice}")
print(f"Words appearing 5+ times:    {often}")
print()
print("In a large corpus, ~50% of unique words appear only once.")
print("The model sees these words too rarely to learn anything about them.")


# --- Problem 2: OUT-OF-VOCABULARY (OOV) PROBLEM ---

print()
print("--- Problem 2: Out-of-Vocabulary (OOV) problem ---")

# Once we build our vocabulary from training data, any new word that
# appears at test time is "out of vocabulary" (OOV).
# We have no token ID for it — the model can't process it at all!

test_sentences = [
    "tokenization is tokenizing tokens",  # "tokenizing" might be OOV
    "the model preprocessed the dataset",  # "preprocessed" might be OOV
    "ChatGPT is a chatbot",               # "ChatGPT" is definitely OOV
    "the WiFi is broken",                 # "WiFi" might be OOV
]

print("Testing for OOV words (words not in training vocabulary):")
for sentence in test_sentences:
    tokens = word_tokenize(sentence)
    oov = [t for t in tokens if t not in set(word_vocab)]
    known = [t for t in tokens if t in set(word_vocab)]
    print(f"\n  Sentence: '{sentence}'")
    print(f"  Tokens:   {tokens}")
    if oov:
        print(f"  OOV words (can't process!): {oov}")
    else:
        print(f"  All words known!")


# --- Problem 3: WORD FORMS PROBLEM ---

print()
print("--- Problem 3: Related words are treated as completely different ---")

related_groups = [
    ["run", "running", "runs", "ran", "runner"],
    ["believe", "believed", "believer", "believable", "unbelievable", "unbelievably"],
    ["token", "tokens", "tokenize", "tokenization", "tokenizing", "tokenizer"],
    ["learn", "learning", "learned", "learner", "unlearned"],
]

print("\nWord tokenization treats each of these as a SEPARATE, UNRELATED token:")
for group in related_groups:
    present = [w for w in group if w in set(word_tokens)]
    absent  = [w for w in group if w not in set(word_tokens)]
    print(f"\n  Group: {group}")
    print(f"    In our vocab: {present}")
    print(f"    NOT in vocab: {absent}  <- model can't handle these!")

print("""
The model has to learn separately that "run" and "running" are related.
A smarter tokenization would keep "run" as a shared piece.
""")


# =============================================================================
# APPROACH 2: CHARACTER TOKENIZATION
# =============================================================================

print()
print("=" * 60)
print("APPROACH 2: Character Tokenization")
print("=" * 60)

# Instead of splitting by word, split by individual character.
# Every word is broken into its letters.

def char_tokenize(text):
    """
    Character tokenizer: every character becomes a token.
    We keep spaces as a special token too.
    """
    text = text.lower()
    # Keep only letters, digits, and spaces
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return list(text)

char_tokens = char_tokenize(CORPUS)
char_vocab = sorted(set(char_tokens))

print(f"Total tokens (with repeats): {len(char_tokens)}")
print(f"Unique tokens (vocabulary size): {len(char_vocab)}")
print()
print("The entire character vocabulary:")
print(char_vocab)

print()
print("Encoding the word 'tokenization' character by character:")
example_word = "tokenization"
char_encoding = list(example_word)
print(f"  '{example_word}' -> {char_encoding}")
print(f"  As IDs: {[char_vocab.index(c) for c in char_encoding if c in char_vocab]}")


# --- Advantage: No OOV problem! ---

print()
print("--- Advantage: No OOV problem ---")
test_word = "supercalifragilistic"
chars = [c for c in test_word if c in set(char_vocab)]
print(f"Even a made-up word like '{test_word}'")
print(f"can be encoded because we know all its characters!")
print(f"Characters: {chars}")


# --- Problem: Sequences become VERY long ---

print()
print("--- Problem: Sequences become too long ---")

word = "unbelievably"
word_as_words = [word]         # 1 token
word_as_chars = list(word)     # 12 tokens

print(f"The word '{word}':")
print(f"  As word token:  {word_as_words}  -> 1 token")
print(f"  As char tokens: {word_as_chars}  -> {len(word_as_chars)} tokens")

sentence = "Machine learning is fascinating"
as_words = word_tokenize(sentence)
as_chars = [c for c in sentence.lower() if c in set(char_vocab)]

print()
print(f"'{sentence}':")
print(f"  Word tokens: {len(as_words)} tokens")
print(f"  Char tokens: {len(as_chars)} tokens")
print()
print("Character tokenization makes sequences ~5x longer than word tokenization.")
print("Longer sequences = more memory + more computation for the model.")
print("The model also has to learn that 'c','a','t' together mean something.")


# --- Problem: Loses natural word boundaries ---

print()
print("--- Problem: Characters alone have little meaning ---")
print("""
  The letter 'a' could be part of: cat, hat, mat, bat, man, ran ...
  Without seeing the whole word, a single character tells us very little.
  The model must learn to combine characters into meaningful units anyway.
  It's doing extra work that word tokenization gives for free.
""")


# =============================================================================
# COMPARISON: Word vs Character tokenization
# =============================================================================

print()
print("=" * 60)
print("COMPARISON: Word vs Character Tokenization")
print("=" * 60)

# Show vocabulary sizes for different corpus sizes
print("\nFor our small corpus:")
print(f"  Word vocabulary size:      {len(word_vocab):5d} tokens")
print(f"  Character vocabulary size: {len(char_vocab):5d} tokens")

# Simulate what happens with larger text
small_text = "the cat sat on the mat"
medium_text = " ".join([small_text] * 50 + [
    "the dog ran fast", "a big red ball", "one two three four five"
] * 20)
large_simulation = medium_text + " " + " ".join([
    f"word{i}" for i in range(500)  # simulate many unique words
])

small_wv = len(set(word_tokenize(small_text)))
medium_wv = len(set(word_tokenize(medium_text)))
large_wv  = len(set(word_tokenize(large_simulation)))

small_cv = len(set(char_tokenize(small_text)))
medium_cv = len(set(char_tokenize(medium_text)))
large_cv  = len(set(char_tokenize(large_simulation)))

print()
print("How vocabulary size grows with more text:")
print(f"{'Text size':20s} {'Word vocab':15s} {'Char vocab':15s}")
print("-" * 50)
print(f"{'Small (22 chars)':20s} {small_wv:15d} {small_cv:15d}")
print(f"{'Medium':20s} {medium_wv:15d} {medium_cv:15d}")
print(f"{'Large (simulated)':20s} {large_wv:15d} {large_cv:15d}")

print("""
Key observation:
  - Word vocabulary keeps GROWING as you add more text (unbounded!)
  - Character vocabulary stays STABLE (English only has ~30 chars)

This is a fundamental property: more text = more unique words,
but the alphabet doesn't change!
""")


# =============================================================================
# THE SOLUTION: SUBWORD TOKENIZATION
# =============================================================================

print("=" * 60)
print("THE SOLUTION: Subword Tokenization (preview)")
print("=" * 60)

print("""
The sweet spot is between words and characters: SUBWORD tokens.

Instead of treating "unbelievable" as one token (word-level)
or 12 separate characters (char-level), split it into:
  un + believ + able

Now:
  - "believable" -> believ + able
  - "unbelievably" -> un + believ + ably
  - "believer" -> believ + er

The piece "believ" is SHARED across all these words!
The model only needs to learn "believ" once.

Subword tokenization gives us:
  - Small vocabulary (like characters): ~30,000-50,000 tokens for LLMs
  - Handles new words (like characters): "ChatGPT" -> "Chat" + "G" + "PT"
  - Keeps common words whole (like words): "the", "is", "a" stay as one token
  - Shared prefixes/suffixes: "un-", "-ing", "-ed", "-tion" are reusable

The most popular subword algorithm is BPE (Byte Pair Encoding).
This is what GPT, LLaMA, Mistral, and most modern LLMs use.
We'll implement it from scratch in the next files!
""")

# Quick demo of what subword tokenization WOULD look like
# (we'll actually implement it in 04_bpe_algorithm.py)
print("Preview of BPE subword tokenization:")
examples = [
    ("unbelievable",   ["un", "believ", "able"]),
    ("tokenization",   ["token", "ization"]),
    ("running",        ["run", "ning"]),
    ("machine",        ["machine"]),          # common word stays whole
    ("ChatGPT",        ["Chat", "G", "PT"]),  # unknown word split into pieces
    ("preprocessing",  ["pre", "process", "ing"]),
]

for word, pieces in examples:
    pieces_str = " + ".join(f'"{p}"' for p in pieces)
    print(f"  '{word:20s}' -> {pieces_str}")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# word_tokenize: lowercases, removes punctuation, splits by whitespace
_wt = word_tokenize("Hello, World! Python is GREAT.")
assert _wt == ['hello', 'world', 'python', 'is', 'great']

# word_tokenize handles extra spaces
_wt2 = word_tokenize("  the   cat  sat  ")
assert _wt2 == ['the', 'cat', 'sat']

# word_tokenize: only letters and digits survive (no punctuation)
_wt3 = word_tokenize("one, two. three!")
assert 'one' in _wt3 and 'two' in _wt3 and 'three' in _wt3
assert ',' not in ' '.join(_wt3)

# char_tokenize: every character becomes a token, spaces become spaces
_ct = char_tokenize("hi")
assert 'h' in _ct and 'i' in _ct

# char_tokenize: vocabulary is small (just letters + space)
assert len(char_vocab) <= 28    # 26 letters + digits + space at most
assert len(char_vocab) >= 20    # most letters present in any English text

# word_vocab is always larger than char_vocab for real text
assert len(word_vocab) > len(char_vocab)

# Character tokenization produces more tokens than word tokenization
_sentence = "machine learning is fascinating"
_as_words = word_tokenize(_sentence)
_as_chars = char_tokenize(_sentence)
assert len(_as_chars) > len(_as_words)

# A word not in training corpus is OOV for word tokenization
_word_set = set(word_vocab)
assert 'chatgpt' not in _word_set   # this made-up proper noun was never seen

# Character tokenization handles any word built from known letters
_test_word = "supercalifragilistic"
_chars_seen = [c for c in _test_word if c in set(char_vocab)]
assert len(_chars_seen) == len(_test_word)   # all letters are known chars

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
  Take this sentence and tokenize it both ways (word and character):
    "Tokenization splits text into meaningful pieces"
  Count how many tokens each approach produces.
  Hint: use the word_tokenize() and char_tokenize() functions above.

Exercise 2:
  Build a word vocabulary from this paragraph:
    "The cat sat on the mat. The cat was fat. The fat cat sat."
  Then try to encode this new sentence using that vocabulary:
    "The thin cat ran on the wet mat."
  Which words are OOV (out of vocabulary)?
  Hint: check each token against the set of known words.

Exercise 3:
  Calculate the "compression ratio" for word tokenization:
    compression_ratio = len(original_text) / len(word_tokens)
  Do the same for character tokenization.
  Which has a higher compression ratio? Why?
  Hint: compression_ratio = characters / tokens

Exercise 4 (challenge):
  Write a function that checks if a given word can be built entirely
  from subpieces of known vocabulary words. For example:
    "running" can be built from "run" + "ning" if both are in vocab
  This is a simplified version of what subword tokenizers do!
  Hint: try splitting the word at every position and check both halves.

Exercise 5 (Heaps' law estimation):
  If 10,000 documents give 50,000 unique words, estimate how many
  unique words 40,000 documents would give (using Heaps' law with beta=0.5).

  Work through it:
    V = k * n^beta
    50,000 = k * 10,000^0.5 = k * 100
    k = 50,000 / 100 = 500

    For n = 40,000:
    V = 500 * 40,000^0.5 = 500 * 200 = 100,000

    So quadrupling the documents (10,000 -> 40,000) only doubles
    the vocabulary (50,000 -> 100,000). This is the sublinear growth
    predicted by Heaps' law!

    If beta were 0.7 (more typical for diverse corpora):
    50,000 = k * 10,000^0.7 = k * 501.2
    k = 99.8
    V = 99.8 * 40,000^0.7 = 99.8 * 1,445.4 = ~144,200

    Higher beta means vocabulary grows faster but still sublinearly.
""")
