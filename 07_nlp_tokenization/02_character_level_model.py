# HOW TO RUN:
#   uv run python 07_nlp_tokenization/02_character_level_model.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE: 02_character_level_model.py
# TOPIC: Character-Level Language Model — the simplest possible language model
#
# KEY IDEA:
#   The simplest language model asks: "Given the current character, what
#   character is most likely to come next?"
#
#   For example, in English:
#     After 'q', the next character is almost always 'u'
#     After 't', common next characters are 'h', 'o', 'e', 'r', 'i'
#     After a space, many letters are possible
#
#   This file builds a "bigram model" — a model that looks at ONE character
#   and predicts the NEXT character. It's called "bigram" because we look at
#   pairs (bi = two) of characters.
#
#   This is literally how early language models worked! Modern LLMs use the
#   same idea but with subword tokens and neural networks instead of counts.
# =============================================================================

import re
from collections import defaultdict, Counter
import random

# We will use a fixed random seed so your output matches the expected output.
# A "seed" makes random choices reproducible — same seed = same "random" result.
random.seed(42)

# =============================================================================
# OUR TEXT CORPUS
# =============================================================================
# A "corpus" is just a collection of text used for training.
# We use a small hardcoded text here. In real models, this would be
# billions of words scraped from the internet.

CORPUS = """
the cat sat on the mat the cat ate a rat
the dog ran on the mat the dog saw the cat
a big dog sat by the log the log was near the bog
the frog sat on a log near the bog in the fog
cats and dogs and frogs and logs
the quick brown fox jumps over the lazy dog
she sells seashells by the seashore
peter piper picked a peck of pickled peppers
how much wood would a woodchuck chuck
if a woodchuck could chuck wood
""".strip().lower()

print("=" * 60)
print("CHARACTER-LEVEL LANGUAGE MODEL")
print("=" * 60)
print()
print("Our training text (corpus):")
print(CORPUS[:200] + "...")
print(f"\nTotal characters in corpus: {len(CORPUS)}")


# =============================================================================
# STEP 1: COUNT CHARACTER PAIRS (BIGRAMS)
# =============================================================================

print()
print("=" * 60)
print("STEP 1: Count character pairs (bigrams)")
print("=" * 60)

# We go through the text two characters at a time and count how often
# each pair appears.
#
# Example in "the cat":
#   't' -> 'h'  (t is followed by h)
#   'h' -> 'e'  (h is followed by e)
#   'e' -> ' '  (e is followed by space)
#   ' ' -> 'c'  (space is followed by c)
#   'c' -> 'a'  (c is followed by a)
#   'a' -> 't'  (a is followed by t)

# bigram_counts[char1][char2] = how many times char2 followed char1
bigram_counts = defaultdict(Counter)

for i in range(len(CORPUS) - 1):
    current_char = CORPUS[i]
    next_char = CORPUS[i + 1]
    bigram_counts[current_char][next_char] += 1

# Show some interesting bigrams
print("\nWhat characters can follow 't'?")
t_followers = bigram_counts['t']
for char, count in sorted(t_followers.items(), key=lambda x: -x[1])[:8]:
    display = repr(char)  # repr() shows spaces as '\n' etc, for clarity
    print(f"  't' -> {display:6s}: {count} times")

print("\nWhat characters can follow 'q'?")
if 'q' in bigram_counts:
    q_followers = bigram_counts['q']
    for char, count in sorted(q_followers.items(), key=lambda x: -x[1]):
        display = repr(char)
        print(f"  'q' -> {display:6s}: {count} times")
else:
    print("  'q' not in corpus!")

print("\nWhat characters can follow a space?")
space_followers = bigram_counts[' ']
for char, count in sorted(space_followers.items(), key=lambda x: -x[1])[:8]:
    display = repr(char)
    print(f"  ' ' -> {display:6s}: {count} times")


# =============================================================================
# FIRST PRINCIPLES: BIGRAM PROBABILITIES FROM MAXIMUM LIKELIHOOD ESTIMATION
# =============================================================================
#
# HOW DO WE GET PROBABILITIES FROM COUNTS?
#
# The formula we use is:
#   P(c2 | c1) = count(c1, c2) / count(c1)
#
# This is called Maximum Likelihood Estimation (MLE). But WHY is it the best?
#
# DERIVATION OF MLE:
# ------------------
# We observe a sequence of characters. We want to find the probabilities
# that MAXIMIZE the likelihood of seeing this exact sequence.
#
# The likelihood of the data is:
#   L = product of P(c_{i+1} | c_i) for all consecutive pairs
#     = product of P(c2 | c1)^count(c1, c2)   for all pairs (c1, c2)
#
# Taking the log (since log is monotonic, maximizing log L = maximizing L):
#   log L = sum over all pairs (c1, c2) of:  count(c1, c2) * log P(c2 | c1)
#
# We want to maximize log L subject to the constraint that for each c1:
#   sum over all c2 of P(c2 | c1) = 1   (probabilities must sum to 1)
#
# Using Lagrange multipliers (or just reasoning about it):
#   Taking the derivative of log L with respect to P(c2 | c1) and setting = 0:
#   count(c1, c2) / P(c2 | c1) = lambda   (where lambda is the Lagrange multiplier)
#
#   Solving: P(c2 | c1) = count(c1, c2) / lambda
#   Using the constraint sum P(c2 | c1) = 1:
#     lambda = sum of count(c1, c2) over all c2 = count(c1)
#
#   Therefore: P(c2 | c1) = count(c1, c2) / count(c1)    <-- the MLE formula!
#
# This proves that dividing counts by totals is the OPTIMAL estimator
# (in the maximum likelihood sense).
#
# SPACE COMPLEXITY:
# -----------------
# The bigram table stores P(c2 | c1) for all pairs.
#   - If alphabet size = |A|, then we need |A|^2 entries.
#   - For 26 lowercase letters + space + newline = 28 chars: 28^2 = 784 entries.
#   - For full ASCII (128 chars): 128^2 = 16,384 entries.
#   - For full Unicode (~150,000 chars): 150,000^2 = 22.5 BILLION entries!
#     This is why Unicode models use sparse storage (only store non-zero counts).
#
# In our code, we use defaultdict(Counter) which is sparse storage —
# it only stores pairs that actually appeared in the corpus.
#
# =============================================================================


# =============================================================================
# STEP 2: CONVERT COUNTS TO PROBABILITIES
# =============================================================================

print()
print("=" * 60)
print("STEP 2: Convert counts to probabilities")
print("=" * 60)

# Right now we have counts. We need probabilities (numbers between 0 and 1
# that add up to 1.0).
#
# Probability = count / total_count
#
# For example, if after 't' we see:
#   'h': 15 times
#   'o': 8 times
#   'e': 5 times
#   total: 28 times
# Then:
#   P(h | t) = 15/28 = 0.536
#   P(o | t) = 8/28  = 0.286
#   P(e | t) = 5/28  = 0.179

# bigram_probs[char1] = {char2: probability, ...}
bigram_probs = {}

for char1, followers in bigram_counts.items():
    total = sum(followers.values())
    bigram_probs[char1] = {char2: count / total
                           for char2, count in followers.items()}

# Show probabilities for 't'
print("\nProbabilities for what follows 't':")
t_probs = bigram_probs['t']
for char, prob in sorted(t_probs.items(), key=lambda x: -x[1])[:8]:
    display = repr(char)
    bar = "#" * int(prob * 40)
    print(f"  't' -> {display:6s}: {prob:.3f}  {bar}")

# Sanity check: all probabilities for a given character should sum to 1.0
total_prob = sum(bigram_probs['t'].values())
print(f"\nSum of all probabilities after 't': {total_prob:.6f}  (should be 1.0)")


# =============================================================================
# STEP 3: GENERATE TEXT BY SAMPLING
# =============================================================================

print()
print("=" * 60)
print("STEP 3: Generate text by sampling")
print("=" * 60)

# "Sampling" means: pick the next character randomly, but weighted by
# probability. Characters with higher probability get picked more often.
#
# This is like rolling a weighted die.

def sample_next_char(current_char, probs_dict):
    """
    Given the current character, sample the next character
    based on the learned probabilities.

    Returns a random character (weighted by probability),
    or a space if the current character was never seen.
    """
    if current_char not in probs_dict:
        return ' '  # fallback if character not in training data

    chars = list(probs_dict[current_char].keys())
    probs = list(probs_dict[current_char].values())

    # random.choices picks from 'chars' with weights 'probs'
    chosen = random.choices(chars, weights=probs, k=1)[0]
    return chosen


def generate_text(start_char, length, probs_dict):
    """
    Generate 'length' characters starting from 'start_char'.
    Each character is sampled based on the previous character.
    """
    result = [start_char]
    current = start_char

    for _ in range(length - 1):
        next_char = sample_next_char(current, probs_dict)
        result.append(next_char)
        current = next_char

    return "".join(result)


# Generate some text!
print("\nGenerated text (starting from 't', 100 characters):")
generated = generate_text('t', 100, bigram_probs)
print(f"  '{generated}'")

print("\nGenerated text (starting from 'a', 100 characters):")
generated2 = generate_text('a', 100, bigram_probs)
print(f"  '{generated2}'")

print("\nGenerated text (starting from space, 150 characters):")
generated3 = generate_text(' ', 150, bigram_probs)
print(f"  '{generated3}'")

print("""
Notice:
  - The text is gibberish! But it has some patterns from English.
  - Common letter combinations (like 'th', 'he', 'an') appear often.
  - This is because our model learned REAL English character patterns.
  - A character bigram model has no concept of words or meaning.
""")


# =============================================================================
# STEP 4: ANALYZING THE MODEL
# =============================================================================

print("=" * 60)
print("STEP 4: Analyzing what the model learned")
print("=" * 60)

# Let's look at the full bigram probability matrix
all_chars = sorted(bigram_probs.keys())
print(f"\nThe model knows {len(all_chars)} unique characters:")
print(repr("".join(all_chars)))

# Find the most "surprising" pairs (pairs with high probability)
print("\nMost predictable character pairs (probability > 0.5):")
for char1 in sorted(bigram_probs.keys()):
    for char2, prob in bigram_probs[char1].items():
        if prob > 0.5:
            pair = repr(char1) + " -> " + repr(char2)
            print(f"  {pair:20s}: {prob:.3f}")

# Show how many unique characters follow each character
print("\nHow many different characters can follow each character?")
for char in sorted(bigram_probs.keys()):
    num_followers = len(bigram_probs[char])
    display = repr(char)
    bar = "#" * num_followers
    print(f"  {display:6s}: {num_followers:2d} possible next chars  {bar}")


# =============================================================================
# STEP 5: CONNECTION TO MODERN LLMs
# =============================================================================

print()
print("=" * 60)
print("STEP 5: How this connects to GPT and LLaMA")
print("=" * 60)

print("""
Our bigram model does this:
  - Look at 1 character
  - Predict the next character
  - Using counts/probabilities from training text

Modern LLMs (like GPT-4, LLaMA) do this:
  - Look at thousands of tokens (words/subwords)
  - Predict the next token
  - Using a neural network trained on trillions of tokens

The CORE IDEA is identical:
  "Given what came before, what comes next?"

The differences:
  1. Context window: we look at 1 char; GPT-4 looks at 128,000 tokens
  2. Tokens: we use characters; GPT uses subword pieces (BPE)
  3. Probabilities: we use simple counts; GPT uses a giant neural network
  4. Training data: we use 10 sentences; GPT uses the whole internet

But fundamentally, ALL language models are doing the same thing we did:
  Learning P(next_token | previous_tokens)
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Every character in bigram_probs should have probabilities that sum to 1.0
for _char in bigram_probs:
    _total = sum(bigram_probs[_char].values())
    assert abs(_total - 1.0) < 1e-6, (
        f"Probabilities for '{_char}' sum to {_total}, not 1.0"
    )

# 't' should be in bigram_probs (it appears many times in the corpus)
assert 't' in bigram_probs

# After 't', 'h' should be one of the possible next characters (from "the", "that")
assert 'h' in bigram_probs['t']

# All probabilities must be between 0 and 1
for _char in bigram_probs:
    for _next_char, _prob in bigram_probs[_char].items():
        assert 0.0 <= _prob <= 1.0, (
            f"P({_next_char}|{_char}) = {_prob} is not in [0,1]"
        )

# generate_text should return a string of the requested length
_gen = generate_text('t', 50, bigram_probs)
assert isinstance(_gen, str)
assert len(_gen) == 50

# generate_text must start with the given start character
assert _gen[0] == 't'

# sample_next_char returns a single character string
_next = sample_next_char('t', bigram_probs)
assert isinstance(_next, str)
assert len(_next) == 1

# Fallback: unknown character returns a space
assert sample_next_char('Z', bigram_probs) == ' '

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================

print("=" * 60)
print("EXERCISES — try these yourself!")
print("=" * 60)
print("""
Exercise 1:
  Change CORPUS to a different text (try a song, a poem, or nursery
  rhymes). How does the generated text change? Does it start to sound
  more like the new text?
  Hint: Just replace the CORPUS string at the top of the file.

Exercise 2:
  Right now we look at 1 character to predict the next (bigram model).
  What if we looked at 2 characters? This is called a "trigram model".
  Try building bigram_counts for 2-character keys instead of 1:
    current_pair = CORPUS[i:i+2]   (2 chars at a time)
    next_char = CORPUS[i+2]
  Does the generated text look more like English?
  Hint: for i in range(len(CORPUS) - 2):

Exercise 3:
  Instead of random sampling, try "greedy" generation:
  always pick the MOST PROBABLE next character (not random).
  Does the text look better or worse? Does it get stuck in loops?
  Hint: max(probs_dict[current_char], key=probs_dict[current_char].get)

Exercise 4 (challenge):
  Calculate the "perplexity" of the model on the training text.
  Perplexity measures how surprised the model is by the text.
  Lower perplexity = model fits the text better.
  Formula: compute the log-probability of each character, average them,
  then raise e to the negative of that average.
  Hint: import math; math.log(probability)

Exercise 5 (MLE calculation):
  Given these bigram counts starting with 't':
    "th" appears 100 times
    "ti" appears  30 times
    "ta" appears  20 times
    "te" appears  50 times
  (Assume these are the ONLY followers of 't' for simplicity.)

  Calculate:
    total count of 't' = 100 + 30 + 20 + 50 = 200

    P(h|t) = 100 / 200 = 0.50   (50% — 'h' is the most likely after 't')
    P(i|t) =  30 / 200 = 0.15   (15%)
    P(a|t) =  20 / 200 = 0.10   (10%)
    P(e|t) =  50 / 200 = 0.25   (25%)

    Check: 0.50 + 0.15 + 0.10 + 0.25 = 1.00  (sums to 1, as required)

  This makes sense: in English, "th" (as in "the", "that", "this") is
  by far the most common bigram starting with 't'.
""")
