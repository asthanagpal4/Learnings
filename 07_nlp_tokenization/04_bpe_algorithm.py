# HOW TO RUN:
#   uv run python 07_nlp_tokenization/04_bpe_algorithm.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE: 04_bpe_algorithm.py
# TOPIC: BPE Algorithm — Step-by-step implementation
#
# KEY IDEA:
#   BPE (Byte Pair Encoding) was originally a data compression algorithm
#   from 1994. In 2016, it was adapted for NLP by Sennrich et al.
#   Today it powers GPT-2, GPT-3, GPT-4, LLaMA, Mistral, and most modern LLMs.
#
#   THE BPE ALGORITHM:
#     Start with text split into individual characters.
#     Repeat N times:
#       1. Count all adjacent pairs of tokens
#       2. Find the most frequent pair (e.g., 't' + 'h')
#       3. Merge that pair into a new token ('th')
#       4. Replace all occurrences of the pair with the new token
#       5. Record the merge rule
#
#   After N merges, you have a vocabulary of characters + merged pieces.
#   Common sequences (like "the", "ing", "tion") become single tokens.
#
#   The merge RULES are saved. They are applied in order to tokenize new text.
#
# WHY "BYTE PAIR ENCODING"?
#   "Byte" = the original version worked on raw bytes (0-255), not characters.
#   "Pair" = we merge pairs of adjacent items.
#   "Encoding" = it's a way to encode/compress text.
# =============================================================================

import re
from collections import Counter, defaultdict

# =============================================================================
# TRAINING CORPUS
# =============================================================================
# We use a small corpus so the merges are visible and easy to understand.
# In real LLMs, this would be terabytes of text.

CORPUS = """
low lower lowest newer newest wider widest
the cat sat on the mat the cat ate a fat rat
the dog ran to the log and the frog sat on the log
new news newspaper showed that lower prices help
low lower lowest slow slower slowest
""".strip()

print("=" * 60)
print("BPE (BYTE PAIR ENCODING) ALGORITHM — Step by Step")
print("=" * 60)
print()
print("Training corpus:")
print(CORPUS)


# =============================================================================
# HELPER: PREPARE CORPUS
# =============================================================================

def get_word_frequencies(text):
    """
    Split text into words and count how often each word appears.

    We represent each word as a TUPLE of characters with a special
    end-of-word marker '</w>'. This marker is important!

    Why </w>? Because "low" and "lower" both start with 'l','o','w'.
    Without </w>, we couldn't tell where one word ends and the next begins
    after merging. The marker lets us track word boundaries.

    Example:
      "low" -> ('l', 'o', 'w', '</w>')
      "lower" -> ('l', 'o', 'w', 'e', 'r', '</w>')
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    words = text.split()

    freq = Counter(words)

    # Convert each word to a tuple of characters + end marker
    word_freqs = {}
    for word, count in freq.items():
        char_tuple = tuple(list(word) + ['</w>'])
        word_freqs[char_tuple] = count

    return word_freqs


# =============================================================================
# STEP 1: INITIAL VOCABULARY
# =============================================================================

print()
print("=" * 60)
print("STEP 1: Start with characters as tokens")
print("=" * 60)

word_freqs = get_word_frequencies(CORPUS)

print("\nWord frequencies in corpus (as character tuples):")
for word_tuple, count in sorted(word_freqs.items(), key=lambda x: -x[1])[:12]:
    word_str = "".join(w if w != '</w>' else '_' for w in word_tuple)
    print(f"  {str(word_tuple):45s}  (appears {count}x)")

# Build initial vocabulary from all unique characters
initial_vocab = set()
for word_tuple in word_freqs.keys():
    for char in word_tuple:
        initial_vocab.add(char)

print(f"\nInitial vocabulary (just characters):")
print(sorted(initial_vocab))
print(f"Initial vocabulary size: {len(initial_vocab)}")


# =============================================================================
# STEP 2: COUNT PAIRS
# =============================================================================

def count_pairs(word_freqs):
    """
    Count how often each adjacent pair of tokens appears across all words.

    For each word (a tuple of tokens), look at every consecutive pair.
    Multiply by the word's frequency (since it appears that many times).

    Example:
      word = ('l', 'o', 'w', '</w>') with frequency 3
      pairs: ('l','o'), ('o','w'), ('w','</w>')
      each pair gets +3 to its count
    """
    pair_counts = Counter()

    for word_tuple, freq in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_counts[pair] += freq

    return pair_counts


print()
print("=" * 60)
print("STEP 2: Count adjacent pairs")
print("=" * 60)

initial_pairs = count_pairs(word_freqs)
print("\nMost frequent adjacent character pairs:")
for pair, count in initial_pairs.most_common(15):
    left  = pair[0] if pair[0] != '</w>' else '_'
    right = pair[1] if pair[1] != '</w>' else '_'
    bar = "#" * count
    print(f"  '{left}' + '{right}' = '{left+right}':  {count}  {bar}")


# =============================================================================
# STEP 3: MERGE A PAIR
# =============================================================================

def merge_pair(word_freqs, pair_to_merge):
    """
    Find every occurrence of 'pair_to_merge' in every word
    and replace it with a single merged token.

    Example:
      pair_to_merge = ('l', 'o')
      ('l', 'o', 'w', '</w>') -> ('lo', 'w', '</w>')
      ('l', 'o', 'g', '</w>') -> ('lo', 'g', '</w>')
    """
    new_word_freqs = {}
    left, right = pair_to_merge
    merged = left + right  # the new token

    for word_tuple, freq in word_freqs.items():
        new_tuple = []
        i = 0
        while i < len(word_tuple):
            # If current and next token match the pair, merge them
            if (i < len(word_tuple) - 1 and
                word_tuple[i] == left and
                word_tuple[i + 1] == right):
                new_tuple.append(merged)
                i += 2  # skip both tokens (we merged them)
            else:
                new_tuple.append(word_tuple[i])
                i += 1

        new_word_freqs[tuple(new_tuple)] = freq

    return new_word_freqs


# =============================================================================
# FIRST PRINCIPLES: BPE TRAINING COMPLEXITY ANALYSIS
# =============================================================================
#
# HOW EXPENSIVE IS BPE TRAINING?
#
# Let:
#   n = total length of the corpus (in tokens/characters)
#   M = number of merges to perform (num_merges)
#   V = initial vocabulary size (number of unique characters)
#
# NAIVE IMPLEMENTATION (what we do here):
#   Each merge step:
#     1. Count all pairs: O(n) — scan entire corpus
#     2. Find the most frequent pair: O(number of unique pairs) <= O(n)
#     3. Replace all occurrences: O(n) — scan entire corpus again
#   Total per merge: O(n)
#   Total for M merges: O(M * n)
#
#   For GPT-2: M = 50,000, n ~ 10 billion characters
#   Naive cost: 50,000 * 10,000,000,000 = 500 TRILLION operations!
#   This is too slow for real use.
#
# EFFICIENT IMPLEMENTATION (used in practice):
#   Use a priority queue (max-heap) to track the most frequent pair:
#     - Initial pair counting: O(n)
#     - Each merge: update only affected pairs: O(occurrences of merged pair)
#     - Priority queue operations: O(log n) per update
#   Total: O(n + M * average_occurrences * log n)
#   In practice, this is much faster because each merge only affects
#   a small fraction of the corpus.
#
# WHY GREEDY PAIR MERGING APPROXIMATES OPTIMAL COMPRESSION:
#   BPE greedily replaces the most frequent pair at each step.
#   This gives the biggest reduction in sequence length per merge:
#     - If pair (a, b) appears f times, merging it reduces length by f
#     - Choosing the most frequent pair maximizes f
#   This is a greedy approximation to dictionary-based compression.
#   It is not globally optimal (a less frequent pair might enable better
#   future merges), but it works very well in practice.
#
# SPACE COMPLEXITY:
#   - Vocabulary grows by exactly 1 token per merge
#   - After M merges from base vocab of size V: total vocab = V + M
#   - GPT-2: V = 256 (bytes) + 50,000 merges = 50,256 + 1 special = 50,257
#   - We also store the merge rules: O(M) space
#   - The corpus representation: O(n) space (shrinks slightly after each merge)
#
# =============================================================================


# =============================================================================
# STEP 4: RUN THE FULL BPE TRAINING LOOP
# =============================================================================

print()
print("=" * 60)
print("STEP 4: Run BPE merges")
print("=" * 60)

NUM_MERGES = 20  # How many merge operations to perform

# We'll track the vocabulary and merge rules
vocabulary = set(initial_vocab)
merge_rules = []  # list of (pair, merged_token) in order

# Make a working copy of word_freqs
current_word_freqs = dict(word_freqs)

print(f"\nRunning {NUM_MERGES} BPE merges...\n")
print(f"{'Merge':5s}  {'Pair':20s}  {'New Token':15s}  {'Count':8s}  {'Vocab Size':10s}")
print("-" * 65)

for merge_num in range(1, NUM_MERGES + 1):
    # Count all pairs in current tokenization
    pair_counts = count_pairs(current_word_freqs)

    if not pair_counts:
        print("No more pairs to merge!")
        break

    # Find the most frequent pair
    best_pair = pair_counts.most_common(1)[0]
    pair, count = best_pair

    # Create new merged token
    new_token = pair[0] + pair[1]

    # Record this merge rule
    merge_rules.append((pair, new_token))

    # Add new token to vocabulary
    vocabulary.add(new_token)

    # Apply the merge to all words
    current_word_freqs = merge_pair(current_word_freqs, pair)

    # Display the pair nicely
    left  = pair[0] if pair[0] != '</w>' else '_'
    right = pair[1] if pair[1] != '</w>' else '_'
    merged_display = new_token.replace('</w>', '_')
    pair_display = f"'{left}' + '{right}'"

    print(f"  {merge_num:3d}  {pair_display:20s}  '{merged_display:13s}'  {count:8d}  {len(vocabulary):10d}")

print()
print(f"Final vocabulary size: {len(vocabulary)}")


# =============================================================================
# STEP 5: SHOW THE VOCABULARY
# =============================================================================

print()
print("=" * 60)
print("STEP 5: The learned vocabulary")
print("=" * 60)

# Separate single characters from multi-character tokens
single_chars = sorted([t for t in vocabulary if len(t) == 1])
multi_chars  = sorted([t for t in vocabulary if len(t) > 1 and '</w>' not in t])
end_tokens   = sorted([t for t in vocabulary if '</w>' in t])

print(f"\nSingle characters ({len(single_chars)}):")
print("  " + "  ".join(f"'{c}'" for c in single_chars))

print(f"\nMulti-character tokens — no word boundary ({len(multi_chars)}):")
print("  " + "  ".join(f"'{t}'" for t in multi_chars))

print(f"\nTokens ending with </w> — complete words ({len(end_tokens)}):")
display_ends = [t.replace('</w>', '_') for t in end_tokens]
print("  " + "  ".join(f"'{t}'" for t in display_ends))


# =============================================================================
# STEP 6: SHOW HOW WORDS ARE NOW TOKENIZED
# =============================================================================

print()
print("=" * 60)
print("STEP 6: How words look after all merges")
print("=" * 60)

print("\nWord representations after BPE merges:")
print("(Each row shows one word split into its current tokens)")
print()

for word_tuple, freq in sorted(current_word_freqs.items(), key=lambda x: -x[1]):
    # Convert </w> to _ for display
    display_tokens = [t.replace('</w>', '_') for t in word_tuple]
    original_word  = "".join(t.replace('</w>', '') for t in word_tuple)
    n_tokens = len(word_tuple)
    print(f"  '{original_word:12s}' -> {display_tokens}  ({n_tokens} tokens)")


# =============================================================================
# STEP 7: DISPLAY THE MERGE RULES
# =============================================================================

print()
print("=" * 60)
print("STEP 7: The merge rules (the trained tokenizer)")
print("=" * 60)

print("""
These rules are the TRAINED TOKENIZER.
To tokenize new text, apply these rules IN ORDER.
First rule was learned first (most frequent pair overall).
Later rules refine the tokenization further.
""")

print("Merge rules (in order learned):")
for i, (pair, merged) in enumerate(merge_rules, 1):
    left   = pair[0].replace('</w>', '_')
    right  = pair[1].replace('</w>', '_')
    merged_display = merged.replace('</w>', '_')
    print(f"  Rule {i:2d}: '{left}' + '{right}' -> '{merged_display}'")


# =============================================================================
# STEP 8: APPLYING MERGE RULES TO NEW TEXT
# =============================================================================

print()
print("=" * 60)
print("STEP 8: Apply the trained tokenizer to new text")
print("=" * 60)

def apply_bpe(word, merge_rules):
    """
    Tokenize a single word using the learned merge rules.

    Start with the word split into characters + </w>.
    Apply each merge rule in order.
    Stop when no more rules can be applied.
    """
    # Start: split into characters + end marker
    tokens = list(word) + ['</w>']

    # Apply each merge rule in order
    for pair, merged in merge_rules:
        left, right = pair
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and
                tokens[i] == left and
                tokens[i + 1] == right):
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens


test_words = ["low", "lower", "lowest", "newer", "newest", "wider",
              "newlow", "slow", "slower"]

print("\nTokenizing test words with learned BPE rules:")
print()
for word in test_words:
    tokens = apply_bpe(word, merge_rules)
    display = [t.replace('</w>', '_') for t in tokens]
    print(f"  '{word:12s}' -> {display}  ({len(tokens)} tokens)")

print("""
Observations:
  - 'low' becomes a single token (it was common enough to merge fully)
  - 'lower' = 'low' + 'er_'  (shares the 'low' piece with 'low'!)
  - 'newest' = 'new' + 'est_' (shares pieces with 'new', 'newest')
  - 'newlow' (a made-up word!) still gets tokenized — no OOV problem!
  - Shared morphemes like 'low', 'new', 'er_', 'est_' are reused
""")


# =============================================================================
# HOW THIS SCALES TO REAL LLMs
# =============================================================================

print("=" * 60)
print("How this scales to GPT / LLaMA")
print("=" * 60)

print("""
Our demo:
  Corpus: ~100 words
  Merges: 20
  Final vocab size: ~40 tokens

GPT-2 (OpenAI, 2019):
  Corpus: ~40 GB of web text
  Merges: 50,000
  Final vocab size: 50,257 tokens

LLaMA (Meta, 2023):
  Corpus: ~1 TB of web text, books, code
  Merges: ~32,000
  Final vocab size: 32,000 tokens

The algorithm is EXACTLY the same. Just much more data and more merges.
More merges = longer tokens = fewer tokens per sentence.

The key insight: BPE is deterministic and fast to apply.
  Training: slow (needs to process all text)
  Inference: fast (just look up merge rules in order)

This is why the tokenizer is trained ONCE and then saved for reuse.
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# get_word_frequencies: each word becomes a tuple ending with '</w>'
_freqs = get_word_frequencies("low lower low")
assert ('l', 'o', 'w', '</w>') in _freqs        # "low" appears
assert _freqs[('l', 'o', 'w', '</w>')] == 2      # "low" appears twice
assert ('l', 'o', 'w', 'e', 'r', '</w>') in _freqs  # "lower" appears

# count_pairs: correctly counts adjacent pairs weighted by frequency
_test_freqs = {('a', 'b', '</w>'): 3}
_pairs = count_pairs(_test_freqs)
assert _pairs[('a', 'b')] == 3
assert _pairs[('b', '</w>')] == 3
assert ('a', '</w>') not in _pairs  # 'a' and '</w>' are not adjacent

# merge_pair: merges the chosen pair in all words
_before = {('l', 'o', 'w', '</w>'): 2, ('l', 'o', 'g', '</w>'): 1}
_after = merge_pair(_before, ('l', 'o'))
assert ('lo', 'w', '</w>') in _after
assert ('lo', 'g', '</w>') in _after
assert _after[('lo', 'w', '</w>')] == 2
assert _after[('lo', 'g', '</w>')] == 1

# merge_pair: leaves words that don't contain the pair unchanged
_before2 = {('c', 'a', 't', '</w>'): 1}
_after2 = merge_pair(_before2, ('l', 'o'))
assert _after2 == _before2   # no change — 'lo' not in 'cat'

# apply_bpe: tokenizes a known word correctly using learned rules
# "low" should become a compact representation after training on our corpus
_low_tokens = apply_bpe("low", merge_rules)
assert isinstance(_low_tokens, list)
assert len(_low_tokens) >= 1
# The last token of any word must end with '</w>'
assert _low_tokens[-1].endswith('</w>')

# apply_bpe: every word's tokens join back to the original word + </w>
for _word in ["low", "lower", "newer"]:
    _toks = apply_bpe(_word, merge_rules)
    _reconstructed = "".join(_toks)
    assert _reconstructed == _word + '</w>', (
        f"Reconstruction failed for '{_word}': got '{_reconstructed}'"
    )

# merge_rules is a list of (pair, merged_token) tuples
assert len(merge_rules) == NUM_MERGES
for _pair, _merged in merge_rules:
    assert isinstance(_pair, tuple) and len(_pair) == 2
    assert _merged == _pair[0] + _pair[1]

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================

print("=" * 60)
print("EXERCISES — try these yourself!")
print("=" * 60)
print("""
Exercise 1:
  Change NUM_MERGES to 5, 10, 30. How does the vocabulary and
  tokenization change? What happens when you use too few merges?
  What happens with too many?

Exercise 2:
  Add more text to CORPUS. Try adding a few sentences about a topic
  you like. Which new tokens appear after running BPE?
  Does the vocabulary make sense given your new text?

Exercise 3:
  The end-of-word marker </w> is important. Try removing it:
    char_tuple = tuple(list(word))  # no </w>
  How does this break the tokenization?
  Hint: think about how "lower" and "low" look without the marker.

Exercise 4 (challenge):
  Implement a function that applies BPE in the WRONG order (reversed rules).
  Does it produce worse tokenization? Why does order matter?
  Hint: reverse the merge_rules list before applying.

Exercise 5 (challenge):
  Count how many characters the original text has vs the total tokens
  after BPE. This is the "compression ratio". Compare at different
  numbers of merges (5, 10, 20, 30). Does more merges = better compression?

Exercise 6 (manual BPE trace):
  Given the string "aaabdaaabac", trace through 3 BPE merges manually.

  Start: a a a b d a a a b a c

  Step 1: Count pairs:
    (a,a) = 4 times  (positions 0-1, 1-2, 5-6, 6-7)
    (a,b) = 2 times  (positions 2-3, 7-8)
    (b,d) = 1 time   (position 3-4)
    (d,a) = 1 time   (position 4-5)
    (b,a) = 1 time   (position 8-9)
    (a,c) = 1 time   (position 9-10)
  Most frequent: (a,a) with count 4
  Merge: aa
  Result: aa a b d aa a b a c

  Step 2: Count pairs:
    (aa,a) = 2 times
    (a,b)  = 2 times
    (b,d)  = 1 time
    (d,aa) = 1 time
    (b,a)  = 1 time
    (a,c)  = 1 time
  Most frequent: (aa,a) or (a,b) — both have count 2 (tie-break by first found)
  Merge: aaa  (merging aa+a)
  Result: aaa b d aaa b a c

  Step 3: Count pairs:
    (aaa,b) = 2 times
    (b,d)   = 1 time
    (d,aaa) = 1 time
    (b,a)   = 1 time
    (a,c)   = 1 time
  Most frequent: (aaa,b) with count 2
  Merge: aaab
  Result: aaab d aaab a c

  After 3 merges: the string went from 11 tokens to 5 tokens!
""")
