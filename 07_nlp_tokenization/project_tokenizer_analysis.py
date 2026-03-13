# HOW TO RUN:
#   uv run python 07_nlp_tokenization/project_tokenizer_analysis.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE: project_tokenizer_analysis.py
# TOPIC: Mini-Project — Compare BPE at Different Vocabulary Sizes
#
# KEY IDEA:
#   In this project, we train BPE tokenizers with different numbers of merges
#   and compare what happens:
#     - How does vocabulary size change?
#     - How many tokens does the same text produce?
#     - What is the "compression ratio"?
#     - What do the tokens look like at each vocabulary size?
#
#   COMPRESSION RATIO:
#     = (number of characters in original text) / (number of tokens)
#
#     Example: "the cat" = 6 chars, 2 tokens -> ratio = 3.0
#     Higher ratio = fewer tokens = more compressed = better efficiency.
#
#   WHY THIS MATTERS:
#     LLMs have a maximum "context window" — the maximum number of tokens
#     they can see at once. GPT-4 supports 128,000 tokens.
#     If your tokenizer is inefficient (low compression), you waste context.
#     BPE with more merges = better compression = more text fits in context.
# =============================================================================

import re
from collections import Counter


# =============================================================================
# REUSE: COPY OF BPETokenizer FROM FILE 05
# =============================================================================
# (In a real project, we'd import this from the other file.
#  We copy it here so this file is fully self-contained and runnable alone.)

class BPETokenizer:
    """Complete BPE Tokenizer (same as in 05_bpe_tokenizer_complete.py)."""

    def __init__(self):
        self.merge_rules = []
        self.vocab = {}
        self.id_to_token = {}
        self.END_OF_WORD = '</w>'
        self.UNK = '<unk>'

    def _get_word_frequencies(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        freq = Counter(words)
        word_freqs = {}
        for word, count in freq.items():
            if word:
                word_freqs[tuple(list(word) + [self.END_OF_WORD])] = count
        return word_freqs

    def _count_pairs(self, word_freqs):
        pair_counts = Counter()
        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair_counts[(word_tuple[i], word_tuple[i + 1])] += freq
        return pair_counts

    def _merge_pair(self, word_freqs, pair_to_merge):
        left, right = pair_to_merge
        merged = left + right
        new_word_freqs = {}
        for word_tuple, freq in word_freqs.items():
            new_tuple = []
            i = 0
            while i < len(word_tuple):
                if (i < len(word_tuple) - 1 and
                        word_tuple[i] == left and word_tuple[i + 1] == right):
                    new_tuple.append(merged)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(new_tuple)] = freq
        return new_word_freqs

    def _build_vocabulary(self, base_chars, merge_rules):
        vocab = {self.UNK: 0}
        current_id = 1
        for char in sorted(base_chars):
            vocab[char] = current_id
            current_id += 1
        for pair, merged_token in merge_rules:
            if merged_token not in vocab:
                vocab[merged_token] = current_id
                current_id += 1
        id_to_token = {idx: token for token, idx in vocab.items()}
        return vocab, id_to_token

    def train(self, text, num_merges=100, verbose=False):
        word_freqs = self._get_word_frequencies(text)
        base_chars = set()
        for word_tuple in word_freqs.keys():
            for char in word_tuple:
                base_chars.add(char)
        self.merge_rules = []
        current_word_freqs = dict(word_freqs)
        for _ in range(num_merges):
            pair_counts = self._count_pairs(current_word_freqs)
            if not pair_counts:
                break
            best_pair = pair_counts.most_common(1)[0][0]
            count = pair_counts[best_pair]
            if count < 2:
                break
            new_token = best_pair[0] + best_pair[1]
            self.merge_rules.append((best_pair, new_token))
            current_word_freqs = self._merge_pair(current_word_freqs, best_pair)
        self.vocab, self.id_to_token = self._build_vocabulary(base_chars, self.merge_rules)

    def encode(self, text):
        if not self.vocab:
            raise RuntimeError("Tokenizer not trained!")
        if not text or not text.strip():
            return []
        cleaned = re.sub(r'[^a-z\s]', '', text.lower())
        words = cleaned.split()
        all_token_ids = []
        unk_id = self.vocab[self.UNK]
        for word in words:
            if not word:
                continue
            tokens = list(word) + [self.END_OF_WORD]
            for pair, merged in self.merge_rules:
                left, right = pair
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and
                            tokens[i] == left and tokens[i + 1] == right):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            for token in tokens:
                all_token_ids.append(self.vocab.get(token, unk_id))
        return all_token_ids

    def decode(self, token_ids):
        if not token_ids:
            return ""
        text = "".join(self.id_to_token.get(i, self.UNK) for i in token_ids)
        return text.replace(self.END_OF_WORD, ' ').strip()

    def get_vocab(self):
        return dict(self.vocab)

    def vocabulary_size(self):
        return len(self.vocab)


# =============================================================================
# CORPUS — A few paragraphs for training
# =============================================================================

CORPUS = """
Machine learning is a branch of artificial intelligence that focuses on
building systems that learn from data. Instead of being explicitly programmed,
these systems improve their performance on tasks through experience.

Deep learning is a subset of machine learning that uses neural networks
with many layers. These deep neural networks can learn to represent data
with multiple levels of abstraction, from raw pixels or characters all the
way to high-level concepts.

Natural language processing is a field of artificial intelligence that
gives computers the ability to understand, interpret, and generate human
language. It combines linguistics, computer science, and machine learning.

Tokenization is one of the first steps in natural language processing.
It is the process of breaking text down into smaller units called tokens.
These tokens can be words, characters, or subword pieces. The choice of
tokenization strategy significantly impacts the performance of language models.

Byte pair encoding is a data compression algorithm that was adapted for
use in natural language processing. It starts with individual characters
and iteratively merges the most frequent adjacent pairs of tokens.
This process continues until the desired vocabulary size is reached.
The resulting vocabulary contains frequently occurring subword units
such as common word endings like ing and tion or common word beginnings
like pre and un. This allows the model to handle rare and unseen words
by breaking them into known subword pieces.

The transformer architecture revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the
input sequence when producing each part of the output sequence.
Large language models trained on massive text corpora can perform a wide
variety of tasks including translation, summarization, question answering,
and code generation without any task-specific training.
""".strip()


# =============================================================================
# FIRST PRINCIPLES: INFORMATION THEORY AND TOKENIZATION
# =============================================================================
#
# ENTROPY — the fundamental limit on compression:
#
#   Entropy H = -sum of p(x) * log2(p(x)) for all symbols x
#
#   This is the MINIMUM average number of bits needed to encode each symbol.
#   No encoding scheme can do better than this (Shannon's source coding theorem).
#
#   Example: if you have 4 tokens with probabilities [0.5, 0.25, 0.125, 0.125]:
#     H = -(0.5*log2(0.5) + 0.25*log2(0.25) + 0.125*log2(0.125) + 0.125*log2(0.125))
#     H = -(0.5*(-1) + 0.25*(-2) + 0.125*(-3) + 0.125*(-3))
#     H = -(  -0.5   +   -0.5   +   -0.375   +   -0.375  )
#     H = -(-1.75) = 1.75 bits per symbol
#
#   Compare to uniform distribution over 4 items:
#     H = -(4 * 0.25 * log2(0.25)) = -(4 * 0.25 * (-2)) = 2.0 bits per symbol
#
#   The non-uniform distribution has LOWER entropy (1.75 < 2.0) because
#   some symbols are more predictable. This means it can be compressed more!
#
# BITS-PER-CHARACTER for a tokenizer:
#   total_bits = sum of -log2(p(token_i)) for each token in encoded text
#   bits_per_character = total_bits / total_characters
#
#   A GOOD tokenizer produces tokens with lower bits-per-character,
#   meaning it encodes text more efficiently.
#
# WHY ENTROPY IS THE THEORETICAL MINIMUM (Shannon's source coding theorem):
#   Shannon proved in 1948 that:
#   - You CANNOT compress data below its entropy rate (on average)
#   - You CAN get arbitrarily close to the entropy rate with smart enough coding
#   - Any code that assigns shorter codes to more frequent symbols
#     approaches the entropy limit (like Huffman coding)
#
#   For tokenizers, this means:
#   - Frequent tokens (like "the", "is") should be short (small ID, few bits)
#   - Rare tokens can be longer (more bits)
#   - BPE naturally does this: frequent sequences become single tokens
#
# COMPRESSION RATIO ANALYSIS:
#   compression_ratio = original_bytes / encoded_bytes
#   - ratio > 1: text was compressed (fewer bytes after encoding)
#   - ratio = 1: no compression
#   - ratio < 1: expansion (encoding is LARGER than original — bad!)
#
#   For BPE tokenizers:
#   - More merges -> higher compression ratio (up to a point)
#   - The compression ratio is bounded by the entropy of the language
#
# =============================================================================

import math

def compute_entropy(probabilities):
    """
    Compute Shannon entropy: H = -sum(p * log2(p)) for all p > 0.
    Returns the minimum average bits per symbol.
    """
    h = 0.0
    for p in probabilities:
        if p > 0:
            h -= p * math.log2(p)
    return h

# Demonstrate entropy calculation
print("=" * 70)
print("INFORMATION THEORY: Entropy Examples")
print("=" * 70)

# Example 1: non-uniform distribution
probs_nonuniform = [0.5, 0.25, 0.125, 0.125]
h_nonuniform = compute_entropy(probs_nonuniform)
print(f"\nDistribution: {probs_nonuniform}")
print(f"Entropy: {h_nonuniform:.4f} bits per symbol")

# Example 2: uniform distribution over 4 items
probs_uniform = [0.25, 0.25, 0.25, 0.25]
h_uniform = compute_entropy(probs_uniform)
print(f"\nUniform distribution: {probs_uniform}")
print(f"Entropy: {h_uniform:.4f} bits per symbol")

print(f"\nThe non-uniform distribution has lower entropy ({h_nonuniform:.2f} < {h_uniform:.2f})")
print("because some symbols are more predictable, so it can be compressed more.")

# Example 3: very skewed distribution (like natural language)
probs_skewed = [0.7, 0.1, 0.1, 0.05, 0.05]
h_skewed = compute_entropy(probs_skewed)
print(f"\nSkewed distribution: {probs_skewed}")
print(f"Entropy: {h_skewed:.4f} bits per symbol")
print(f"Uniform over 5 items would be: {compute_entropy([0.2]*5):.4f} bits")
print(f"Savings from skew: {compute_entropy([0.2]*5) - h_skewed:.4f} bits per symbol")
print()


# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

# The sentence we'll use to compare tokenization across vocab sizes.
BENCHMARK_SENTENCE = (
    "machine learning tokenization helps deep neural networks understand language"
)

# Vocabulary sizes to test (achieved by varying num_merges).
# We start small and go big so we can see the progression.
MERGE_COUNTS = [10, 30, 60, 100, 150, 200]


# =============================================================================
# RUN THE ANALYSIS
# =============================================================================

print("=" * 70)
print("TOKENIZER ANALYSIS: BPE at Different Vocabulary Sizes")
print("=" * 70)
print()
print("Training corpus length:", len(CORPUS), "characters")
print("Benchmark sentence:", f'"{BENCHMARK_SENTENCE}"')
print(f"Benchmark sentence character count (no spaces): "
      f"{len(BENCHMARK_SENTENCE.replace(' ', ''))}")
print()


# Store results for the summary table
results = []

for num_merges in MERGE_COUNTS:
    print("-" * 70)
    print(f"Training with num_merges = {num_merges}")
    print("-" * 70)

    # Train a fresh tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, num_merges=num_merges)

    vocab_size = tokenizer.vocabulary_size()
    print(f"  Vocabulary size: {vocab_size}")

    # Encode the benchmark sentence
    ids = tokenizer.encode(BENCHMARK_SENTENCE)
    num_tokens = len(ids)
    print(f"  Benchmark tokens: {num_tokens}")

    # Compute compression ratio
    # chars = number of non-space characters in the benchmark
    char_count = len(BENCHMARK_SENTENCE.replace(' ', ''))
    compression_ratio = char_count / num_tokens if num_tokens > 0 else 0
    print(f"  Characters in benchmark: {char_count}")
    print(f"  Compression ratio: {char_count} / {num_tokens} = {compression_ratio:.2f}")

    # Show what the tokens actually look like
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token.get(i, '<unk>') for i in ids]
    display_tokens = [t.replace('</w>', '_') for t in tokens]
    print(f"  Tokens: {display_tokens}")

    # Show the longest tokens in the vocabulary (the most merged)
    all_tokens = [t for t in vocab.keys()
                  if t not in (tokenizer.UNK, tokenizer.END_OF_WORD)
                  and t != tokenizer.END_OF_WORD]

    # Longest tokens = most heavily merged = "whole words"
    longest = sorted(all_tokens, key=lambda t: len(t), reverse=True)
    # Filter out single chars
    multi_char = [t for t in longest if len(t) > 2][:10]
    display_long = [t.replace('</w>', '_') for t in multi_char]
    print(f"  Longest tokens (top 10): {display_long}")

    # Verify roundtrip
    decoded = tokenizer.decode(ids)
    roundtrip_ok = decoded == BENCHMARK_SENTENCE
    print(f"  Roundtrip check: {'PASS' if roundtrip_ok else 'FAIL'}")
    if not roundtrip_ok:
        print(f"    Expected: '{BENCHMARK_SENTENCE}'")
        print(f"    Got:      '{decoded}'")

    print()

    # Store results
    results.append({
        'merges': num_merges,
        'vocab_size': vocab_size,
        'num_tokens': num_tokens,
        'compression_ratio': compression_ratio,
    })


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print()
print(f"{'Merges':>8}  {'Vocab Size':>12}  {'# Tokens':>10}  {'Compression':>12}  {'Efficiency'}")
print("-" * 70)

for r in results:
    merges           = r['merges']
    vocab_size       = r['vocab_size']
    num_tokens       = r['num_tokens']
    compression      = r['compression_ratio']

    # "Efficiency" bar — visual representation of compression ratio
    # Max compression ratio here ~ 10, so scale bar to 20 chars
    bar_len = int(compression * 2)
    bar = "#" * min(bar_len, 30)

    print(f"  {merges:6d}  {vocab_size:12d}  {num_tokens:10d}  {compression:12.2f}  {bar}")

print()


# =============================================================================
# ANALYSIS AND INTERPRETATION
# =============================================================================

print("=" * 70)
print("ANALYSIS")
print("=" * 70)

min_compression = min(r['compression_ratio'] for r in results)
max_compression = max(r['compression_ratio'] for r in results)
min_tokens      = min(r['num_tokens'] for r in results)
max_tokens      = max(r['num_tokens'] for r in results)

print(f"""
From the table above, we can see:

1. COMPRESSION RATIO IMPROVES WITH MORE MERGES
   - Lowest compression (fewest merges):  {min_compression:.2f}x  ({max_tokens} tokens)
   - Highest compression (most merges):   {max_compression:.2f}x  ({min_tokens} tokens)
   - More merges = longer tokens = fewer tokens per sentence

2. VOCABULARY SIZE GROWS WITH MORE MERGES
   - Starting vocabulary: {results[0]['vocab_size']} tokens (chars only + special tokens)
   - After {results[-1]['merges']} merges:  {results[-1]['vocab_size']} tokens
   - Each merge adds exactly 1 new token to the vocabulary

3. DIMINISHING RETURNS
   - Early merges (1-30): big compression gains (merging very common pairs)
   - Later merges (100+): smaller gains (only rare pairs left to merge)
   - Real LLMs stop at 30,000-50,000 merges (balance: good compression,
     manageable vocabulary size)

4. PRACTICAL IMPLICATIONS
   - GPT-4 context window: 128,000 tokens
   - With compression ratio 3.0x, that fits ~384,000 characters of text
   - With compression ratio 5.0x, that fits ~640,000 characters of text
   - Better tokenizer = more text fits in the same context window!

5. WHY NOT JUST USE HUGE VOCABULARIES?
   - Larger vocab = larger model (each token needs its own embedding vector)
   - GPT-2 vocabulary (50,257 tokens) = 50,257 embedding vectors to store
   - Very rare tokens are hard to learn (the model sees them too rarely)
   - Sweet spot for most English LLMs: 30,000 - 65,000 tokens
""")


# =============================================================================
# BONUS: SHOW HOW INDIVIDUAL WORDS TOKENIZE ACROSS VOCAB SIZES
# =============================================================================

print("=" * 70)
print("BONUS: How individual words tokenize at different vocab sizes")
print("=" * 70)

test_words = [
    "tokenization",
    "understanding",
    "machine",
    "learning",
    "networks",
]

print()
print(f"{'Word':>15}  ", end="")
for num_merges in MERGE_COUNTS:
    print(f"  merges={num_merges:3d}", end="")
print()
print("-" * 85)

for word in test_words:
    print(f"  {word:15s}", end="")
    for num_merges in MERGE_COUNTS:
        # Retrain a tokenizer (small cost for demonstration)
        t = BPETokenizer()
        t.train(CORPUS, num_merges=num_merges)
        ids = t.encode(word)
        print(f"  {len(ids):>10d} tok", end="")
    print()

print()
print("(Each number is the number of tokens that word produces at that vocab size)")
print("Lower = better compressed = model needs less 'space' to represent the word")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# All compression ratios must be positive numbers
for _r in results:
    assert _r['compression_ratio'] > 0, (
        f"Compression ratio is not positive for merges={_r['merges']}: "
        f"{_r['compression_ratio']}"
    )

# More merges should generally produce equal or better compression
# (non-decreasing compression ratio)
for _i in range(len(results) - 1):
    assert results[_i]['compression_ratio'] <= results[_i + 1]['compression_ratio'] + 0.01, (
        f"Compression ratio decreased from merges={results[_i]['merges']} "
        f"({results[_i]['compression_ratio']:.2f}) to "
        f"merges={results[_i+1]['merges']} ({results[_i+1]['compression_ratio']:.2f})"
    )

# Vocabulary size grows as merges increase
for _i in range(len(results) - 1):
    assert results[_i]['vocab_size'] <= results[_i + 1]['vocab_size'], (
        f"Vocab size should not decrease when adding more merges"
    )

# Token counts decrease (or stay equal) as merges increase
for _i in range(len(results) - 1):
    assert results[_i]['num_tokens'] >= results[_i + 1]['num_tokens'], (
        f"Token count should not increase with more merges"
    )

# compute_entropy: known value — [0.5, 0.25, 0.125, 0.125] should give 1.75 bits
_h = compute_entropy([0.5, 0.25, 0.125, 0.125])
assert abs(_h - 1.75) < 1e-6, f"Expected entropy 1.75, got {_h}"

# Uniform distribution over 4 items should give 2.0 bits
_h_uniform = compute_entropy([0.25, 0.25, 0.25, 0.25])
assert abs(_h_uniform - 2.0) < 1e-6, f"Expected entropy 2.0, got {_h_uniform}"

# Non-uniform should have LOWER entropy than uniform (more predictable)
assert _h < _h_uniform

# compute_entropy with a single event (certainty) should give 0
assert abs(compute_entropy([1.0]) - 0.0) < 1e-6

# compute_entropy ignores zero probabilities (no log(0) error)
assert compute_entropy([0.5, 0.5, 0.0]) == compute_entropy([0.5, 0.5])

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================

print()
print("=" * 70)
print("EXERCISES — try these yourself!")
print("=" * 70)
print("""
Exercise 1:
  Change BENCHMARK_SENTENCE to a sentence of your own choice.
  How does the compression ratio change for your sentence?
  Try a sentence with long/rare words vs. a sentence with short/common words.
  Hint: Just replace the BENCHMARK_SENTENCE string at the top.

Exercise 2:
  Add more merge counts to MERGE_COUNTS, e.g. [5, 10, 20, 50, 100, 200, 500].
  Plot (mentally) the compression ratio vs. number of merges.
  Where do the "diminishing returns" start for our corpus?
  Hint: Look at how much the compression ratio improves between consecutive rows.

Exercise 3:
  Change CORPUS to a completely different kind of text, like:
    - A nursery rhyme repeated many times
    - A Python code snippet repeated many times
    - A recipe
  How does the vocabulary and compression ratio change?
  What tokens does BPE learn for code vs. English prose?
  Hint: Replace the CORPUS string at the top.

Exercise 4:
  Calculate the average token length (in characters) for each vocab size.
  Formula: average_token_length = total_chars_in_encoded_text / num_tokens
  Do more merges produce longer average tokens?
  Hint: After encoding, convert IDs back to tokens and average their lengths.

Exercise 5 (challenge):
  Implement a function that finds the "most unique" tokens at each vocab size.
  These are tokens that appear in the vocab for a large number of merges but
  NOT in smaller vocab sizes.
  Hint: Train two tokenizers with different num_merges, get their vocab sets,
  and compute the difference with Python's set() subtraction: setA - setB.

Exercise 6 (entropy calculation):
  Calculate the entropy of the distribution [0.5, 0.25, 0.125, 0.125]
  and compare to a uniform distribution over 4 items.

  Work through it:
    Non-uniform: [0.5, 0.25, 0.125, 0.125]
    H = -(0.5 * log2(0.5) + 0.25 * log2(0.25) + 0.125 * log2(0.125) + 0.125 * log2(0.125))
    H = -(0.5 * (-1) + 0.25 * (-2) + 0.125 * (-3) + 0.125 * (-3))
    H = -(-0.5 - 0.5 - 0.375 - 0.375)
    H = 1.75 bits per symbol

    Uniform: [0.25, 0.25, 0.25, 0.25]
    H = -(4 * 0.25 * log2(0.25))
    H = -(4 * 0.25 * (-2))
    H = 2.0 bits per symbol

    The uniform distribution has MAXIMUM entropy (2.0 bits).
    The non-uniform distribution saves 0.25 bits per symbol.
    This means: if you know some symbols are more common, you can
    assign them shorter codes and save space on average.

    Connection to tokenization: a good tokenizer creates a token
    distribution where common text patterns get frequent tokens
    (lower entropy per character = better compression).
""")
