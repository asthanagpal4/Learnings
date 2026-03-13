# HOW TO RUN:
#   uv run python 07_nlp_tokenization/05_bpe_tokenizer_complete.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE: 05_bpe_tokenizer_complete.py
# TOPIC: Complete BPE Tokenizer Class
#
# KEY IDEA:
#   We now package everything from 04_bpe_algorithm.py into a clean,
#   reusable class. This class has the same structure as real tokenizer
#   libraries (like HuggingFace's tokenizers).
#
#   The three core operations are:
#     train(text, num_merges)  — learn merge rules from training text
#     encode(text)             — convert text string -> list of integer IDs
#     decode(token_ids)        — convert list of integer IDs -> text string
#
#   After this file, you will have implemented a working tokenizer from
#   scratch — the same fundamental algorithm used in GPT, LLaMA, and others.
# =============================================================================

import re
from collections import Counter


# =============================================================================
# THE BPETokenizer CLASS
# =============================================================================

class BPETokenizer:
    """
    A complete BPE (Byte Pair Encoding) tokenizer.

    Usage:
        tokenizer = BPETokenizer()
        tokenizer.train("your training text here", num_merges=100)

        ids = tokenizer.encode("new text to tokenize")
        text = tokenizer.decode(ids)
    """

    def __init__(self):
        # merge_rules: ordered list of ((token_a, token_b), merged_token)
        # Order matters! We apply rules in the order they were learned.
        self.merge_rules = []

        # vocab: maps token_string -> integer_id
        self.vocab = {}

        # id_to_token: maps integer_id -> token_string (for decoding)
        self.id_to_token = {}

        # Special tokens
        self.END_OF_WORD = '</w>'   # marks word boundaries
        self.UNK = '<unk>'          # unknown characters not in training data

        print("BPETokenizer created. Call .train(text, num_merges) to train.")

    # -------------------------------------------------------------------------
    # PRIVATE HELPER METHODS (start with _ by convention)
    # -------------------------------------------------------------------------

    def _get_word_frequencies(self, text):
        """
        Break text into words and count each word's frequency.
        Each word is represented as a tuple of characters + </w>.

        Example: "the" -> ('t', 'h', 'e', '</w>')
        """
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()

        freq = Counter(words)

        word_freqs = {}
        for word, count in freq.items():
            if word:  # skip empty strings
                char_tuple = tuple(list(word) + [self.END_OF_WORD])
                word_freqs[char_tuple] = count

        return word_freqs

    def _count_pairs(self, word_freqs):
        """
        Count all adjacent token pairs across the corpus.
        Each pair is weighted by its word's frequency.
        """
        pair_counts = Counter()
        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pair_counts[pair] += freq
        return pair_counts

    def _merge_pair(self, word_freqs, pair_to_merge):
        """
        Apply one merge rule to all words in the corpus.
        Replaces every occurrence of (token_a, token_b) with merged_token.
        """
        left, right = pair_to_merge
        merged = left + right
        new_word_freqs = {}

        for word_tuple, freq in word_freqs.items():
            new_tuple = []
            i = 0
            while i < len(word_tuple):
                if (i < len(word_tuple) - 1 and
                        word_tuple[i] == left and
                        word_tuple[i + 1] == right):
                    new_tuple.append(merged)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(new_tuple)] = freq

        return new_word_freqs

    def _build_vocabulary(self, base_chars, merge_rules):
        """
        Build the vocab and id_to_token dictionaries.

        Vocabulary includes:
          - Special tokens: <unk>
          - All base characters
          - All merged tokens (in order of merging)
        """
        vocab = {}
        current_id = 0

        # Add special tokens first
        vocab[self.UNK] = current_id
        current_id += 1

        # Add base characters (single chars + </w>)
        for char in sorted(base_chars):
            vocab[char] = current_id
            current_id += 1

        # Add merged tokens in order they were learned
        for pair, merged_token in merge_rules:
            if merged_token not in vocab:
                vocab[merged_token] = current_id
                current_id += 1

        # Build reverse mapping
        id_to_token = {idx: token for token, idx in vocab.items()}

        return vocab, id_to_token

    # -------------------------------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------------------------------

    def train(self, text, num_merges=100):
        """
        Learn BPE merge rules from training text.

        Parameters:
          text (str): The training corpus.
          num_merges (int): How many merge operations to perform.
                            More merges = larger vocabulary = longer tokens.

        This method:
          1. Builds initial character-level vocabulary
          2. Runs BPE merges to find the best token merges
          3. Saves merge rules and builds final vocabulary
        """
        print(f"\n--- Training BPETokenizer with {num_merges} merges ---")

        # Get word frequencies
        word_freqs = self._get_word_frequencies(text)
        print(f"  Unique words in corpus: {len(word_freqs)}")

        # Collect base characters (initial vocabulary)
        base_chars = set()
        for word_tuple in word_freqs.keys():
            for char in word_tuple:
                base_chars.add(char)
        print(f"  Initial character vocabulary: {len(base_chars)} chars")

        # Run BPE merges
        self.merge_rules = []
        current_word_freqs = dict(word_freqs)

        for merge_num in range(num_merges):
            pair_counts = self._count_pairs(current_word_freqs)

            if not pair_counts:
                print(f"  Stopped early at merge {merge_num} (no more pairs)")
                break

            # Find best pair
            best_pair = pair_counts.most_common(1)[0][0]
            count = pair_counts[best_pair]

            # Only merge if the pair appears more than once
            if count < 2:
                print(f"  Stopped at merge {merge_num}: all pairs appear < 2 times")
                break

            # Create merged token and record rule
            new_token = best_pair[0] + best_pair[1]
            self.merge_rules.append((best_pair, new_token))

            # Apply merge
            current_word_freqs = self._merge_pair(current_word_freqs, best_pair)

        print(f"  Merge rules learned: {len(self.merge_rules)}")

        # Build vocabulary dictionaries
        self.vocab, self.id_to_token = self._build_vocabulary(
            base_chars, self.merge_rules
        )
        print(f"  Final vocabulary size: {len(self.vocab)} tokens")
        print("  Training complete!")

    # -------------------------------------------------------------------------
    # FIRST PRINCIPLES: ENCODE vs DECODE COMPLEXITY
    # -------------------------------------------------------------------------
    #
    # ENCODE COMPLEXITY:
    #   For a text of length n (characters) and M merge rules:
    #   - Split into words: O(n)
    #   - For each word, apply M merge rules sequentially
    #   - Each merge rule scans the token list: O(word_length)
    #   - Total per word: O(M * word_length)
    #   - Total for all words: O(M * n) worst case
    #
    #   This can be improved with a trie-based lookup:
    #   - Build a trie from all merge rules
    #   - For each position in the word, find the longest matching merge
    #   - Reduces to O(n * max_token_length) which is much better
    #
    # DECODE COMPLEXITY:
    #   - For k tokens: just look up each token string in the vocabulary: O(1) per token
    #   - Concatenate all strings: O(total_output_length)
    #   - Total: O(k) lookups + O(n) concatenation = O(n)
    #   - MUCH faster than encode!
    #
    # WHY ENCODE IS SLOWER THAN DECODE:
    #   Encode must try each merge rule in order (sequential dependency).
    #   Decode just does a dictionary lookup per token (independent).
    #
    # COMPARISON TO OTHER TOKENIZATION METHODS:
    #   - WordPiece (used by BERT): uses likelihood ratio instead of frequency
    #     to decide merges. Picks the pair that maximizes P(ab)/P(a)*P(b).
    #     This favors pairs where a and b co-occur more than expected by chance.
    #   - Unigram (used by T5, mBART): starts with a LARGE vocabulary and
    #     REMOVES tokens that least affect the overall likelihood. Opposite
    #     direction from BPE! Uses the EM algorithm for optimization.
    #   - SentencePiece: a library that implements both BPE and Unigram,
    #     works directly on raw text (no pre-tokenization needed).
    #
    # -------------------------------------------------------------------------

    def encode(self, text):
        """
        Convert a text string to a list of integer token IDs.

        Steps:
          1. Clean and split text into words
          2. For each word, apply BPE merge rules (same as training)
          3. Convert each token to its integer ID
          4. Unknown characters get the <unk> ID

        Parameters:
          text (str): Text to encode.

        Returns:
          list of int: Token IDs.
        """
        if not self.vocab:
            raise RuntimeError("Tokenizer not trained! Call .train() first.")

        # Handle empty string
        if not text or not text.strip():
            return []

        # Clean text the same way as during training
        cleaned = text.lower()
        cleaned = re.sub(r'[^a-z\s]', '', cleaned)
        words = cleaned.split()

        all_token_ids = []
        unk_id = self.vocab[self.UNK]

        for word in words:
            if not word:
                continue

            # Start with characters + end marker
            tokens = list(word) + [self.END_OF_WORD]

            # Apply merge rules in order
            for pair, merged in self.merge_rules:
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

            # Convert tokens to IDs
            for token in tokens:
                if token in self.vocab:
                    all_token_ids.append(self.vocab[token])
                else:
                    # Unknown token: use <unk>
                    all_token_ids.append(unk_id)

        return all_token_ids

    def decode(self, token_ids):
        """
        Convert a list of integer token IDs back to text.

        Steps:
          1. Convert each ID to its token string
          2. Join all tokens together
          3. Replace </w> markers with spaces

        Parameters:
          token_ids (list of int): Token IDs to decode.

        Returns:
          str: Reconstructed text.
        """
        if not self.id_to_token:
            raise RuntimeError("Tokenizer not trained! Call .train() first.")

        if not token_ids:
            return ""

        # Convert IDs back to token strings
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tokens.append(self.id_to_token[tid])
            else:
                tokens.append(self.UNK)

        # Join all tokens and replace end-of-word markers with spaces
        text = "".join(tokens)
        text = text.replace(self.END_OF_WORD, ' ')
        text = text.strip()

        return text

    def get_vocab(self):
        """
        Return the vocabulary dictionary {token_string: integer_id}.
        """
        return dict(self.vocab)

    def show_vocab(self, max_items=50):
        """
        Print the vocabulary in a readable format.
        """
        if not self.vocab:
            print("Vocabulary is empty. Train the tokenizer first.")
            return

        print(f"\nVocabulary ({len(self.vocab)} tokens, showing up to {max_items}):")
        print(f"  {'Token':20s}  {'ID':5s}")
        print("  " + "-" * 30)
        for token, idx in list(self.vocab.items())[:max_items]:
            display = token.replace(self.END_OF_WORD, '_')
            print(f"  {display:20s}  {idx:5d}")
        if len(self.vocab) > max_items:
            print(f"  ... ({len(self.vocab) - max_items} more)")


# =============================================================================
# TRAINING DATA
# =============================================================================

TRAINING_TEXT = """
the cat sat on the mat the cat ate a fat rat
the dog ran on the mat the dog saw the cat
a big dog sat by the log the log was near the bog
the frog sat on a log near the bog in the fog
cats and dogs and frogs and logs
the quick brown fox jumps over the lazy dog
she sells seashells by the seashore
how much wood would a woodchuck chuck
low lower lowest newer newest wider widest
machine learning is a type of artificial intelligence
the tokenizer splits text into meaningful tokens
natural language processing helps computers understand text
deep learning uses neural networks with many layers
""".strip()


# =============================================================================
# DEMO: TRAIN THE TOKENIZER
# =============================================================================

print("=" * 60)
print("COMPLETE BPE TOKENIZER DEMO")
print("=" * 60)

# Create and train the tokenizer
tokenizer = BPETokenizer()
tokenizer.train(TRAINING_TEXT, num_merges=50)


# =============================================================================
# DEMO: SHOW THE VOCABULARY
# =============================================================================

tokenizer.show_vocab(max_items=60)


# =============================================================================
# DEMO: ENCODE TEXT
# =============================================================================

print()
print("=" * 60)
print("ENCODING EXAMPLES")
print("=" * 60)

test_sentences = [
    "the cat sat on the mat",
    "dogs and frogs",
    "machine learning",
    "the quick brown fox",
]

for sentence in test_sentences:
    ids = tokenizer.encode(sentence)
    print(f"\n  Original: '{sentence}'")
    print(f"  Token IDs: {ids}")
    print(f"  Number of tokens: {len(ids)}")
    print(f"  Number of chars:  {len(sentence.replace(' ', ''))}")


# =============================================================================
# DEMO: DECODE BACK TO TEXT
# =============================================================================

print()
print("=" * 60)
print("ENCODE -> DECODE ROUNDTRIP")
print("=" * 60)

roundtrip_tests = [
    "the cat sat on the mat",
    "the dog ran on the mat",
    "cats and dogs",
    "the quick brown fox jumps over the lazy dog",
    "machine learning",
    "neural networks",
]

print("\nTesting that encode then decode gives back the original text:\n")
all_passed = True

for original in roundtrip_tests:
    # Encode to IDs
    ids = tokenizer.encode(original)

    # Decode back to text
    decoded = tokenizer.decode(ids)

    # Check if they match
    # Note: we need to normalize both for comparison
    # (encoding lowercases and removes punctuation)
    original_normalized = re.sub(r'[^a-z\s]', '', original.lower()).strip()
    decoded_normalized = decoded.strip()

    match = original_normalized == decoded_normalized
    status = "PASS" if match else "FAIL"
    if not match:
        all_passed = False

    print(f"  [{status}] Original:  '{original_normalized}'")
    print(f"        Decoded:   '{decoded_normalized}'")
    if not match:
        print(f"        IDs were:  {ids}")
    print()

if all_passed:
    print("All roundtrip tests PASSED! Encode -> Decode is lossless.")
else:
    print("Some tests FAILED. Check the tokenizer logic.")


# =============================================================================
# DEMO: SHOW TOKEN-LEVEL BREAKDOWN
# =============================================================================

print()
print("=" * 60)
print("TOKEN-LEVEL BREAKDOWN")
print("=" * 60)

print("\nSeeing exactly which tokens a sentence splits into:")

breakdown_sentences = [
    "the cat sat on the mat",
    "cats and dogs",
    "low lower lowest",
    "machine learning",
]

vocab = tokenizer.get_vocab()
id_to_token = {v: k for k, v in vocab.items()}

for sentence in breakdown_sentences:
    ids = tokenizer.encode(sentence)
    tokens = [id_to_token.get(i, '<unk>') for i in ids]
    display_tokens = [t.replace('</w>', '_') for t in tokens]

    print(f"\n  '{sentence}'")
    print(f"  Tokens: {display_tokens}")
    print(f"  IDs:    {ids}")


# =============================================================================
# DEMO: HANDLING UNKNOWN CHARACTERS
# =============================================================================

print()
print("=" * 60)
print("EDGE CASES")
print("=" * 60)

edge_cases = [
    "",                         # empty string
    "   ",                      # only spaces
    "hello",                    # short word
    "thequickbrownfox",         # no spaces
    "a b c",                    # single chars
]

print("\nEdge case handling:")
for text in edge_cases:
    try:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"  Input:   {repr(text)}")
        print(f"  IDs:     {ids}")
        print(f"  Decoded: {repr(decoded)}")
        print()
    except Exception as e:
        print(f"  Input: {repr(text)} -> ERROR: {e}")
        print()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
What we built:

  BPETokenizer with 3 main methods:

  1. .train(text, num_merges)
     - Learns {len(tokenizer.merge_rules)} merge rules from training text
     - Builds a vocabulary of {len(tokenizer.vocab)} tokens

  2. .encode(text) -> list of int
     - Applies merge rules to new text
     - Returns list of integer token IDs
     - Unknown characters become <unk>

  3. .decode(ids) -> str
     - Converts IDs back to tokens
     - Joins tokens and strips </w> markers
     - Roundtrip: decode(encode(text)) == text

This is the same algorithm used in:
  - GPT-2, GPT-3, GPT-4 (OpenAI)
  - LLaMA, LLaMA 2, LLaMA 3 (Meta)
  - Mistral, Mixtral (Mistral AI)
  - DALL-E, Codex, and many others
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# The tokenizer must have been trained (vocab is non-empty)
assert len(tokenizer.vocab) > 0, "Tokenizer vocab is empty — was training skipped?"
assert len(tokenizer.merge_rules) > 0, "No merge rules found — was training skipped?"

# encode returns a list of integers
_ids = tokenizer.encode("the cat sat")
assert isinstance(_ids, list)
assert all(isinstance(i, int) for i in _ids)
assert len(_ids) > 0

# encode of empty string returns empty list
assert tokenizer.encode("") == []
assert tokenizer.encode("   ") == []

# decode of empty list returns empty string
assert tokenizer.decode([]) == ""

# encode -> decode roundtrip: decode(encode(text)) matches normalized text
def _normalize(t):
    """Same normalization the tokenizer applies internally."""
    return re.sub(r'[^a-z\s]', '', t.lower()).strip()

for _text in [
    "the cat sat on the mat",
    "dogs and frogs",
    "machine learning",
    "the quick brown fox",
    "cats and dogs",
]:
    _enc = tokenizer.encode(_text)
    _dec = tokenizer.decode(_enc)
    assert _dec == _normalize(_text), (
        f"Roundtrip failed for '{_text}': got '{_dec}', expected '{_normalize(_text)}'"
    )

# vocab contains special token <unk>
assert '<unk>' in tokenizer.vocab

# get_vocab returns a plain dict copy
_v = tokenizer.get_vocab()
assert isinstance(_v, dict)
assert len(_v) == len(tokenizer.vocab)

# Modifying the copy does not affect the tokenizer's internal vocab
_v['_test_key_'] = 99999
assert '_test_key_' not in tokenizer.vocab

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================

print("=" * 60)
print("EXERCISES — try these yourself!")
print("=" * 60)
print("""
Exercise 1:
  Train the tokenizer with different num_merges values: 10, 25, 50, 100.
  For each, encode the sentence "the cat sat on the mat" and count tokens.
  What pattern do you notice?
  Hint: More merges = fewer tokens per sentence.

Exercise 2:
  Add a method .token_count(text) that returns the number of tokens
  a text encodes into. This is useful for estimating API costs!
  (OpenAI charges per token.)
  Hint: return len(self.encode(text))

Exercise 3:
  Add a method .vocabulary_size() that returns len(self.vocab).
  Then add a method .coverage(text) that calculates what percentage
  of tokens in a text are NOT <unk>.
  Hint: count how many IDs are NOT equal to self.vocab[self.UNK]

Exercise 4:
  Try training on a completely different language style — for example,
  a Python code snippet (use lowercase variable names and keywords).
  Does the tokenizer learn code-specific tokens like 'def', 'for', 'if'?
  Hint: add Python keywords to TRAINING_TEXT.

Exercise 5 (challenge):
  Save and load the tokenizer. Write two functions:
    save_tokenizer(tokenizer, filename) — saves merge_rules and vocab to a file
    load_tokenizer(filename) — loads them back
  This is how real tokenizers are distributed (e.g., tokenizer.json)
  Hint: use Python's built-in repr() to write data, or look up json module.

Exercise 6 (encode vs decode analysis):
  Explain why BPE encode is slower than decode, and suggest a data
  structure to speed up encoding.

  Answer:
    ENCODE is slow because:
      - It must apply M merge rules one by one, in order
      - Each rule scans through the token list looking for the pair
      - Rules have sequential dependencies: rule 5 might merge tokens
        created by rule 3, so you can't skip or parallelize

    DECODE is fast because:
      - Each token ID just needs a dictionary lookup: O(1)
      - No dependencies between tokens — each lookup is independent
      - Just concatenate the results

    To speed up encoding, use a TRIE (prefix tree):
      - Build a trie from all tokens in the vocabulary
      - For each position in the input, find the longest matching token
      - This avoids scanning through all M rules for every position
      - Reduces complexity from O(M * n) to O(n * max_token_length)
      - This is similar to how the Aho-Corasick algorithm works

    Another approach: precompute a lookup table mapping every possible
    pair of tokens to its merged result (if a rule exists), turning
    each merge step into O(n) with a small constant.
""")
