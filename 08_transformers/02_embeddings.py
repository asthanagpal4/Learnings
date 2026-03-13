# HOW TO RUN:
#   uv run python 08_transformers/02_embeddings.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# LESSON 2: Embeddings and Positional Encoding
# =============================================================================
#
# THE PROBLEM: Neural networks work with numbers, not words.
# -------------------------------------------------------
# To feed text into a neural network, we need to turn each word (or subword
# token) into a vector of numbers. But HOW?
#
# Approach 1: One-hot encoding (simple but bad)
# Approach 2: Learned embeddings (what transformers use — much better!)
#
# Then there's a second problem: transformers process ALL tokens at once
# (in parallel), so unlike RNNs they have no built-in sense of ORDER.
# We fix this with POSITIONAL ENCODING.
#
# Run this file:
#   uv run python 02_embeddings.py
# =============================================================================

import torch
import torch.nn as nn
import math

print("=" * 60)
print("PART 1: One-Hot Encoding — simple but wasteful")
print("=" * 60)

# --- What is one-hot encoding? ---
# Imagine our vocabulary has 5 words: ["cat", "sat", "on", "the", "mat"]
# We assign each word an ID:
#   cat=0, sat=1, on=2, the=3, mat=4
#
# One-hot encoding represents each word as a vector of zeros with a single 1
# at the word's position:
#   cat -> [1, 0, 0, 0, 0]
#   sat -> [0, 1, 0, 0, 0]
#   mat -> [0, 0, 0, 0, 1]

vocab = ["cat", "sat", "on", "the", "mat"]
vocab_size = len(vocab)
word_to_id = {word: i for i, word in enumerate(vocab)}

def one_hot(word_id, vocab_size):
    """Create a one-hot vector for a given word ID."""
    vec = torch.zeros(vocab_size)
    vec[word_id] = 1.0
    return vec

print("Vocabulary:", vocab)
print("Word-to-ID mapping:", word_to_id)
print()

for word in vocab:
    vec = one_hot(word_to_id[word], vocab_size)
    print(f"  '{word}' (id={word_to_id[word]}) -> {vec.tolist()}")

print("""
Problems with one-hot encoding:
  1. HUGE vectors: real vocabularies have 50,000+ words.
     50,000-element vectors are very wasteful.
  2. NO similarity info: "cat" and "dog" are as different as "cat" and "quantum"
     (all dot products between different words = 0).
  3. Not learnable: these are fixed, hand-crafted representations.
""")

# =============================================================================
# FIRST PRINCIPLES: Why One-Hot Is Wasteful (The Math)
# =============================================================================
#
# One-hot encoding for vocabulary V:
#   - Each token is a vector of length V with exactly one 1 and (V-1) zeros.
#   - Space per token: O(V).
#   - For V = 50,000 and a sequence of 1024 tokens:
#       50,000 x 1024 = 51,200,000 numbers just to represent the input!
#   - Dot product between any two DIFFERENT one-hot vectors = 0.
#     This means the representation encodes NO similarity at all.
#     "king" and "queen" are as far apart as "king" and "banana".
#
# Learned embedding of dimension d (where d << V):
#   - Each token is a dense vector of length d (e.g., d = 768).
#   - Space per token: O(d).  For d=768, that is 65x smaller than V=50,000.
#   - Because these vectors are LEARNED, the model can place similar words
#     near each other in this d-dimensional space.
#
# Parameter count:
#   - The embedding table is a matrix of shape (V x d).
#   - For GPT-2: 50,257 x 768 = 38,597,376 learnable parameters.
#   - Positional encoding (sinusoidal): 0 learnable parameters (fixed formula).
#   - Positional encoding (learned): max_seq_len x d learnable parameters.
# =============================================================================


print("=" * 60)
print("PART 2: Learned Embeddings — dense, meaningful vectors")
print("=" * 60)

# --- What are embeddings? ---
# Instead of a huge sparse vector, give each word a SMALL DENSE vector
# (e.g. 8 numbers, or 128 numbers in real models).
#
# These numbers are LEARNED during training. The model adjusts them so that
# words with similar meanings end up with similar vectors!
#
# For example, after training:
#   king  ~= [0.9, 0.2, -0.3, ...]
#   queen ~= [0.8, 0.7, -0.3, ...]    (similar to king!)
#   table ~= [-0.1, 0.0, 0.9, ...]    (very different)
#
# nn.Embedding is basically a lookup table (a matrix):
#   - rows = one row per word in vocabulary
#   - columns = embedding dimension (how many numbers per word)

VOCAB_SIZE = 10      # small vocabulary for this demo
EMBED_DIM  = 4       # each word gets a 4-number vector (tiny for demo)

# Create the embedding layer
embedding_layer = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBED_DIM)

print(f"Embedding table shape: {embedding_layer.weight.shape}")
print(f"  -> {VOCAB_SIZE} words, each represented by {EMBED_DIM} numbers")
print(f"\nFull embedding table (randomly initialized):\n{embedding_layer.weight.data}")

# --- Looking up embeddings ---
# We give it a list of integer token IDs, it returns their embedding vectors.

token_ids = torch.tensor([0, 2, 4])   # look up words 0, 2, and 4
embeddings = embedding_layer(token_ids)
print(f"\nLooking up tokens {token_ids.tolist()}:")
print(embeddings)
print(f"Shape: {embeddings.shape}  (3 tokens, each with {EMBED_DIM} numbers)")

# --- These are learnable parameters ---
print(f"\nIs the embedding table learnable? {embedding_layer.weight.requires_grad}")
print("Yes! During training, these numbers get adjusted so similar words")
print("end up with similar vectors.")

# --- A sentence as embeddings ---
print("\n--- Embedding a whole sentence ---")
# Sentence: "cat sat on the mat"
sentence_ids = torch.tensor([word_to_id["cat"],
                              word_to_id["sat"],
                              word_to_id["on"],
                              word_to_id["the"],
                              word_to_id["mat"]])

# Use a new embedding for the 5-word vocab with embed_dim=6
small_embed = nn.Embedding(num_embeddings=5, embedding_dim=6)
sentence_embeddings = small_embed(sentence_ids)
print(f"Sentence token IDs: {sentence_ids.tolist()}")
print(f"Sentence embeddings shape: {sentence_embeddings.shape}")
print(f"  -> (5 tokens, 6 numbers each)")
print(sentence_embeddings)


print("\n" + "=" * 60)
print("PART 3: Positional Encoding — teaching the model about ORDER")
print("=" * 60)

# --- Why do we need positional encoding? ---
# Transformers read ALL tokens at the same time (in parallel).
# This is fast, but it means the model can't tell:
#   "The dog bit the man"  from  "The man bit the dog"
# ... because the same words appear in both!
#
# We fix this by adding a unique "position signal" to each token's embedding.
# After adding position info:
#   token at position 0 gets embedding + position_vector_0
#   token at position 1 gets embedding + position_vector_1
#   ... and so on.
#
# The position vectors are computed using sine and cosine waves at different
# frequencies. Each position gets a unique pattern of waves.

# =============================================================================
# FIRST PRINCIPLES: Sinusoidal Positional Encoding — Why Sin and Cos?
# =============================================================================
#
# Formula:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
#
# WHY sin/cos? Because PE(pos+k) can be written as a LINEAR function of PE(pos)!
#
# Proof sketch using sin/cos addition formulas:
#   sin(pos+k)/w = sin(pos/w)*cos(k/w) + cos(pos/w)*sin(k/w)
#   cos(pos+k)/w = cos(pos/w)*cos(k/w) - sin(pos/w)*sin(k/w)
#
# This means: [PE(pos+k, 2i)  ]   [ cos(k/w)  sin(k/w)] [PE(pos, 2i)  ]
#              [PE(pos+k, 2i+1)] = [-sin(k/w)  cos(k/w)] [PE(pos, 2i+1)]
#
# For any fixed offset k, the transformation is a ROTATION MATRIX applied
# to each (sin, cos) pair. This is a linear transformation of PE(pos)!
# The model can therefore learn relative positions using linear operations.
#
# WHY different frequencies?
#   - Dimension pair (2i, 2i+1) uses wavelength = 2*pi*10000^(2i/d).
#   - Low dimensions (small i) have SHORT wavelengths -> capture fine position.
#   - High dimensions (large i) have LONG wavelengths -> capture coarse position.
#   - The model can attend to different SCALES of relative distance.
#
# EXERCISE: Show that for any fixed offset k, PE(pos+k) is a linear
# transformation of PE(pos).
#   Hint: For each pair (2i, 2i+1), let w = 10000^(2i/d). Then:
#     PE(pos+k, 2i)   = sin((pos+k)/w) = sin(pos/w)cos(k/w) + cos(pos/w)sin(k/w)
#     PE(pos+k, 2i+1) = cos((pos+k)/w) = cos(pos/w)cos(k/w) - sin(pos/w)sin(k/w)
#   This is a 2x2 rotation matrix [cos(k/w), sin(k/w); -sin(k/w), cos(k/w)]
#   applied to [PE(pos, 2i), PE(pos, 2i+1)]. Linear!
# =============================================================================

def sinusoidal_positional_encoding(seq_len, embed_dim):
    """
    Create positional encoding matrix of shape (seq_len, embed_dim).

    For position pos and dimension i:
      PE[pos, 2i]   = sin(pos / 10000^(2i/embed_dim))
      PE[pos, 2i+1] = cos(pos / 10000^(2i/embed_dim))

    The magic: each dimension uses a different frequency wave,
    so every position gets a unique combination of values.
    """
    pe = torch.zeros(seq_len, embed_dim)

    # positions: [0, 1, 2, ..., seq_len-1], shape (seq_len, 1)
    positions = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)

    # frequency divisors, shape (embed_dim/2,)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float)
        * -(math.log(10000.0) / embed_dim)
    )

    # fill even indices with sin, odd indices with cos
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)

    return pe   # shape: (seq_len, embed_dim)


SEQ_LEN   = 5
EMBED_DIM = 6

pos_enc = sinusoidal_positional_encoding(SEQ_LEN, EMBED_DIM)
print(f"Positional encoding shape: {pos_enc.shape}")
print(f"  -> {SEQ_LEN} positions, {EMBED_DIM} dimensions each")
print("\nPositional encoding matrix (each row = unique position signal):")
for i, row in enumerate(pos_enc):
    print(f"  Position {i}: {[f'{v:.3f}' for v in row.tolist()]}")

print("\nNotice: every row is different -> each position has a unique fingerprint!")


print("\n" + "=" * 60)
print("PART 4: Full pipeline — tokens -> embeddings -> add positions")
print("=" * 60)

# --- Putting it all together ---
# This is exactly what happens at the very start of a transformer:
#
#   raw text  ->  tokenizer  ->  token IDs
#   token IDs ->  nn.Embedding  ->  token embeddings
#   positions ->  positional encoding  ->  position embeddings
#   token embeddings + position embeddings  ->  input to transformer

class TokenAndPositionEmbedding(nn.Module):
    """
    Combines token embeddings with positional encoding.
    Input:  token ID tensor of shape (batch_size, seq_len)
    Output: embedding tensor of shape (batch_size, seq_len, embed_dim)
    """

    def __init__(self, vocab_size, embed_dim, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Fixed positional encoding (not learned, but could be)
        pe = sinusoidal_positional_encoding(max_seq_len, embed_dim)
        # register_buffer: saves pe as part of the model but NOT as a learned
        # parameter (won't be updated by the optimizer)
        self.register_buffer("pos_encoding", pe)

    def forward(self, token_ids):
        # token_ids shape: (batch_size, seq_len)
        batch_size, seq_len = token_ids.shape

        # Get token embeddings: (batch_size, seq_len, embed_dim)
        tok_emb = self.token_embedding(token_ids)

        # Get positional encoding for seq_len positions: (seq_len, embed_dim)
        pos_emb = self.pos_encoding[:seq_len, :]    # slice to actual length

        # Add them together (broadcasting handles the batch dimension)
        # tok_emb:  (batch_size, seq_len, embed_dim)
        # pos_emb:  (          seq_len, embed_dim)  <- broadcast over batch
        combined = tok_emb + pos_emb

        return combined


# --- Demo ---
VOCAB_SIZE   = 20
EMBED_DIM    = 8
BATCH_SIZE   = 2
SEQ_LEN      = 5

embed_layer = TokenAndPositionEmbedding(VOCAB_SIZE, EMBED_DIM)

# Simulate a batch of 2 sentences, each 5 tokens long
token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
print("Input token IDs (2 sentences, 5 tokens each):")
print(token_ids)
print(f"Shape: {token_ids.shape}")

output = embed_layer(token_ids)
print(f"\nOutput embeddings shape: {output.shape}")
print(f"  -> (batch=2, seq_len=5, embed_dim=8)")
print("\nFirst sentence, first token embedding (token embed + position signal):")
print(output[0, 0])

print("\nParameter count:")
total = sum(p.numel() for p in embed_layer.parameters())
print(f"  Token embedding table: {VOCAB_SIZE} x {EMBED_DIM} = {VOCAB_SIZE * EMBED_DIM} parameters")
print(f"  Positional encoding:   NOT learnable (fixed sine/cosine waves)")
print(f"  Total learnable params: {total}")

print("\nSUMMARY of the pipeline:")
print("  Raw text  ->  tokenizer  ->  token IDs")
print("  token IDs ->  nn.Embedding  ->  token embeddings")
print("  positions ->  sinusoidal_PE ->  position embeddings")
print("  token embeddings + position embeddings  =  input to transformer blocks")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- One-hot encoding tests ---
for word in vocab:
    vec = one_hot(word_to_id[word], vocab_size)
    assert vec.shape == torch.Size([vocab_size]), f"one_hot shape wrong for {word}"
    assert vec.sum().item() == 1.0, f"one_hot should sum to 1 for {word}"
    assert vec[word_to_id[word]].item() == 1.0, f"one_hot should have 1 at correct index for {word}"

# --- Learned embedding tests (embedding_layer was created with vocab=10, dim=4) ---
assert embedding_layer.weight.shape == torch.Size([10, 4]), \
    "embedding table shape should be (10, 4)"
assert embedding_layer.weight.requires_grad, "embedding weights should be learnable"

# Test lookup with valid IDs for the original embedding_layer (vocab=10)
_test_ids = torch.tensor([0, 2, 4])
_emb_test = embedding_layer(_test_ids)
assert _emb_test.shape == torch.Size([3, 4]), \
    "lookup of 3 tokens should give shape (3, 4)"

# --- Sentence embeddings test (created with embed_dim=6, 5 tokens) ---
assert sentence_embeddings.shape == torch.Size([5, 6]), \
    "sentence embeddings should be (5 tokens, 6 dims)"

# --- Positional encoding tests (current EMBED_DIM=8, SEQ_LEN=5) ---
pe_test = sinusoidal_positional_encoding(SEQ_LEN, EMBED_DIM)
assert pe_test.shape == torch.Size([SEQ_LEN, EMBED_DIM]), \
    f"positional encoding should be ({SEQ_LEN}, {EMBED_DIM})"

# Each position should have a unique encoding (no two rows identical)
for i in range(SEQ_LEN):
    for j in range(i + 1, SEQ_LEN):
        assert not torch.allclose(pe_test[i], pe_test[j]), \
            f"positions {i} and {j} should have different encodings"

# --- TokenAndPositionEmbedding tests (VOCAB_SIZE=20, EMBED_DIM=8, BATCH=2, SEQ=5) ---
assert output.shape == torch.Size([BATCH_SIZE, SEQ_LEN, EMBED_DIM]), \
    f"combined embedding output should be ({BATCH_SIZE}, {SEQ_LEN}, {EMBED_DIM})"

# Parameter count: token embedding = VOCAB_SIZE * EMBED_DIM = 20 * 8 = 160
expected_params = VOCAB_SIZE * EMBED_DIM
assert total == expected_params, \
    f"learnable params should be {expected_params} (token embedding only), got {total}"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("\n" + "=" * 60)
print("EXERCISES (try these yourself!)")
print("=" * 60)
print("""
1. One-hot vs embedding size:
   - GPT-2 has a vocabulary of 50,257 words and uses embed_dim=768.
   - How many numbers does a one-hot vector need?      (Answer: 50,257)
   - How many numbers does an embedding vector need?   (Answer: 768)
   - How many times smaller is the embedding?

2. Embedding lookup practice:
   - Create nn.Embedding(num_embeddings=100, embedding_dim=16).
   - Look up embeddings for token IDs [5, 10, 15, 20, 25].
   - Print the shape of the result.
   - Print the embedding for token ID 10.

3. Positional encoding visualization:
   - Call sinusoidal_positional_encoding(50, 8) to get a 50x8 matrix.
   - Print the values in column 0 (all positions, dimension 0).
   - Notice they follow a sine wave pattern!
   - Print column 1. What pattern do you see?
   (Hint: column 0 uses sin, column 1 uses cos — they should look like waves.)

4. Extend TokenAndPositionEmbedding:
   - Add a dropout layer (nn.Dropout(p=0.1)) to the __init__.
   - Apply dropout to the combined embedding in the forward method.
   - Why might dropout help during training?
   (Hint: dropout randomly zeroes some values — it acts like a regularizer.)
""")
