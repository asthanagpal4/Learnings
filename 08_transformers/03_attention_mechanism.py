# HOW TO RUN:
#   uv run python 08_transformers/03_attention_mechanism.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# LESSON 3: The Attention Mechanism
# =============================================================================
#
# THE KEY INSIGHT
# ---------------
# When you read "The cat sat on the mat because IT was tired", what does "IT"
# refer to? You automatically focus on "cat" — not "mat", not "sat".
#
# Attention lets the model LEARN what to focus on for each token.
# Instead of treating every word equally, the model learns to assign higher
# "attention weight" to the most relevant words.
#
# This is the core idea behind ALL modern language models (GPT, BERT, etc.).
#
# Run this file:
#   uv run python 03_attention_mechanism.py
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("PART 1: Query, Key, Value — the library analogy")
print("=" * 60)

# --- Understanding Q, K, V ---
#
# Imagine a library:
#   - You walk in with a QUERY: "I want a book about deep learning."
#   - Every book has a KEY on its spine: "machine learning textbook", etc.
#   - Each book has actual VALUE (its content).
#
# The librarian:
#   1. Compares your query to every key.
#   2. Finds the best matches (assigns scores).
#   3. Hands you a weighted mix of the matching books' values.
#
# In a transformer:
#   - QUERY (Q): "what information am I looking for?"  (from the current token)
#   - KEY   (K): "what information do I contain?"      (from every token)
#   - VALUE (V): "here is my actual content"           (from every token)
#
# The attention weights say: "to understand token i, pay X% attention to
# token j, Y% attention to token k, ..."
#
# ALL of Q, K, V are computed from the input embeddings using learned linear
# projections (weight matrices). The model learns WHICH aspects to use as
# queries vs keys vs values.

print("Library analogy:")
print("  Query  = 'I want books about deep learning'")
print("  Keys   = labels on book spines")
print("  Values = actual book contents")
print("  Attention weights = how well each book matches your query")
print()
print("In a transformer:")
print("  Q, K, V are all derived from the same input (self-attention)")
print("  The model LEARNS the query/key/value projections during training")



# =============================================================================
# FIRST PRINCIPLES: Deriving Attention from Scratch
# =============================================================================
#
# Start with the problem: INFORMATION RETRIEVAL.
#   Given a query "what am I looking for?", find relevant data and return it.
#
# Step 1 — HARD ATTENTION (argmax):
#   For each query, find the SINGLE most similar key (using dot product).
#   Return only that key's value. Problem: argmax is NOT differentiable!
#   We cannot train this with gradient descent.
#
# Step 2 — SOFT ATTENTION (softmax):
#   Instead of picking one key, compute a WEIGHTED AVERAGE of all values.
#   Weight = how similar each key is to the query (via dot product).
#   This is differentiable! We can train it end-to-end.
#
# Step 3 — THE SCALING PROBLEM:
#   If query q and key k each have components drawn from N(0,1),
#   then the dot product q . k = sum(q_i * k_i) for i=1..d_k.
#   Each q_i * k_i has mean 0 and variance 1 (product of independent N(0,1)).
#   The sum of d_k such terms has mean 0 and variance d_k.
#   Standard deviation = sqrt(d_k).
#
#   For d_k = 64: dot products have std ~8. Softmax of values spread over
#   [-16, +16] produces vectors like [0, 0, 0, ..., 0.999, ..., 0, 0].
#   The gradients of softmax are TINY when outputs are near 0 or 1
#   (vanishing gradients). Training stalls.
#
#   FIX: divide by sqrt(d_k) to normalize variance back to 1.
#   Now softmax inputs have a reasonable spread, and gradients flow well.
#
# COMPLEXITY:
#   - Computing Q @ K^T: O(n^2 * d_k) where n = sequence length.
#     Every token's query is compared to every token's key -> n^2 pairs.
#   - Softmax: O(n^2) per row, n rows -> O(n^2).
#   - Weights @ V: O(n^2 * d_v).
#   - Total: O(n^2 * d) where d = max(d_k, d_v).
#   - The n^2 term is why transformers struggle with very long sequences!
#
# MULTI-HEAD ATTENTION:
#   Split d into h heads, each of size d/h.
#   Each head runs independent attention -> can learn different relationships.
#   Total complexity: h * O(n^2 * d/h) = O(n^2 * d). Same total!
#   But now the model can attend to MULTIPLE things simultaneously.
#
# EXERCISE: Calculate attention memory for seq_len=2048 with 12 heads.
#   Attention weights shape: (batch, num_heads, seq_len, seq_len)
#   Per sample: 12 * 2048 * 2048 = 50,331,648 float32 values
#   Memory: 50,331,648 * 4 bytes = ~192 MB per sample!
#   With batch_size=8: ~1.5 GB just for attention weights.
#   This is why long-context models need so much GPU memory.
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: Scaled Dot-Product Attention — step by step")
print("=" * 60)

# --- Formula ---
# Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
#
# Step by step:
#   1. Compute scores:   scores = Q @ K^T          (how similar is each Q to each K?)
#   2. Scale:            scores = scores / sqrt(d_k) (prevents scores from getting huge)
#   3. Softmax:          weights = softmax(scores)   (convert to probabilities 0..1, sum=1)
#   4. Weighted sum:     output = weights @ V        (mix values by attention weights)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query  tensor, shape (..., seq_len_q, d_k)
        K: Key    tensor, shape (..., seq_len_k, d_k)
        V: Value  tensor, shape (..., seq_len_k, d_v)
        mask: optional boolean mask, shape (..., seq_len_q, seq_len_k)
              True = masked (ignore this position)

    Returns:
        output:  shape (..., seq_len_q, d_v)
        weights: attention weights, shape (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]   # dimension of keys

    # Step 1: Compute raw attention scores
    # Q @ K^T: for each query, compute a dot product with every key
    scores = Q @ K.transpose(-2, -1)   # shape: (..., seq_len_q, seq_len_k)

    # Step 2: Scale by sqrt(d_k)
    # Why? Without scaling, dot products can be very large for high d_k,
    # pushing softmax into regions with tiny gradients (vanishing gradients).
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply causal mask (if provided)
    # Masked positions get score = -infinity, so after softmax they become ~0.
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Step 4: Softmax over the key dimension -> attention weights
    # Each row sums to 1: "distribute 100% attention across all positions"
    weights = F.softmax(scores, dim=-1)

    # Step 5: Weighted sum of values
    output = weights @ V   # shape: (..., seq_len_q, d_v)

    return output, weights


# --- Manual demo with tiny numbers ---
print("Manual attention demo:")
print()

# Let's say we have 3 tokens, each represented by 4-dimensional Q/K/V vectors
seq_len = 3
d_k     = 4

torch.manual_seed(0)
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

print(f"Q shape: {Q.shape}  (3 tokens, 4-dim query vectors)")
print(f"K shape: {K.shape}  (3 tokens, 4-dim key vectors)")
print(f"V shape: {V.shape}  (3 tokens, 4-dim value vectors)")

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"\nAttention weights (rows = queries, cols = keys):")
print(weights.detach().round(decimals=3))
print("Each ROW sums to 1.0:")
print(weights.sum(dim=-1).detach().round(decimals=4))

print(f"\nOutput shape: {output.shape}")
print("Each output token is a WEIGHTED MIX of all value vectors.")
print("Token 0's output = 0.X * V[0] + 0.Y * V[1] + 0.Z * V[2]")


print("\n" + "=" * 60)
print("PART 3: Causal Masking — no cheating!")
print("=" * 60)

# --- Why masking? ---
# In a LANGUAGE MODEL, the job is to predict the NEXT token given previous ones.
# If token at position 5 could attend to position 7, it would be "cheating"
# (looking at future information that shouldn't be available yet).
#
# We fix this with a CAUSAL MASK: a lower-triangular matrix.
# Token i can only attend to positions 0, 1, ..., i  (not i+1, i+2, ...)
#
#   Position:  0  1  2  3
#   Token 0: [ 1  0  0  0 ]  <- can only see itself
#   Token 1: [ 1  1  0  0 ]  <- can see positions 0 and 1
#   Token 2: [ 1  1  1  0 ]  <- can see positions 0, 1, 2
#   Token 3: [ 1  1  1  1 ]  <- can see everything

seq_len = 5

# torch.triu with diagonal=1 gives the UPPER triangle (the forbidden future)
# We set those positions to True (=masked)
causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

print(f"Causal mask (seq_len={seq_len}):")
print(f"True = masked (future position, set to -infinity before softmax)")
print(causal_mask.int())   # print as 0/1 for readability
print()
print("What this means:")
print("  Token 0 can attend to: position 0 only")
print("  Token 1 can attend to: positions 0, 1")
print("  Token 2 can attend to: positions 0, 1, 2")
print("  Token 4 can attend to: all 5 positions")

# Demo with mask
Q_demo = torch.randn(seq_len, d_k)
K_demo = torch.randn(seq_len, d_k)
V_demo = torch.randn(seq_len, d_k)

output_masked, weights_masked = scaled_dot_product_attention(
    Q_demo, K_demo, V_demo, mask=causal_mask
)
print(f"\nAttention weights WITH causal mask:")
print(weights_masked.detach().round(decimals=3))
print("(upper triangle = 0, confirming no future peeking!)")


print("\n" + "=" * 60)
print("PART 4: Multi-Head Attention — multiple perspectives")
print("=" * 60)

# --- Why multiple heads? ---
# A single attention head can only learn ONE way of relating tokens.
# Multi-head attention uses H independent attention "heads" in PARALLEL.
# Each head projects Q, K, V differently and attends to different aspects:
#   Head 1 might learn syntactic relationships (subject -> verb)
#   Head 2 might learn semantic relationships (synonyms)
#   Head 3 might learn positional relationships (nearby words)
#   ... etc.
#
# Steps:
#   1. Project Q, K, V into H smaller subspaces (each of size d_k/H)
#   2. Run scaled dot-product attention independently in each head
#   3. Concatenate all head outputs back together
#   4. Apply one final linear projection
#
#   d_model = full embedding dimension (e.g. 512)
#   d_k     = d_model / num_heads        (e.g. 512 / 8 = 64 per head)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module.

    Args:
        embed_dim:  total embedding dimension (d_model)
        num_heads:  number of attention heads
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads  # dimension per head

        # Three projection matrices: one for Q, one for K, one for V
        # These are the learnable parameters that transform the input into Q/K/V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Final output projection: combines all heads back into embed_dim
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, head_dim) and rearrange.
        Input:  (batch, seq_len, embed_dim)
        Output: (batch, num_heads, seq_len, head_dim)
        """
        batch, seq_len, embed_dim = x.shape
        # Reshape: (batch, seq_len, num_heads, head_dim)
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        # Transpose to: (batch, num_heads, seq_len, head_dim)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        """
        Args:
            x:    input tensor, shape (batch, seq_len, embed_dim)
            mask: optional causal mask, shape (seq_len, seq_len)

        Returns:
            output: shape (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = x.shape

        # Step 1: Project input into Q, K, V
        Q = self.W_q(x)   # (batch, seq_len, embed_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Split into multiple heads
        Q = self.split_heads(Q)   # (batch, num_heads, seq_len, head_dim)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Step 3: Apply scaled dot-product attention on each head
        # If mask provided, expand dims to broadcast over batch and heads
        if mask is not None:
            # mask shape: (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(0)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output shape: (batch, num_heads, seq_len, head_dim)

        # Step 4: Reassemble all heads
        # Transpose back: (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Merge heads: (batch, seq_len, embed_dim)
        attn_output = attn_output.view(batch, seq_len, self.embed_dim)

        # Step 5: Final output projection
        output = self.W_o(attn_output)   # (batch, seq_len, embed_dim)

        return output, attn_weights


# --- Demo ---
BATCH_SIZE = 2
SEQ_LEN    = 6
EMBED_DIM  = 32
NUM_HEADS  = 4

mha = MultiHeadAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
print(f"MultiHeadAttention config:")
print(f"  embed_dim = {EMBED_DIM}")
print(f"  num_heads = {NUM_HEADS}")
print(f"  head_dim  = {EMBED_DIM // NUM_HEADS}  (per head)")

# Count parameters
total_params = sum(p.numel() for p in mha.parameters())
print(f"  Total parameters: {total_params}")
print()

# Random input: batch of 2 sequences, each 6 tokens, each 32-dimensional
x_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
print(f"Input shape:  {x_input.shape}  (batch=2, seq=6, embed=32)")

# Create causal mask
causal_mask_mha = torch.triu(
    torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1
)

# Forward pass
output, weights = mha(x_input, mask=causal_mask_mha)
print(f"Output shape: {output.shape}  (same as input — attention preserves shape!)")
print(f"Attention weights shape: {weights.shape}  (batch, heads, seq, seq)")

print("\nAttention weights for batch=0, head=0:")
print(weights[0, 0].detach().round(decimals=3))
print("(upper triangle = 0 because of causal mask)")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Attention mechanism summary:

  1. SINGLE-HEAD ATTENTION:
     output = softmax( Q @ K^T / sqrt(d_k) ) @ V
     - Q, K, V are linear projections of the input
     - Each output token is a weighted sum of all value vectors
     - Weights come from how similar each query is to each key

  2. CAUSAL MASKING:
     - Set upper-triangle scores to -infinity before softmax
     - Prevents token i from attending to positions > i
     - Essential for language model training (no future peeking)

  3. MULTI-HEAD ATTENTION:
     - Run H attention heads in parallel, each in a d_k/H subspace
     - Each head can specialize in different kinds of relationships
     - Concatenate and project outputs back to embed_dim
     - Output shape = input shape (attention is "shape-preserving")
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- Scaled dot-product attention tests (re-run with known inputs) ---
_Q_test = torch.randn(3, 4)
_K_test = torch.randn(3, 4)
_V_test = torch.randn(3, 4)
_out_test, _w_test = scaled_dot_product_attention(_Q_test, _K_test, _V_test)
assert _out_test.shape == torch.Size([3, 4]), \
    f"attention output should be (3, 4), got {_out_test.shape}"
assert _w_test.shape == torch.Size([3, 3]), \
    f"attention weights should be (3, 3), got {_w_test.shape}"

# Each row of weights must sum to 1.0 (they are probabilities)
_row_sums = _w_test.sum(dim=-1)
assert torch.allclose(_row_sums, torch.ones(3), atol=1e-5), \
    "each row of attention weights must sum to 1.0"

# --- Causal mask tests ---
# Masked attention: upper triangle of weights should be ~0
for i in range(seq_len):        # seq_len == 5 here (the mask demo variable)
    for j in range(i + 1, seq_len):
        assert weights_masked[i, j].item() < 1e-6, \
            f"causal mask: position ({i},{j}) should be ~0 (future token)"

# Masked weights rows should still sum to 1
row_sums_masked = weights_masked.sum(dim=-1)
assert torch.allclose(row_sums_masked, torch.ones(seq_len), atol=1e-5), \
    "masked attention weights rows must still sum to 1.0"

# --- MultiHeadAttention tests ---
# output, weights are from MHA call: shape (batch, seq, embed) and (batch, heads, seq, seq)
assert output.shape == torch.Size([BATCH_SIZE, SEQ_LEN, EMBED_DIM]), \
    f"MHA output should be ({BATCH_SIZE}, {SEQ_LEN}, {EMBED_DIM}), got {output.shape}"
assert weights.shape == torch.Size([BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN]), \
    f"MHA weights shape should be (batch, heads, seq, seq), got {weights.shape}"

# MHA with a fresh input
_x_test = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
_causal = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
_out, _w = mha(_x_test, mask=_causal)
assert _out.shape == torch.Size([BATCH_SIZE, SEQ_LEN, EMBED_DIM]), \
    f"MHA output should be (batch={BATCH_SIZE}, seq={SEQ_LEN}, embed={EMBED_DIM})"

# MHA upper-triangle weights should be ~0 (causal mask enforced)
for i in range(SEQ_LEN):
    for j in range(i + 1, SEQ_LEN):
        assert _w[0, 0, i, j].item() < 1e-6, \
            f"MHA causal mask: weight at (0,0,{i},{j}) should be ~0"

# Parameter count: 4 matrices each of shape (EMBED_DIM, EMBED_DIM), no bias
expected_mha_params = 4 * EMBED_DIM * EMBED_DIM
assert total_params == expected_mha_params, \
    f"MHA params should be {expected_mha_params} (4 x {EMBED_DIM}^2), got {total_params}"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("=" * 60)
print("EXERCISES (try these yourself!)")
print("=" * 60)
print("""
1. Understand attention weights:
   - Create Q = K = V = torch.eye(4)  (4x4 identity matrix).
   - Call scaled_dot_product_attention(Q, K, V).
   - What do the attention weights look like? Why?
   (Hint: when Q=K=identity, each query is identical to exactly one key.)

2. Effect of scaling:
   - Create random Q and K with d_k=64.
   - Compute Q @ K.T both WITH and WITHOUT the 1/sqrt(d_k) scaling.
   - Call .std() on both to compare how spread-out the values are.
   - What happens to softmax when values are very large?

3. Causal mask check:
   - Create a MultiHeadAttention(embed_dim=16, num_heads=2).
   - Pass in a (1, 8, 16) input WITH the causal mask.
   - Print the attention weights for head 0.
   - Confirm that the upper-triangular part is all zeros.

4. Bidirectional attention (no mask):
   - Repeat exercise 3 WITHOUT the mask.
   - Notice that now every token can attend to every other token.
   - This is used in BERT (encoder-only models), not GPT (decoder-only).

5. Attention to a specific word (thought experiment, no code needed):
   - In "The cat sat on the mat because IT was tired",
     which token should "IT" most attend to, and why?
   - How would you verify this with a trained model?
""")
