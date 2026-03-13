# HOW TO RUN:
#   uv run python 08_transformers/04_transformer_block.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# LESSON 4: The Transformer Block
# =============================================================================
#
# WHAT IS A TRANSFORMER BLOCK?
# ----------------------------
# One transformer block stacks four components together:
#
#   Input
#     |
#     +-------> Multi-Head Self-Attention
#     |               |
#     +-- Residual --> Add --> LayerNorm   (first "Add & Norm")
#                       |
#                       +-------> Feed-Forward Network (FFN)
#                       |               |
#                       +-- Residual --> Add --> LayerNorm   (second "Add & Norm")
#                                         |
#                                       Output
#
# Key ideas:
#   - RESIDUAL CONNECTIONS ("skip connections"): Add the input back to the
#     output. This lets gradients flow directly to early layers, enabling
#     very deep networks. Without residuals, gradients vanish.
#
#   - LAYER NORMALIZATION: Normalizes values to have mean~0 and std~1.
#     Keeps activations stable throughout training.
#
#   - FEED-FORWARD NETWORK (FFN): Two linear layers with a nonlinearity.
#     Processes each token independently. Often described as where the
#     "knowledge" gets stored.
#
# A full transformer model stacks N of these blocks.
#
# Run this file:
#   uv run python 04_transformer_block.py
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# Re-implement the components we need (self-contained, no imports from other
# lesson files)
# =============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention (from Lesson 3)."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention (from Lesson 3)."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def _split_heads(self, x):
        b, s, e = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, mask=None):
        b, s, _ = x.shape
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(b, s, self.embed_dim)
        return self.W_o(out)


def sinusoidal_positional_encoding(seq_len, embed_dim):
    """Sinusoidal positional encoding (from Lesson 2)."""
    pe = torch.zeros(seq_len, embed_dim)
    positions = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float)
        * -(math.log(10000.0) / embed_dim)
    )
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe


# =============================================================================
# NEW COMPONENTS
# =============================================================================

print("=" * 60)
print("PART 1: Layer Normalization — keeping values stable")
print("=" * 60)

# --- What is Layer Normalization? ---
# During forward passes through many layers, the values can grow or shrink
# dramatically (exploding or vanishing activations). This makes training
# very unstable.
#
# LayerNorm fixes this by normalizing each token's vector to have:
#   - mean  = 0
#   - standard deviation = 1
# ... then scaling and shifting with learned parameters (gamma, beta).
#
# Formula:
#   y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
#
# Important: this is applied PER TOKEN (across the feature dimension),
# NOT across the batch or sequence dimension.

x_demo = torch.tensor([[10.0, 200.0, -50.0, 5.0],
                        [ 1.0,   2.0,   3.0, 4.0]])

print("Input values:")
print(x_demo)
print(f"Row 0 stats: mean={x_demo[0].mean():.2f}, std={x_demo[0].std():.2f}")
print(f"Row 1 stats: mean={x_demo[1].mean():.2f}, std={x_demo[1].std():.2f}")

layer_norm = nn.LayerNorm(normalized_shape=4)   # normalize the last dim (size 4)
x_normed   = layer_norm(x_demo)

print("\nAfter LayerNorm:")
print(x_normed.detach().round(decimals=3))
print(f"Row 0 stats: mean={x_normed[0].mean().item():.4f}, std={x_normed[0].std().item():.4f}")
print(f"Row 1 stats: mean={x_normed[1].mean().item():.4f}, std={x_normed[1].std().item():.4f}")
print("Both rows now have mean~0 and std~1, regardless of their original values!")


print("\n" + "=" * 60)
print("PART 2: Feed-Forward Network (FFN) — per-token processing")
print("=" * 60)

# --- What is the FFN? ---
# The FFN is two linear layers with a nonlinear activation in between:
#
#   FFN(x) = Linear2( GELU( Linear1(x) ) )
#
# Properties:
#   - Applied to EACH TOKEN INDEPENDENTLY (no cross-token information here)
#   - The middle layer is usually 4x wider than embed_dim
#     e.g. embed_dim=128, middle=512
#   - GELU (Gaussian Error Linear Unit) is the nonlinearity used in GPT models
#     It's similar to ReLU but smoother.
#
# Purpose: while attention GATHERS information from other tokens, the FFN
# PROCESSES and TRANSFORMS that information for each token individually.
# Researchers have found this is where the model stores factual knowledge.

class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network:
        Linear(embed_dim -> 4*embed_dim) -> GELU -> Linear(4*embed_dim -> embed_dim)
    """

    def __init__(self, embed_dim, expansion_factor=4):
        super().__init__()
        hidden_dim = embed_dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),                          # smooth nonlinearity (used in GPT)
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        # The FFN is applied identically to EVERY TOKEN (no interaction)
        return self.net(x)


# Demo
EMBED_DIM = 32
ffn = FeedForwardNetwork(embed_dim=EMBED_DIM)
x_ffn = torch.randn(2, 5, EMBED_DIM)   # batch=2, seq=5, embed=32
out_ffn = ffn(x_ffn)
print(f"FFN input shape:  {x_ffn.shape}")
print(f"FFN output shape: {out_ffn.shape}  (same — it's shape-preserving)")
total_ffn = sum(p.numel() for p in ffn.parameters())
print(f"FFN parameter count: {total_ffn}")
print(f"  (two linear layers: {EMBED_DIM}->{EMBED_DIM*4} and {EMBED_DIM*4}->{EMBED_DIM})")

# Show GELU vs ReLU
print("\nGELU vs ReLU at various x values:")
x_vals = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
print(f"  x:    {x_vals.tolist()}")
print(f"  ReLU: {F.relu(x_vals).tolist()}")
print(f"  GELU: {[round(v, 3) for v in F.gelu(x_vals).tolist()]}")
print("GELU is smooth and allows small negative outputs (unlike ReLU)")


print("\n" + "=" * 60)
print("PART 3: Residual Connections — the secret to deep networks")
print("=" * 60)

# =============================================================================
# FIRST PRINCIPLES: Why Residual Connections Work (Gradient Math)
# =============================================================================
#
# Without residual connections:
#   output = F(x)
#   Gradient: d(output)/dx = dF/dx
#   Through L layers: gradient = dF_L/dx * dF_{L-1}/dx * ... * dF_1/dx
#   If each dF_i/dx < 1 (very common), the product VANISHES exponentially.
#   If each dF_i/dx > 1, the product EXPLODES.
#
# WITH residual connections:
#   output = x + F(x)
#   Gradient: d(output)/dx = 1 + dF(x)/dx
#   The "1" is the KEY INSIGHT! Even if dF(x)/dx -> 0, the gradient
#   is still at least 1. Gradients always have a highway back.
#
#   Through L layers with residuals:
#   gradient includes terms like: 1 * 1 * ... * 1 = 1 (the direct path)
#   PLUS many cross-terms involving the F_i gradients.
#   The gradient NEVER vanishes to zero because of the all-ones path.
#
# This is what enabled training 100+ layer networks (ResNet, 2015) and
# later transformers with 96+ layers (GPT-3).
# =============================================================================

# =============================================================================
# FIRST PRINCIPLES: LayerNorm Derivation
# =============================================================================
#
# LayerNorm formula:
#   mu = mean of activations across features = (1/d) * sum(x_i)
#   sigma = std of activations = sqrt((1/d) * sum((x_i - mu)^2) + eps)
#   LayerNorm(x) = gamma * (x - mu) / sigma + beta
#
# WHERE:
#   gamma (scale) and beta (shift) are LEARNABLE parameters, each of size d.
#   eps is a tiny constant (e.g., 1e-5) to prevent division by zero.
#
# WHY normalize?
#   Without normalization, activations can drift to very large or very small
#   values as they pass through many layers. This makes training unstable
#   because gradients become either huge or tiny.
#   Normalizing keeps activations in a stable range -> faster, more stable training.
#
# LayerNorm vs BatchNorm:
#   - BatchNorm: normalizes across the BATCH dimension (for each feature,
#     compute mean and std across all samples in the batch).
#     Problem: fails when batch_size=1 (no meaningful statistics).
#     Problem: requires tracking running statistics for inference.
#   - LayerNorm: normalizes across the FEATURE dimension (for each sample,
#     compute mean and std across all features).
#     Works with ANY batch size, even batch=1.
#     No running statistics needed. This is why transformers use LayerNorm.
# =============================================================================

# =============================================================================
# FIRST PRINCIPLES: Transformer Block Parameter Count
# =============================================================================
#
# For a transformer block with d_model, n_heads, d_ff:
#
# 1. Self-Attention:
#    - W_q: d_model x d_model  (projects input to queries)
#    - W_k: d_model x d_model  (projects input to keys)
#    - W_v: d_model x d_model  (projects input to values)
#    - W_o: d_model x d_model  (projects concatenated heads to output)
#    Total: 4 * d_model^2  (no bias in our implementation)
#
# 2. Feed-Forward Network (FFN):
#    - Linear1: d_model x d_ff + d_ff (weight + bias)
#    - Linear2: d_ff x d_model + d_model (weight + bias)
#    Total: 2 * d_model * d_ff + d_model + d_ff
#    Approximately: 2 * d_model * d_ff
#
# 3. LayerNorm (x2):
#    - Each LayerNorm has gamma (d_model) and beta (d_model)
#    Total: 2 * 2 * d_model = 4 * d_model
#
# GRAND TOTAL per block:
#    4*d^2 + 2*d*d_ff + 4*d  (ignoring biases in attention)
#    Typically d_ff = 4*d, so: 4*d^2 + 8*d^2 + 4*d = 12*d^2 + 4*d ~ 12*d^2
#
# EXERCISE: Calculate parameter count for d_model=768, n_heads=12, d_ff=3072:
#    Attention:  4 * 768^2          = 2,359,296
#    FFN:        2 * 768 * 3072     = 4,718,592 (plus biases: +768+3072 = 3840)
#    LayerNorm:  4 * 768            = 3,072
#    Total per block:               ~ 7,080,960
#    GPT-2 small has 12 such blocks: ~ 85 million params (just the blocks)
# =============================================================================

# --- Why residual connections? ---
# If you stack many layers, gradients must flow backward through ALL of them.
# In deep networks, gradients can shrink to nearly zero (vanishing gradient),
# making early layers almost untrainable.
#
# The residual connection is beautifully simple:
#   output = LayerNorm(x + SubLayer(x))
#
# This means there's always a DIRECT PATH for gradients to flow through:
# even if the sublayer's gradients vanish, the gradient of x flows back
# unchanged through the "+x" path.
#
# This trick enabled training networks with 100+ layers (and later 1000s)!

print("Residual connection pattern:")
print()
print("  x  ───────────────────────────────────┐")
print("  |                                     |")
print("  └──> SubLayer(x) ──> + ──> LayerNorm ─┘")
print("                       ^")
print("                      (x is added back here)")
print()
print("The '+x' at the end is the 'residual' or 'skip connection'.")
print("It gives gradients a highway back to early layers.")


print("\n" + "=" * 60)
print("PART 4: The Transformer Block — putting it all together")
print("=" * 60)

class TransformerBlock(nn.Module):
    """
    A single transformer block:

        x -> MultiHeadAttention -> + -> LayerNorm -> FFN -> + -> LayerNorm -> output
             (with residual: x +)                   (with residual)

    This is the "Pre-LN" variant: LayerNorm is applied BEFORE each sublayer.
    (The original paper uses Post-LN, but Pre-LN trains more stably.)
    """

    def __init__(self, embed_dim, num_heads, ffn_expansion=4, dropout=0.1):
        super().__init__()

        # Attention sub-layer
        self.attention  = MultiHeadAttention(embed_dim, num_heads)

        # Feed-forward sub-layer
        self.ffn        = FeedForwardNetwork(embed_dim, ffn_expansion)

        # Layer norms (one before attention, one before FFN — "Pre-LN" style)
        self.norm1      = nn.LayerNorm(embed_dim)
        self.norm2      = nn.LayerNorm(embed_dim)

        # Dropout for regularization (randomly zeroes some values during training)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x shape: (batch, seq_len, embed_dim)
        """
        # --- Self-attention sub-layer with residual ---
        # Pre-LN: normalize BEFORE attention
        normed = self.norm1(x)
        attn_out = self.attention(normed, mask=mask)
        x = x + self.dropout(attn_out)   # residual: add input back in

        # --- Feed-forward sub-layer with residual ---
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)   # residual: add input back in

        return x


# Demo a single block
EMBED_DIM  = 32
NUM_HEADS  = 4
SEQ_LEN    = 6
BATCH_SIZE = 2

block = TransformerBlock(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
x_in  = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

causal_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)

block.eval()   # disable dropout for the demo
with torch.no_grad():
    x_out = block(x_in, mask=causal_mask)

print(f"TransformerBlock demo:")
print(f"  Input  shape: {x_in.shape}")
print(f"  Output shape: {x_out.shape}  (identical — blocks are shape-preserving)")
block_params = sum(p.numel() for p in block.parameters())
print(f"  Parameters:   {block_params}")


print("\n" + "=" * 60)
print("PART 5: Full Model — stacking N transformer blocks")
print("=" * 60)

class MiniTransformerLM(nn.Module):
    """
    A complete (small) transformer language model:

      token IDs -> embedding + positional encoding
                -> N transformer blocks
                -> LayerNorm
                -> linear projection to vocab size
                -> logits (raw scores for each vocabulary token)

    To get probabilities, apply softmax to the logits.
    To train, use cross-entropy loss between logits and target token IDs.
    """

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 max_seq_len=128, ffn_expansion=4, dropout=0.1):
        super().__init__()
        self.embed_dim   = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding lookup table
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding (learnable version this time — simpler to implement)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_expansion, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm (after all blocks)
        self.final_norm = nn.LayerNorm(embed_dim)

        # Output projection: embed_dim -> vocab_size
        # These are the "logits" — one score per vocabulary token
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and lm_head
        # This is a common trick that reduces parameters and improves quality.
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights (small random values, important for stability)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        """
        Args:
            token_ids: shape (batch, seq_len)  — integer token IDs

        Returns:
            logits: shape (batch, seq_len, vocab_size)
                    logits[b, t, :] = scores for each possible next token at position t
        """
        batch, seq_len = token_ids.shape
        assert seq_len <= self.max_seq_len, \
            f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        # Token embeddings: (batch, seq_len, embed_dim)
        tok_emb = self.token_embedding(token_ids)

        # Positional embeddings: position IDs [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.pos_embedding(positions)   # (seq_len, embed_dim)

        # Combine: add position info to token info
        x = tok_emb + pos_emb   # broadcasting: (batch, seq_len, embed_dim)

        # Create causal mask (prevent attending to future tokens)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=token_ids.device),
            diagonal=1
        )

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        # Final layer norm
        x = self.final_norm(x)

        # Project to vocabulary size -> logits
        logits = self.lm_head(x)   # (batch, seq_len, vocab_size)

        return logits


# =============================================================================
# Shape walkthrough with random data
# =============================================================================
print("Shape walkthrough — forward pass with random token IDs")
print()

VOCAB_SIZE  = 100
EMBED_DIM   = 32
NUM_HEADS   = 4
NUM_LAYERS  = 3
MAX_SEQ_LEN = 16
BATCH_SIZE  = 2
SEQ_LEN     = 8

model = MiniTransformerLM(
    vocab_size   = VOCAB_SIZE,
    embed_dim    = EMBED_DIM,
    num_heads    = NUM_HEADS,
    num_layers   = NUM_LAYERS,
    max_seq_len  = MAX_SEQ_LEN,
)

# Random token IDs
token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
print(f"Input:  token_ids shape = {token_ids.shape}  (batch=2, seq=8)")
print()

# Walk through the shapes manually
tok_emb  = model.token_embedding(token_ids)
positions = torch.arange(SEQ_LEN)
pos_emb  = model.pos_embedding(positions)
x        = tok_emb + pos_emb

print(f"Step 1: token_embedding(token_ids)   -> {tok_emb.shape}")
print(f"Step 2: pos_embedding(positions)     -> {pos_emb.shape}")
print(f"Step 3: tok_emb + pos_emb            -> {x.shape}")

for i in range(NUM_LAYERS):
    print(f"Step {4+i}: TransformerBlock {i+1}           -> (same shape)")

model.eval()
with torch.no_grad():
    logits = model(token_ids)

print(f"Step {4+NUM_LAYERS}: final_norm + lm_head          -> {logits.shape}")
print()
print(f"Output: logits shape = {logits.shape}")
print(f"  -> (batch=2, seq=8, vocab=100)")
print(f"  -> for each of the 8 positions, we get 100 scores (one per vocab token)")
print(f"  -> the highest-scoring token is the model's prediction for the next token")

# Parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal model parameters: {total_params:,}")

print("\nParameter breakdown:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name:20s}: {params:>8,} params")

print("\nNote: lm_head has 0 params because it shares weights with token_embedding")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
A transformer block packages four components:

  1. MULTI-HEAD SELF-ATTENTION
     - Each token queries all other tokens
     - Learns what to attend to (which words are relevant)
     - Causal mask prevents future-peeking

  2. LAYER NORMALIZATION
     - Keeps activations in a healthy range
     - Applied before each sublayer (Pre-LN variant)

  3. FEED-FORWARD NETWORK
     - Two linear layers with GELU activation
     - Processes each token independently
     - 4x wider middle layer

  4. RESIDUAL CONNECTIONS
     - x + sublayer(x)  applied around each sublayer
     - Creates gradient highways through deep networks
     - Makes training 100s of layers possible

Full model = embedding + positional encoding + N blocks + norm + lm_head
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- LayerNorm tests ---
assert x_normed.shape == x_demo.shape, "LayerNorm should preserve shape"
# Each row should have mean ~0 and std ~1
for i in range(x_normed.shape[0]):
    assert abs(x_normed[i].mean().item()) < 1e-5, \
        f"LayerNorm row {i} mean should be ~0"

# --- FeedForwardNetwork tests ---
assert out_ffn.shape == x_ffn.shape, \
    f"FFN output shape should match input shape {x_ffn.shape}"
# Expected param count: Linear(32->128): 32*128+128=4224, Linear(128->32): 128*32+32=4128 => 8352
expected_ffn_params = EMBED_DIM * (EMBED_DIM * 4) + (EMBED_DIM * 4) + \
                      (EMBED_DIM * 4) * EMBED_DIM + EMBED_DIM
assert total_ffn == expected_ffn_params, \
    f"FFN param count should be {expected_ffn_params}, got {total_ffn}"

# --- TransformerBlock tests ---
assert x_out.shape == x_in.shape, \
    f"TransformerBlock output shape should match input {x_in.shape}, got {x_out.shape}"
assert isinstance(block_params, int) and block_params > 0, \
    "block_params should be a positive integer"

# --- Full MiniTransformerLM tests ---
assert logits.shape == torch.Size([BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]), \
    f"logits should be ({BATCH_SIZE}, {SEQ_LEN}, {VOCAB_SIZE}), got {logits.shape}"
assert total_params > 0, "model should have positive parameter count"

# Weight tying: lm_head and token_embedding share the same weight tensor
assert model.lm_head.weight is model.token_embedding.weight, \
    "lm_head.weight and token_embedding.weight should be the same tensor (weight tying)"

# Logits softmax should produce valid probabilities at each position
import torch.nn.functional as _F
probs_test = _F.softmax(logits[0, 0], dim=-1)
assert torch.allclose(probs_test.sum(), torch.tensor(1.0), atol=1e-5), \
    "softmax of logits should sum to 1.0"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("=" * 60)
print("EXERCISES (try these yourself!)")
print("=" * 60)
print("""
1. Understand the residual connection:
   - Create a TransformerBlock with embed_dim=16, num_heads=2.
   - Create input x = torch.zeros(1, 4, 16) (all zeros).
   - Run a forward pass.
   - Is the output also all zeros? Why or why not?
   (Hint: think about what LayerNorm does with all-zero input, and what
   the random weights in attention/FFN will produce.)

2. Scale up the model:
   - Create MiniTransformerLM with num_layers=6, embed_dim=64, num_heads=4.
   - Count the parameters. How many?
   - How many parameters does GPT-2 (small) have? (Answer: 117 million)
   - How much bigger is GPT-2 than your model?

3. Logits to probabilities:
   - Run a forward pass and get logits of shape (1, 5, 100).
   - Apply F.softmax(logits, dim=-1) to get probabilities.
   - Verify that probabilities at position 0 sum to 1.0.
   - What is the argmax (most likely token) at each position?

4. Weight tying experiment:
   - Check that model.lm_head.weight and model.token_embedding.weight
     are literally the same tensor (same id() in Python).
   - Why does weight tying make sense? Think about what both layers do.
   (Hint: embedding encodes tokens -> vectors; lm_head decodes vectors -> tokens.)

5. Dropout behavior:
   - Run a forward pass in model.train() mode vs model.eval() mode
     on the same input.
   - Are the outputs the same? Why or why not?
   (Hint: dropout randomly zeros values ONLY during training.)
""")
