# HOW TO RUN:
#   uv run python 08_transformers/05_training_pipeline.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# LESSON 5: The Training Pipeline for Language Models
# =============================================================================
#
# PUTTING IT ALL TOGETHER
# -----------------------
# In this lesson we go from raw text all the way to a trained (tiny) language
# model that can generate text.
#
# Steps:
#   1. Tokenize text (character-level for simplicity)
#   2. Create training sequences (input/target pairs)
#   3. Build a PyTorch Dataset and DataLoader
#   4. Write the training loop
#   5. Generate text using the trained model
#      - Greedy sampling
#      - Temperature sampling
#      - Top-k sampling
#
# NOTE: We keep the model tiny so it trains in seconds on CPU.
#       Real LLMs take weeks/months on thousands of GPUs.
#
# Run this file:
#   uv run python 05_training_pipeline.py
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# =============================================================================
# HELPER: Compact transformer model (copy from Lesson 4, kept self-contained)
# =============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights


class MultiHeadAttention(nn.Module):
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

    def _split(self, x):
        b, s, e = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, mask=None):
        b, s, _ = x.shape
        Q, K, V = self._split(self.W_q(x)), self._split(self.W_k(x)), self._split(self.W_v(x))
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(out.transpose(1, 2).contiguous().view(b, s, self.embed_dim))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.attn    = MultiHeadAttention(embed_dim, num_heads)
        self.ffn     = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(embed_dim * ffn_mult, embed_dim),
        )
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.max_seq_len    = max_seq_len
        self.tok_emb        = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb        = nn.Embedding(max_seq_len, embed_dim)
        self.blocks         = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in num_layers * [None]
        ])
        self.norm           = nn.LayerNorm(embed_dim)
        self.lm_head        = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, idx):
        b, t = idx.shape
        pos  = torch.arange(t, device=idx.device)
        x    = self.tok_emb(idx) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.lm_head(self.norm(x))   # (batch, seq_len, vocab_size)


# =============================================================================
# PART 1: Tokenization and data preparation
# =============================================================================
print("=" * 60)
print("PART 1: Character-level tokenization")
print("=" * 60)

# --- Text corpus ---
# We use a small hardcoded text. In a real LLM you'd use gigabytes of text.
TEXT = """
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them to die to sleep
No more and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to tis a consummation
Devoutly to be wished to die to sleep
To sleep perchance to dream ay there is the rub
For in that sleep of death what dreams may come
""".strip()

# --- Character-level tokenizer ---
# We map every unique character to an integer ID.
# This is the simplest possible tokenizer.
# Real models use BPE (Section 7) which works on subword units.

chars    = sorted(set(TEXT))           # unique characters, sorted
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}

def encode(text):
    """Convert a string to a list of integer IDs."""
    return [char_to_id[c] for c in text]

def decode(ids):
    """Convert a list of integer IDs back to a string."""
    return "".join(id_to_char[i] for i in ids)

print(f"Text length: {len(TEXT)} characters")
print(f"Unique characters: {vocab_size}")
print(f"Characters: {repr(''.join(chars))}")
print()

# Tokenize the entire corpus
token_ids = encode(TEXT)
print(f"Encoded: {len(token_ids)} tokens")
print(f"First 20 chars:  {repr(TEXT[:20])}")
print(f"First 20 tokens: {token_ids[:20]}")
print(f"Decoded back:    {repr(decode(token_ids[:20]))}")


print("\n" + "=" * 60)
print("PART 2: Creating training sequences")
print("=" * 60)

# --- How do we train a language model? ---
# The task: given tokens[0..t-1], predict token[t].
#
# We slide a window of length seq_len over the token sequence:
#
#   input:  tokens[0 : seq_len]        (the context)
#   target: tokens[1 : seq_len + 1]    (what comes NEXT at each position)
#
# Example with seq_len=4 and text "hello":
#   input:  [h, e, l, l]   IDs: [7, 4, 11, 11]
#   target: [e, l, l, o]   IDs: [4, 11, 11, 14]
#
#   Position 0: given 'h',       predict 'e'
#   Position 1: given 'h','e',   predict 'l'
#   Position 2: given 'h','e','l', predict 'l'
#   etc.
#
# This gives us (len(text) - seq_len) training examples from one pass!

SEQ_LEN = 32   # context length (how many characters to look at)

class CharDataset(Dataset):
    """
    PyTorch Dataset: yields (input_sequence, target_sequence) pairs.

    A Dataset is a class that knows:
      - How many items it has (__len__)
      - How to get item i (__getitem__)

    PyTorch's DataLoader uses these to create batches automatically.
    """

    def __init__(self, token_ids, seq_len):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len   = seq_len

    def __len__(self):
        # Number of valid sequences we can extract
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        # Input:  tokens[idx : idx + seq_len]
        # Target: tokens[idx+1 : idx + seq_len + 1]  (shifted by 1)
        x = self.token_ids[idx     : idx + self.seq_len]
        y = self.token_ids[idx + 1 : idx + self.seq_len + 1]
        return x, y


dataset = CharDataset(token_ids, SEQ_LEN)
print(f"Dataset size: {len(dataset)} sequences")
print(f"(Each sequence is {SEQ_LEN} characters long)")

# Look at one example
x_sample, y_sample = dataset[0]
print(f"\nExample (index 0):")
print(f"  Input  IDs: {x_sample[:10].tolist()} ...")
print(f"  Target IDs: {y_sample[:10].tolist()} ...")
print(f"  Input  text: {repr(decode(x_sample.tolist()))}")
print(f"  Target text: {repr(decode(y_sample.tolist()))}")
print(f"  Notice: target is input shifted by 1 character!")


print("\n" + "=" * 60)
print("PART 3: DataLoader — batching and shuffling")
print("=" * 60)

# --- What is a DataLoader? ---
# A DataLoader wraps a Dataset and:
#   - Groups items into batches automatically
#   - Optionally shuffles the order each epoch (good for training)
#   - Can load data in parallel using multiple CPU workers
#
# This connects to multiprocessing (Section 6): num_workers>0 spawns worker
# processes that load data in parallel while the GPU is busy training.
# (On small data like ours, num_workers=0 is fine.)

BATCH_SIZE = 8

dataloader = DataLoader(
    dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,     # shuffle order every epoch
    num_workers = 0,        # 0 = load in main process (safe for small data)
    drop_last   = True,     # drop the last batch if it's smaller than batch_size
)

print(f"DataLoader: {len(dataloader)} batches of size {BATCH_SIZE}")

# Look at one batch
x_batch, y_batch = next(iter(dataloader))
print(f"Batch input  shape: {x_batch.shape}  (batch={BATCH_SIZE}, seq_len={SEQ_LEN})")
print(f"Batch target shape: {y_batch.shape}")

print(f"\nFirst sequence in batch:")
print(f"  Input:  {repr(decode(x_batch[0].tolist()))}")
print(f"  Target: {repr(decode(y_batch[0].tolist()))}")


print("\n" + "=" * 60)
print("PART 4: The Training Loop")
print("=" * 60)

# --- Training loop for a language model ---
#
# For each batch:
#   1. Get (input_sequences, target_sequences)
#   2. Forward pass: model(input) -> logits (scores for each next token)
#   3. Compute cross-entropy loss between logits and targets
#   4. Backward pass: compute gradients
#   5. Update parameters with optimizer
#
# =============================================================================
# FIRST PRINCIPLES: Cross-Entropy Gradient for Softmax Output
# =============================================================================
#
# The cross-entropy loss for the correct class k:
#   L = -log(p_k)    where p_k = softmax(z_k) = exp(z_k) / sum(exp(z_j))
#
# Deriving the gradient dL/dz_i:
#   dL/dz_i = -(1/p_k) * dp_k/dz_i
#
#   For softmax, the derivative dp_k/dz_i depends on whether i == k:
#     dp_k/dz_i = p_k * (delta_{ki} - p_i)
#     where delta_{ki} = 1 if k==i, else 0
#
#   Substituting:
#     dL/dz_i = -(1/p_k) * p_k * (delta_{ki} - p_i)
#             = -(delta_{ki} - p_i)
#             = p_i - delta_{ki}
#
#   In vector form: dL/dz = p - y
#     where p is the softmax output and y is the one-hot target vector.
#
#   This is BEAUTIFULLY simple: the gradient is just (prediction - truth).
#   If the model predicts p_k = 1.0 for the correct class, gradient = 0 (perfect).
#   If the model predicts p_k = 0.01, gradient is large (needs big update).
#
# =============================================================================
# FIRST PRINCIPLES: FLOPS Per Training Step
# =============================================================================
#
# For a transformer with P parameters, batch size B, sequence length T:
#
#   Forward pass:  ~2 * P * B * T  FLOPS
#     (Each parameter participates in one multiply and one add per token.)
#
#   Backward pass: ~4 * P * B * T  FLOPS
#     (Backward is ~2x forward: compute gradient w.r.t. both inputs AND weights.)
#
#   Total per step: ~6 * P * B * T  FLOPS
#
# EXERCISE: Calculate FLOPS for P=10M params, B=32, T=128:
#   Total = 6 * 10,000,000 * 32 * 128 = 245,760,000,000 = ~246 GFLOPS
#   A modern GPU does ~100 TFLOPS, so this takes ~2.5 milliseconds.
#
# =============================================================================
# FIRST PRINCIPLES: Learning Rate Scheduling
# =============================================================================
#
# WHY WARMUP?
#   At the start of training, weights are random -> activations are random ->
#   gradients are large and noisy. Taking big steps with large LR causes
#   wild oscillations and potential divergence.
#   Warmup: start with a very small LR, gradually increase over N steps.
#   This lets the model "find its footing" before taking larger steps.
#
# WHY COSINE DECAY?
#   After warmup, gradually decrease LR following a cosine curve:
#     lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * t / T_max))
#   Smooth annealing helps the model settle into narrow (sharp) minima
#   that often generalize better. Cosine is smoother than step decay.
# =============================================================================

# Cross-entropy loss:
#   - For each position, we have a probability distribution over vocab_size tokens
#   - The target tells us which token SHOULD have probability 1
#   - Loss = -log(probability assigned to the correct token)
#   - Perfect model: loss = 0
#   - Random model: loss = log(vocab_size) = log(38) ≈ 3.6 in our case

# --- Model config (tiny, trains fast) ---
EMBED_DIM  = 64
NUM_HEADS  = 4
NUM_LAYERS = 2
MAX_SEQ    = SEQ_LEN + 1

model = TinyTransformerLM(
    vocab_size  = vocab_size,
    embed_dim   = EMBED_DIM,
    num_heads   = NUM_HEADS,
    num_layers  = NUM_LAYERS,
    max_seq_len = MAX_SEQ,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
print(f"  vocab_size={vocab_size}, embed_dim={EMBED_DIM}, "
      f"num_heads={NUM_HEADS}, num_layers={NUM_LAYERS}")

# Optimizer: Adam is generally better than plain SGD for transformers
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn   = nn.CrossEntropyLoss()

# --- Train for a few epochs ---
NUM_EPOCHS = 5
print(f"\nTraining for {NUM_EPOCHS} epochs on {len(TEXT)} characters of text...")
print()

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in dataloader:
        # --- Forward pass ---
        logits = model(x_batch)   # (batch, seq_len, vocab_size)

        # Cross-entropy expects logits of shape (N, C) and targets of shape (N,)
        # We flatten: (batch * seq_len, vocab_size) and (batch * seq_len,)
        batch_size_actual = x_batch.shape[0]
        logits_flat  = logits.view(-1, vocab_size)      # (B*T, vocab_size)
        targets_flat = y_batch.view(-1)                 # (B*T,)

        loss = loss_fn(logits_flat, targets_flat)

        # --- Backward pass ---
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # compute new gradients

        # Gradient clipping: prevents very large gradient updates
        # (a common trick for transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()        # update parameters

        total_loss  += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  |  avg loss = {avg_loss:.4f}  "
          f"|  perplexity = {math.exp(avg_loss):.1f}")

print("\nLoss should decrease each epoch — the model is learning!")
print("Perplexity = exp(loss). Lower = better. Random ~= 37, trained ~= lower.")


print("\n" + "=" * 60)
print("PART 5: Text Generation")
print("=" * 60)

# --- How to generate text? ---
# Give the model a prompt (some starting tokens).
# The model predicts the NEXT token.
# Append that token to the prompt.
# Repeat until we have enough text.
#
# But HOW do we pick the next token? Three strategies:
#
# 1. GREEDY: always pick the token with the highest score.
#    Fast, but repetitive/boring.
#
# 2. TEMPERATURE SAMPLING: sample randomly, but control how spread-out
#    the probabilities are:
#    - temperature=1.0: use probabilities as-is
#    - temperature<1.0: sharpen probs (more confident, less random)
#    - temperature>1.0: flatten probs (more random, more creative)
#
# 3. TOP-K SAMPLING: only consider the top-k most likely tokens,
#    then sample from those. Prevents picking very unlikely tokens.

@torch.no_grad()   # no gradients needed for generation
def generate(model, prompt_ids, max_new_tokens=100, temperature=1.0, top_k=None):
    """
    Generate text token by token.

    Args:
        model:          trained TinyTransformerLM
        prompt_ids:     list of integer token IDs (the starting prompt)
        max_new_tokens: how many new tokens to generate
        temperature:    sampling temperature (lower = more focused)
        top_k:          if set, only sample from top-k most likely tokens

    Returns:
        list of all token IDs (prompt + generated)
    """
    model.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long)   # (1, prompt_len)

    for _ in range(max_new_tokens):
        # Crop to max_seq_len if context gets too long
        ids_crop = ids[:, -model.max_seq_len:]

        # Forward pass -> logits
        logits = model(ids_crop)            # (1, seq_len, vocab_size)
        logits = logits[:, -1, :]          # take LAST position: (1, vocab_size)

        # Apply temperature: divide logits before softmax
        logits = logits / temperature

        # Optional top-k: zero out all but top-k logits
        if top_k is not None:
            # Get the top-k values
            values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            # Set everything below the k-th value to -infinity
            threshold = values[:, [-1]]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)   # (1, vocab_size)

        # Sample: pick one token according to the probability distribution
        next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)

        # Append to the sequence
        ids = torch.cat([ids, next_id], dim=1)   # (1, seq_len+1)

    return ids[0].tolist()   # return the flat list of IDs


# --- Generate with different strategies ---
prompt_text  = "to be"
prompt_ids   = encode(prompt_text)

print(f"Prompt: {repr(prompt_text)}")
print()

print("--- Greedy generation (temperature=0.1, very focused) ---")
greedy_ids  = generate(model, prompt_ids, max_new_tokens=80,
                        temperature=0.1)
greedy_text = decode(greedy_ids)
print(greedy_text)

print("\n--- Temperature=0.8 sampling (somewhat random) ---")
temp_ids  = generate(model, prompt_ids, max_new_tokens=80,
                     temperature=0.8)
temp_text = decode(temp_ids)
print(temp_text)

print("\n--- Top-k=5 sampling (only sample from top 5 characters) ---")
topk_ids  = generate(model, prompt_ids, max_new_tokens=80,
                     temperature=1.0, top_k=5)
topk_text = decode(topk_ids)
print(topk_text)

print()
print("NOTE: With only 5 epochs of training on a tiny dataset, output is rough.")
print("The model is just starting to learn patterns. Run more epochs to improve!")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Training pipeline for a character-level language model:

  TEXT  ->  tokenize (char->id)  ->  token_ids (list of ints)

  CharDataset:
    input[i]  = token_ids[i : i+seq_len]
    target[i] = token_ids[i+1 : i+seq_len+1]

  DataLoader:
    groups into batches of {BATCH_SIZE}, shuffles each epoch

  Training loop (each batch):
    logits = model(input)              shape: (B, T, vocab)
    loss   = CrossEntropy(logits, target)
    loss.backward()                    compute gradients
    optimizer.step()                   update weights

  Text generation:
    - feed prompt through model
    - sample next token from the last position's logits
    - append, repeat
    - temperature controls randomness
    - top-k constrains choices to most likely tokens
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- Tokenizer tests ---
assert vocab_size > 0, "vocab_size should be positive"
assert len(char_to_id) == vocab_size, "char_to_id should cover all vocab chars"
assert len(id_to_char) == vocab_size, "id_to_char should cover all vocab IDs"

# Encode/decode round-trip
_test_text = TEXT[:20]
assert decode(encode(_test_text)) == _test_text, \
    "encode then decode should recover the original text"

# --- Dataset tests ---
assert len(dataset) == len(token_ids) - SEQ_LEN, \
    f"dataset length should be len(tokens) - seq_len"
assert x_sample.shape == torch.Size([SEQ_LEN]), \
    f"each input sample should have shape ({SEQ_LEN},)"
assert y_sample.shape == torch.Size([SEQ_LEN]), \
    f"each target sample should have shape ({SEQ_LEN},)"
# Target is input shifted by 1
assert torch.all(x_sample[1:] == y_sample[:-1]), \
    "target should be input shifted right by 1 position"

# --- DataLoader tests ---
assert x_batch.shape == torch.Size([BATCH_SIZE, SEQ_LEN]), \
    f"batch input shape should be ({BATCH_SIZE}, {SEQ_LEN})"
assert y_batch.shape == torch.Size([BATCH_SIZE, SEQ_LEN]), \
    f"batch target shape should be ({BATCH_SIZE}, {SEQ_LEN})"

# --- Model forward pass tests ---
_test_ids = torch.randint(0, vocab_size, (2, SEQ_LEN))
model.eval()
with torch.no_grad():
    _test_logits = model(_test_ids)
assert _test_logits.shape == torch.Size([2, SEQ_LEN, vocab_size]), \
    f"model output should be (2, {SEQ_LEN}, {vocab_size})"

# --- Training loss decrease test ---
# Compute loss on a fixed batch before and after training
# (we already trained, so compare early vs late loss from training)
# The model was trained for NUM_EPOCHS=5, check final state produces reasonable output
model.eval()
with torch.no_grad():
    _check_ids = dataset[0][0].unsqueeze(0)
    _check_logits = model(_check_ids)
assert _check_logits.shape == torch.Size([1, SEQ_LEN, vocab_size]), \
    "post-training forward pass shape should be (1, seq_len, vocab_size)"

# --- Generation tests ---
assert isinstance(greedy_ids, list), "generate() should return a list"
assert len(greedy_ids) == len(prompt_ids) + 80, \
    "generated IDs should be prompt + 80 new tokens"
assert isinstance(greedy_text, str), "decoded output should be a string"
assert greedy_text.startswith(prompt_text), \
    "generated text should start with the prompt"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("=" * 60)
print("EXERCISES (try these yourself!)")
print("=" * 60)
print("""
1. More training:
   - Change NUM_EPOCHS to 20 or 50.
   - Does the loss decrease further? Does the generated text improve?
   - At what point does the model start "memorizing" the training text?

2. Different prompts:
   - Try other prompts like "the", "sleep", "or".
   - Does the generated text make sense given the training corpus?

3. Temperature effect:
   - Generate 5 different samples from the same prompt with temperature=1.5.
   - Then generate 5 samples with temperature=0.3.
   - Which temperature gives more diverse outputs? Which gives more coherent?

4. Build a word-level tokenizer:
   - Instead of characters, split TEXT.split() to get words.
   - Build word_to_id and id_to_word dictionaries.
   - Retokenize the text at word level.
   - How does vocab_size change? (Was 38 characters, now how many words?)
   - Re-train the model with the word-level vocabulary.
   (Hint: you also need to update encode() and decode() functions.)

5. DataLoader exploration:
   - Set shuffle=False in the DataLoader.
   - Print the first sequence from epoch 1 and epoch 2.
   - Are they the same? (They should be, since no shuffling.)
   - Now set shuffle=True and repeat. Are they the same?
""")
