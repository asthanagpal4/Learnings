# HOW TO RUN:
#   uv run python 08_transformers/project_pretrain_mini_llm.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# CAPSTONE PROJECT: Pre-train a Mini LLM from Scratch
# =============================================================================
#
# WHAT WE ARE BUILDING
# --------------------
# A ~2 million parameter transformer language model trained from scratch on a
# text corpus. This is the same process used to build GPT, LLaMA, etc. —
# just massively scaled up.
#
# Here we do the same thing in miniature:
#   - ~2M parameters (GPT-2 has 117M, GPT-4 has ~1.8 trillion)
#   - Character-level tokenization (real LLMs use BPE from Section 7)
#   - ~100-200 training steps on a small text corpus
#   - Trains in under a minute on CPU
#
# NOTE ON TOKENIZER:
#   We use character-level tokenization here to keep this file completely
#   self-contained. In a real scenario you would use the BPE tokenizer from
#   Section 7, which gives a much smaller sequence length for the same text
#   (because BPE groups common character sequences into single tokens).
#
# GOAL: After training, the model generates text that imitates the style and
# patterns of the training corpus — this is exactly what "pre-training" means!
#
# Run this file:
#   uv run python project_pretrain_mini_llm.py
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time

# =============================================================================
# CONFIGURATION — tune these to experiment
# =============================================================================

CONFIG = {
    # Model architecture
    "embed_dim"   : 128,    # embedding dimension (d_model)
    "num_heads"   : 4,      # number of attention heads
    "num_layers"  : 4,      # number of transformer blocks
    "ffn_mult"    : 4,      # FFN hidden dim = embed_dim * ffn_mult
    "dropout"     : 0.1,    # dropout rate

    # Training
    "seq_len"     : 64,     # context length (characters)
    "batch_size"  : 16,     # sequences per batch
    "num_steps"   : 200,    # training steps (not epochs)
    "lr"          : 3e-3,   # learning rate
    "print_every" : 10,     # print loss every N steps

    # Generation
    "gen_length"  : 200,    # characters to generate
    "temperature" : 0.8,    # generation temperature
    "top_k"       : 10,     # top-k sampling (characters)
}

# =============================================================================
# TEXT CORPUS
# =============================================================================
# We use a hardcoded text that is rich enough for the model to learn patterns.
# Around 3000 characters — long enough to see loss decrease meaningfully.

CORPUS = """
In the beginning there was the word and the word was with the universe.
The universe was vast and dark and cold and silent for a long long time.
Then slowly patterns began to emerge from the chaos of the early cosmos.
Stars formed from clouds of gas and dust pulled together by gravity.
These stars burned brightly for millions and billions of years.
Around some of these stars planets formed from the leftover material.
On one small blue planet a remarkable thing happened quite by accident.
Simple molecules combined and recombined in the warm shallow oceans.
Eventually these molecules could copy themselves and life had begun.
Life evolved slowly over billions of years into countless forms.
From the first simple cells to fish to reptiles to mammals to humans.
Each generation inherited the patterns of the previous generation.
But also added small variations and changes through mutation and selection.
The process of evolution is patient and relentless and creative.
It has no goal but it produces goals it has no mind but produces minds.
The human mind is the universe becoming aware of itself for the first time.
Through human eyes the cosmos sees its own stars and wonders at them.
Through human hands the universe reaches out and shapes the world.
We are the universe thinking about the universe.
We are matter that has learned to contemplate matter.
We are energy that has organized itself into patterns that ask why.
Language is the tool that makes all of this possible.
With words we can share thoughts across time and space.
With writing we can speak to people not yet born.
With mathematics we can describe the laws of nature precisely.
With science we can discover hidden truths about reality.
The search for knowledge is one of the great human adventures.
Each discovery opens new questions and new mysteries to explore.
The more we learn the more we realize how much remains to be discovered.
Curiosity is the engine of human progress and understanding.
Ask questions and seek answers and share what you find with others.
That is the spirit of science and learning and human civilization.
The universe is not obligated to make sense to us but somehow it does.
The laws of physics are the same everywhere in the observable universe.
A hydrogen atom on the other side of the galaxy follows the same rules.
This deep uniformity of nature is one of the greatest miracles of all.
Intelligence is the ability to learn patterns from experience.
A child learns language by hearing thousands of examples over many years.
A neural network learns patterns by processing millions of examples too.
Both use gradient descent in some form to update their internal models.
The transformer architecture has revolutionized machine learning.
By learning to pay attention to the right things at the right time.
By building deep hierarchical representations of language and thought.
These models can translate languages summarize documents write code.
They can answer questions generate images compose music explain concepts.
We are living through a remarkable transformation in human capability.
The tools we build are becoming extensions of our own intelligence.
This brings great opportunity and also great responsibility.
We must be thoughtful about how we develop and deploy these systems.
The future belongs to those who understand both the power and the limits.
Learning is not just about acquiring information but about building understanding.
True understanding means being able to apply knowledge in new situations.
It means seeing the connections between different ideas and domains.
It means being able to explain things clearly to others.
The best way to learn something deeply is to try to teach it to someone else.
Or to build it yourself from scratch as you are doing right now.
""".strip()

print("=" * 60)
print("CAPSTONE PROJECT: Pre-train a Mini LLM from Scratch")
print("=" * 60)
print(f"\nCorpus: {len(CORPUS)} characters")


# =============================================================================
# STEP 1: CHARACTER-LEVEL TOKENIZER
# =============================================================================
print("\n" + "-" * 50)
print("STEP 1: Building the tokenizer")
print("-" * 50)

# Build vocabulary from corpus
chars      = sorted(set(CORPUS))
VOCAB_SIZE = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}

def encode(text):
    """Convert text string -> list of integer token IDs."""
    return [char_to_id[c] for c in text if c in char_to_id]

def decode(ids):
    """Convert list of integer token IDs -> text string."""
    return "".join(id_to_char.get(i, "?") for i in ids)

print(f"Vocabulary size: {VOCAB_SIZE} unique characters")
print(f"Characters: {repr(''.join(chars))}")

# Encode the entire corpus
all_token_ids = encode(CORPUS)
print(f"Corpus tokenized: {len(all_token_ids)} tokens")
print(f"Compression: each character = 1 token (no compression at char level)")
print(f"  (BPE from Section 7 would reduce this by ~3-4x by grouping chars)")


# =============================================================================
# STEP 2: DATASET PREPARATION
# =============================================================================
print("\n" + "-" * 50)
print("STEP 2: Preparing the dataset")
print("-" * 50)

SEQ_LEN = CONFIG["seq_len"]

class TextDataset(Dataset):
    """
    Sliding window dataset over a token sequence.

    Each item is an (input, target) pair:
      input  = tokens[i     : i + seq_len]
      target = tokens[i + 1 : i + seq_len + 1]

    The model learns: given the first T tokens, predict the next token
    at every position simultaneously.
    """
    def __init__(self, token_ids, seq_len):
        self.data    = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


dataset    = TextDataset(all_token_ids, SEQ_LEN)
dataloader = DataLoader(
    dataset,
    batch_size  = CONFIG["batch_size"],
    shuffle     = True,
    drop_last   = True,
)

print(f"Sequence length:  {SEQ_LEN} characters")
print(f"Dataset size:     {len(dataset)} sequences")
print(f"Batch size:       {CONFIG['batch_size']}")
print(f"Batches per pass: {len(dataloader)}")
print()
# Show one example
x0, y0 = dataset[0]
print("Example training pair (index 0):")
print(f"  Input:  {repr(decode(x0[:30].tolist()))} ...")
print(f"  Target: {repr(decode(y0[:30].tolist()))} ...")
print("  (target is input shifted right by 1 character)")


# =============================================================================
# STEP 3: THE TRANSFORMER MODEL
# =============================================================================
print("\n" + "-" * 50)
print("STEP 3: Building the Transformer model")
print("-" * 50)

# --- Helper: scaled dot-product attention ---
def attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    scale  = math.sqrt(Q.shape[-1])
    scores = Q @ K.transpose(-2, -1) / scale
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ V


# --- Multi-head self-attention ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        # Single matrix for Q, K, V projection (more efficient than 3 separate)
        self.qkv_proj  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        # Compute Q, K, V in one matrix multiply, then split
        qkv = self.qkv_proj(x)                      # (B, T, 3*C)
        Q, K, V = qkv.split(self.embed_dim, dim=-1)  # each (B, T, C)

        # Split into heads: (B, T, C) -> (B, heads, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Expand mask for batch and head dims
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(0).unsqueeze(0)

        # Attention
        out = attention(Q, K, V, mask=attn_mask)     # (B, heads, T, head_dim)

        # Merge heads: (B, heads, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# --- Feed-Forward Network ---
class FFN(nn.Module):
    def __init__(self, embed_dim, mult=4, dropout=0.1):
        super().__init__()
        hidden = embed_dim * mult
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# --- Single transformer block ---
class TransformerBlock(nn.Module):
    """
    One transformer block:
      x -> norm1 -> MHA -> + residual
        -> norm2 -> FFN -> + residual
    """
    def __init__(self, embed_dim, num_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn   = FFN(embed_dim, ffn_mult, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        x = x + self.drop(self.attn(self.norm1(x), mask=mask))
        # Feed-forward with residual
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# --- Full transformer language model ---
class MiniLLM(nn.Module):
    """
    Mini transformer language model.

    Architecture:
      token IDs
        -> token embedding + positional embedding
        -> num_layers * TransformerBlock
        -> LayerNorm
        -> linear head (embed_dim -> vocab_size)
        -> logits
    """

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 max_seq_len, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.emb_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])

        # Final norm + output projection
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head    = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share weights between input embedding and output projection.
        # This saves parameters and is standard practice.
        self.lm_head.weight = self.tok_emb.weight

        # Initialize weights for stable training
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        """
        Args:
            token_ids: (batch, seq_len)  integer token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = token_ids.shape
        assert T <= self.max_seq_len, \
            f"Sequence {T} > max_seq_len {self.max_seq_len}"

        # Build embeddings
        tok  = self.tok_emb(token_ids)                               # (B, T, C)
        pos  = self.pos_emb(torch.arange(T, device=token_ids.device))  # (T, C)
        x    = self.emb_drop(tok + pos)

        # Causal mask: token i cannot attend to positions > i
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=token_ids.device),
            diagonal=1
        )

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Project to vocabulary
        logits = self.lm_head(self.final_norm(x))   # (B, T, vocab_size)
        return logits


# --- Instantiate model ---
model = MiniLLM(
    vocab_size   = VOCAB_SIZE,
    embed_dim    = CONFIG["embed_dim"],
    num_heads    = CONFIG["num_heads"],
    num_layers   = CONFIG["num_layers"],
    max_seq_len  = SEQ_LEN + 1,
    ffn_mult     = CONFIG["ffn_mult"],
    dropout      = CONFIG["dropout"],
)

# =============================================================================
# FIRST PRINCIPLES: Full Parameter Count Derivation
# =============================================================================
#
# For this model with embed_dim=128, num_heads=4, num_layers=4, ffn_mult=4,
# vocab_size=V (depends on corpus), max_seq_len=65:
#
# 1. Token embedding (tok_emb): V x 128  parameters
# 2. Position embedding (pos_emb): 65 x 128 = 8,320  parameters
# 3. Embedding dropout (emb_drop): 0  parameters
#
# 4. Each TransformerBlock (x4 blocks):
#    a. LayerNorm (norm1): 128 + 128 = 256  (gamma + beta)
#    b. MultiHeadSelfAttention:
#       - qkv_proj: 128 x (3*128) = 49,152  (no bias)
#       - out_proj: 128 x 128     = 16,384  (no bias)
#       - attn_drop: 0
#       Subtotal attention: 65,536
#    c. LayerNorm (norm2): 128 + 128 = 256
#    d. FFN:
#       - Linear1: 128 x 512 + 512 = 66,048  (weight + bias)
#       - GELU: 0
#       - Linear2: 512 x 128 + 128 = 65,664  (weight + bias)
#       - Dropout: 0
#       Subtotal FFN: 131,712
#    e. Dropout (drop): 0
#    Block total: 256 + 65,536 + 256 + 131,712 = 197,760  per block
#    All 4 blocks: 4 x 197,760 = 791,040
#
# 5. Final LayerNorm (final_norm): 128 + 128 = 256
# 6. LM head (lm_head): V x 128  (but TIED with tok_emb, so 0 extra params)
#
# Total unique params = V*128 + 8,320 + 791,040 + 256 = V*128 + 799,616
# For V ~ 55: 55*128 + 799,616 = 7,040 + 799,616 = 806,656
#
# =============================================================================
# FIRST PRINCIPLES: Memory Usage Analysis
# =============================================================================
#
# PARAMETERS (float32 = 4 bytes each):
#   ~800K params x 4 bytes = ~3.2 MB
#
# OPTIMIZER STATES (Adam/AdamW stores 2 extra values per parameter):
#   - m (first moment / mean of gradients): same size as params = ~3.2 MB
#   - v (second moment / variance of gradients): same size as params = ~3.2 MB
#   Total optimizer: ~6.4 MB
#
# GRADIENTS (same size as parameters):
#   ~3.2 MB
#
# ACTIVATIONS (for backward pass, stored per layer):
#   Each layer stores: batch_size x seq_len x embed_dim activations
#   Per layer: 16 x 64 x 128 x 4 bytes = ~512 KB
#   With attention weights: 16 x 4_heads x 64 x 64 x 4 bytes = ~1 MB
#   Across 4 layers: ~6 MB
#
# TOTAL TRAINING MEMORY:
#   ~3.2 (params) + 6.4 (optimizer) + 3.2 (gradients) + 6 (activations) ~ 19 MB
#
# RULE OF THUMB: Training memory ~ 16-20x parameter count in bytes.
#   For Adam with mixed precision (float16 params, float32 optimizer states).
#   Our model: 800K params x 4 bytes x 16 ~ 51 MB (conservative estimate).
#   Actual is lower because our model is small and overhead is relatively less.
#
# =============================================================================
# FIRST PRINCIPLES: Training FLOPS Estimate
# =============================================================================
#
# Rule of thumb: Total training FLOPS ~ 6 x num_params x num_tokens_processed
#
# For our mini LLM:
#   num_params = ~800K
#   num_tokens_processed = num_steps x batch_size x seq_len
#                        = 200 x 16 x 64 = 204,800 tokens
#   Total FLOPS ~ 6 x 800,000 x 204,800 ~ 983 billion FLOPS ~ 1 TFLOP
#   On a modern CPU at ~100 GFLOPS: ~10 seconds. On GPU: <1 second.
#
# For comparison, GPT-2 (1.5B params, ~40B tokens):
#   Total FLOPS ~ 6 x 1.5e9 x 40e9 = 3.6e20 FLOPS = 360 ExaFLOPS
#   That is ~360 MILLION times more compute than our mini LLM!
#
# =============================================================================
# FIRST PRINCIPLES: Scaling Laws (Chinchilla)
# =============================================================================
#
# Chinchilla scaling law (2022): for compute-optimal training,
# the number of training tokens should scale LINEARLY with model size.
#
# Optimal ratio: ~20 tokens per parameter.
#   - 1B parameter model -> train on ~20B tokens
#   - 7B parameter model -> train on ~140B tokens
#   - 70B parameter model -> train on ~1.4T tokens
#
# Our mini LLM: ~800K params, trained on ~205K tokens.
#   Optimal would be: 800K x 20 = 16M tokens.
#   We are ~80x UNDERTRAINED! This is fine for a demo, but explains
#   why our generated text is rough.
#
# EXERCISE: Estimate total training cost in FLOPS for our mini LLM
# vs GPT-2 (1.5B params, ~40B tokens):
#   Mini LLM:  6 x 800,000 x 204,800       ~ 9.8 x 10^11 FLOPS
#   GPT-2:     6 x 1,500,000,000 x 40e9    ~ 3.6 x 10^20 FLOPS
#   GPT-2 used ~370 MILLION times more compute!
# =============================================================================

# --- Parameter count ---
total_params = sum(p.numel() for p in model.parameters())
# lm_head shares weights with tok_emb, so don't double-count
unique_params = sum(p.numel() for p in set(model.parameters()))

print(f"Model configuration:")
print(f"  embed_dim  = {CONFIG['embed_dim']}")
print(f"  num_heads  = {CONFIG['num_heads']}")
print(f"  num_layers = {CONFIG['num_layers']}")
print(f"  seq_len    = {SEQ_LEN}")
print(f"  vocab_size = {VOCAB_SIZE}")
print()
print(f"Total parameters (with weight tying): {unique_params:,}")
print()

# Breakdown
print("Parameter breakdown:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name:15s}: {params:>10,}")

print(f"\n  lm_head shares weights with tok_emb (counted once above)")
print()

# Sanity check: forward pass
x_test  = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
logits  = model(x_test)
print(f"Forward pass test:")
print(f"  Input:   {x_test.shape}  (batch=2, seq={SEQ_LEN})")
print(f"  Output:  {logits.shape}  (batch=2, seq={SEQ_LEN}, vocab={VOCAB_SIZE})")
print(f"  Initial random loss (expected ~log({VOCAB_SIZE})={math.log(VOCAB_SIZE):.2f}):")
loss_test = F.cross_entropy(logits.view(-1, VOCAB_SIZE), x_test.view(-1))
print(f"  Actual:  {loss_test.item():.4f}")


# =============================================================================
# STEP 4: TRAINING LOOP
# =============================================================================
print("\n" + "-" * 50)
print("STEP 4: Training the model")
print("-" * 50)

# --- Optimizer ---
# AdamW is Adam with weight decay — a slight penalty on large weights that
# prevents overfitting. Standard choice for training transformers.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = CONFIG["lr"],
    weight_decay = 0.01,
)

# --- Optional: learning rate warmup ---
# Real LLM training uses learning rate schedulers.
# We implement a simple linear warmup: start with very low LR, ramp up.
# This helps with training stability at the start.
WARMUP_STEPS = 20

def get_lr(step):
    if step < WARMUP_STEPS:
        return CONFIG["lr"] * (step + 1) / WARMUP_STEPS
    return CONFIG["lr"]


# --- Training ---
print(f"Training for {CONFIG['num_steps']} steps...")
print(f"(print loss every {CONFIG['print_every']} steps)\n")
print(f"{'Step':>6}  {'Loss':>8}  {'Perplexity':>12}  {'LR':>10}  {'Time(s)':>9}")
print("-" * 55)

model.train()
data_iter    = iter(dataloader)
start_time   = time.time()
step         = 0
loss_history = []

while step < CONFIG["num_steps"]:
    # Get next batch (loop the dataloader if we run out)
    try:
        x_batch, y_batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        x_batch, y_batch = next(data_iter)

    # Adjust learning rate for warmup
    current_lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    # --- Forward pass ---
    logits = model(x_batch)                         # (B, T, vocab_size)
    loss   = F.cross_entropy(
        logits.view(-1, VOCAB_SIZE),                # (B*T, vocab_size)
        y_batch.view(-1)                            # (B*T,)
    )

    # --- Backward pass ---
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping: prevents training instability from large gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    step += 1

    loss_history.append(loss.item())

    # Print progress
    if step % CONFIG["print_every"] == 0:
        elapsed    = time.time() - start_time
        avg_loss   = sum(loss_history[-CONFIG["print_every"]:]) / CONFIG["print_every"]
        perplexity = math.exp(min(avg_loss, 20))    # cap at e^20 to avoid overflow
        print(f"  {step:>4}  {avg_loss:>8.4f}  {perplexity:>12.1f}  "
              f"{current_lr:>10.5f}  {elapsed:>7.1f}s")

total_time = time.time() - start_time
final_loss = sum(loss_history[-10:]) / 10
print(f"\nTraining complete!")
print(f"  Total time:  {total_time:.1f} seconds")
print(f"  Final loss:  {final_loss:.4f}  (initial was ~{math.log(VOCAB_SIZE):.2f})")
print(f"  Improvement: loss reduced by {math.log(VOCAB_SIZE) - final_loss:.2f}")

# Show loss curve summary
thirds = CONFIG["num_steps"] // 3
early  = sum(loss_history[:thirds]) / thirds
mid    = sum(loss_history[thirds:2*thirds]) / thirds
late   = sum(loss_history[2*thirds:]) / (CONFIG["num_steps"] - 2*thirds)
print(f"\nLoss over time:")
print(f"  Early  (steps 1-{thirds}):   {early:.4f}")
print(f"  Middle (steps {thirds+1}-{2*thirds}):  {mid:.4f}")
print(f"  Late   (steps {2*thirds+1}-{CONFIG['num_steps']}): {late:.4f}")
print(f"  Trend: {'decreasing (good!)' if late < early else 'not decreasing — try lower lr'}")


# =============================================================================
# STEP 5: TEXT GENERATION
# =============================================================================
print("\n" + "-" * 50)
print("STEP 5: Generating text")
print("-" * 50)

@torch.no_grad()
def generate_text(model, prompt, max_new_tokens=200, temperature=0.8, top_k=10):
    """
    Generate text from a prompt.

    Args:
        model:          trained MiniLLM
        prompt:         string prompt to continue
        max_new_tokens: number of new characters to generate
        temperature:    controls randomness (lower = more focused)
        top_k:          sample from only the top-k most likely characters

    Returns:
        generated string (prompt + new text)
    """
    model.eval()

    # Encode the prompt
    ids = encode(prompt)
    if not ids:
        print("Warning: prompt contains unknown characters, using default.")
        ids = encode("the")

    ids_tensor = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_new_tokens):
        # Crop context to max_seq_len
        context = ids_tensor[:, -model.max_seq_len:]

        # Get logits for the last position
        logits = model(context)[:, -1, :] / temperature   # (1, vocab_size)

        # Top-k filtering
        if top_k is not None:
            k = min(top_k, VOCAB_SIZE)
            top_vals, _ = torch.topk(logits, k)
            cutoff = top_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        # Sample next token
        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids_tensor = torch.cat([ids_tensor, next_id], dim=1)

    return decode(ids_tensor[0].tolist())


# --- Generate text from multiple prompts ---
prompts = [
    "the universe",
    "learning is",
    "in the beginning",
    "intelligence is",
]

for prompt in prompts:
    print(f"\nPrompt: {repr(prompt)}")
    print("Generated:")
    text = generate_text(
        model,
        prompt,
        max_new_tokens = CONFIG["gen_length"],
        temperature    = CONFIG["temperature"],
        top_k          = CONFIG["top_k"],
    )
    print(text)
    print()

# --- Try different temperatures ---
print("\n--- Temperature comparison (same prompt: 'the') ---")
for temp in [0.3, 0.7, 1.2]:
    text = generate_text(model, "the", max_new_tokens=80, temperature=temp, top_k=None)
    print(f"\nTemperature={temp}:")
    print(text)


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("WHAT WE BUILT — SUMMARY")
print("=" * 60)
print(f"""
You just pre-trained a language model from scratch!

MODEL ARCHITECTURE:
  Token embedding      : {VOCAB_SIZE} chars x {CONFIG['embed_dim']} dims
  Positional embedding : {SEQ_LEN} positions x {CONFIG['embed_dim']} dims
  Transformer blocks   : {CONFIG['num_layers']} blocks
    Each block:
      - Multi-head attention ({CONFIG['num_heads']} heads x {CONFIG['embed_dim']//CONFIG['num_heads']} dims)
      - Feed-forward network ({CONFIG['embed_dim']} -> {CONFIG['embed_dim']*CONFIG['ffn_mult']} -> {CONFIG['embed_dim']})
      - Layer norm (x2)
      - Residual connections (x2)
  Final layer norm
  Output head          : {CONFIG['embed_dim']} -> {VOCAB_SIZE} (logits)
  Weight tying         : yes (saves ~{VOCAB_SIZE*CONFIG['embed_dim']:,} params)

  Total parameters: {unique_params:,}

TRAINING:
  Tokenizer    : character-level ({VOCAB_SIZE} tokens)
  Corpus       : {len(CORPUS)} characters / {len(all_token_ids)} tokens
  Seq length   : {SEQ_LEN}
  Batch size   : {CONFIG['batch_size']}
  Optimizer    : AdamW with LR warmup
  Steps        : {CONFIG['num_steps']}
  Time         : {total_time:.0f} seconds on CPU

WHAT REAL LLMs DO DIFFERENTLY:
  - BPE tokenizer (Section 7): vocabulary of 50k-100k tokens, not 38 chars
  - Much larger: GPT-3 has 175 BILLION parameters (not 2 million)
  - Trained on: terabytes of text (books, web, code) not ~3000 characters
  - Trained for: weeks on thousands of GPUs, not seconds on one CPU
  - Gradient: same principles! Same architecture! Just... bigger.

You understand the CORE architecture. Scaling is "just" engineering.
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- Tokenizer tests ---
assert VOCAB_SIZE > 0, "vocabulary should be non-empty"
assert len(char_to_id) == VOCAB_SIZE, "char_to_id size should equal VOCAB_SIZE"
_sample_text = CORPUS[:15]
assert decode(encode(_sample_text)) == _sample_text, \
    "encode/decode round-trip should recover original text"

# --- Dataset tests ---
assert len(dataset) == len(all_token_ids) - SEQ_LEN, \
    "dataset length should be num_tokens - seq_len"
_xi, _yi = dataset[0]
assert _xi.shape == torch.Size([SEQ_LEN]), f"dataset input shape should be ({SEQ_LEN},)"
assert _yi.shape == torch.Size([SEQ_LEN]), f"dataset target shape should be ({SEQ_LEN},)"
assert torch.all(_xi[1:] == _yi[:-1]), "target should be input shifted right by 1"

# --- Model output shape tests ---
_test_in = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
model.eval()
with torch.no_grad():
    _test_out = model(_test_in)
assert _test_out.shape == torch.Size([2, SEQ_LEN, VOCAB_SIZE]), \
    f"model output should be (2, {SEQ_LEN}, {VOCAB_SIZE}), got {_test_out.shape}"

# Logits at each position should form valid unnormalized scores (finite values)
assert torch.isfinite(_test_out).all(), \
    "model output logits should be finite (no NaN or Inf)"

# --- Weight tying test ---
assert model.lm_head.weight is model.tok_emb.weight, \
    "lm_head and tok_emb should share the same weight tensor"

# --- Training convergence test ---
# Loss should decrease: compare early loss (first 10%) vs late loss (last 10%)
early_steps = max(1, CONFIG["num_steps"] // 10)
late_steps  = max(1, CONFIG["num_steps"] // 10)
_early_loss = sum(loss_history[:early_steps]) / early_steps
_late_loss  = sum(loss_history[-late_steps:]) / late_steps
assert _late_loss < _early_loss, \
    f"training loss should decrease: early={_early_loss:.4f}, late={_late_loss:.4f}"

# --- Generation output type test ---
_gen_out = generate_text(model, "the", max_new_tokens=20)
assert isinstance(_gen_out, str), "generate_text should return a string"
assert _gen_out.startswith("the"), "generated text should start with the prompt"
assert len(_gen_out) >= 3, "generated text should be at least as long as prompt"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("=" * 60)
print("EXERCISES (try these yourself!)")
print("=" * 60)
print("""
1. Scale up carefully:
   - Double embed_dim to 256. How does parameter count change?
   - Train for 300 steps. Does loss go lower?
   - How much longer does it take to train?

2. Add more text:
   - Add another 2000-3000 characters to the CORPUS.
   - Does the model need more training steps to achieve the same loss?
   - Does the generated text become more varied?

3. Learning rate schedule:
   - Try lr=1e-3 vs lr=1e-2 vs lr=1e-4.
   - Plot (mentally) the loss curve for each.
   - Which learning rate causes training instability?

4. Analyze attention heads:
   - Modify MultiHeadSelfAttention.forward() to return attention weights.
   - Generate a short sequence and look at the attention pattern.
   - Do different heads attend to different positions?

5. BPE connection (Section 7):
   - Our VOCAB_SIZE is 38 (characters).
   - With BPE (Section 7), we might have vocab_size=1000 with avg token=3 chars.
   - The same corpus would be ~len(all_token_ids)/3 tokens long.
   - How would this change the SEQ_LEN we need? How would it affect training?

6. CAPSTONE CHALLENGE: Train on your own text!
   - Replace CORPUS with any text you like (a chapter from a book,
     Wikipedia article, your own writing, etc.).
   - The model will learn to imitate that text's style.
   - Try at least 3000 characters for interesting results.
   - How many steps until the loss stabilizes?
""")
