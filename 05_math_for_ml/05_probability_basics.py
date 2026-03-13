# HOW TO RUN:
#   uv run python 05_math_for_ml/05_probability_basics.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE 5: Probability Basics
# =============================================================================
# Probability is the language of uncertainty.
# Machine learning models are uncertain — they don't say "this IS a cat",
# they say "I'm 87% sure this is a cat". That's probability!
#
# Topics:
#   - Basic probability concepts (dice, coins)
#   - Probability distributions (uniform, normal)
#   - Softmax: turning any numbers into probabilities
#   - Cross-entropy loss: measuring how wrong probability predictions are
#   - Connection to ML classification problems
# =============================================================================

import numpy as np

print("=" * 60)
print("PART 1: BASIC PROBABILITY CONCEPTS")
print("=" * 60)

# ------------------------------------------------------------------
# Probability is a number between 0 and 1 that expresses HOW LIKELY
# an event is to happen.
#   P = 0   means impossible (never happens)
#   P = 1   means certain (always happens)
#   P = 0.5 means 50/50 (like flipping a fair coin)
#
# Rule 1: All possible outcomes must add up to 1.
# Rule 2: P(event) = (number of favorable outcomes) / (total outcomes)
# ------------------------------------------------------------------

print("\n--- Coin Flipping ---")
# Fair coin: P(heads) = 0.5, P(tails) = 0.5
p_heads = 0.5
p_tails = 0.5
print(f"P(heads) = {p_heads}")
print(f"P(tails) = {p_tails}")
print(f"Sum = {p_heads + p_tails}  (must always equal 1)")

# Simulate 10000 coin flips
np.random.seed(42)
flips  = np.random.choice(['H', 'T'], size=10000, p=[0.5, 0.5])
n_heads = np.sum(flips == 'H')
n_tails = np.sum(flips == 'T')
print(f"\nSimulating 10,000 coin flips:")
print(f"  Heads: {n_heads}  ({n_heads/10000:.3f})")
print(f"  Tails: {n_tails}  ({n_tails/10000:.3f})")
print(f"  (Should be close to 0.5 each — law of large numbers)")

print("\n--- Dice Rolling ---")
# Fair 6-sided die: each face has probability 1/6
sides   = np.arange(1, 7)
p_each  = 1 / 6
print(f"Possible outcomes: {sides}")
print(f"P(any single face) = 1/6 = {p_each:.4f}")
print(f"Sum of all probabilities = {p_each * 6:.1f}")

# Simulate 10000 dice rolls
rolls = np.random.randint(1, 7, size=10000)
print(f"\nSimulating 10,000 dice rolls:")
print(f"{'Face':>6} | {'Count':>6} | {'Frequency':>10} | {'Expected':>10}")
print("-" * 40)
for face in range(1, 7):
    count = np.sum(rolls == face)
    freq  = count / 10000
    print(f"{face:>6} | {count:>6} | {freq:>10.4f} | {p_each:>10.4f}")

# Probability of rolling ABOVE 4
p_above_4 = np.mean(rolls > 4)
print(f"\nP(roll > 4) = {p_above_4:.4f}  (expected: {2/6:.4f})")

print("\n" + "=" * 60)
print("PART 2: PROBABILITY DISTRIBUTIONS")
print("=" * 60)

# ------------------------------------------------------------------
# A probability DISTRIBUTION describes all possible values a random
# variable can take, along with how likely each value is.
# ------------------------------------------------------------------

print("\n--- Uniform Distribution ---")
# Every value in a range is equally likely.
# Example: picking a random hour between 0 and 24.

uniform_samples = np.random.uniform(low=0, high=10, size=10000)
print(f"Uniform(0, 10) — 10,000 samples:")
print(f"  Mean:     {uniform_samples.mean():.3f}  (expected: 5.0)")
print(f"  Min:      {uniform_samples.min():.3f}  (expected: ~0)")
print(f"  Max:      {uniform_samples.max():.3f}  (expected: ~10)")
print(f"  P(x < 5): {np.mean(uniform_samples < 5):.3f}  (expected: 0.5)")

# Histogram-style print
print("\n  Histogram of Uniform(0,10):")
bins = np.arange(0, 11, 1)
for i in range(len(bins) - 1):
    count = np.sum((uniform_samples >= bins[i]) & (uniform_samples < bins[i+1]))
    bar   = "#" * (count // 30)
    print(f"  [{bins[i]}-{bins[i+1]}): {bar}  ({count})")

print("\n--- Normal (Gaussian) Distribution ---")
# The "bell curve" — most values cluster around the mean.
# Defined by: mean (center) and std (spread).
# About 68% of values are within 1 std of the mean.
# About 95% are within 2 stds.
# About 99.7% are within 3 stds.
# This is called the "68-95-99.7 rule".

np.random.seed(0)
normal_samples = np.random.normal(loc=0, scale=1, size=10000)  # mean=0, std=1
print(f"Normal(mean=0, std=1) — 10,000 samples:")
print(f"  Mean:              {normal_samples.mean():.4f}  (expected: 0.0)")
print(f"  Std deviation:     {normal_samples.std():.4f}  (expected: 1.0)")
print(f"  P(-1 < x < 1):    {np.mean(np.abs(normal_samples) < 1):.4f}  (expected: ~0.68)")
print(f"  P(-2 < x < 2):    {np.mean(np.abs(normal_samples) < 2):.4f}  (expected: ~0.95)")
print(f"  P(-3 < x < 3):    {np.mean(np.abs(normal_samples) < 3):.4f}  (expected: ~0.997)")

# Bell curve histogram
print("\n  Bell curve histogram (Normal(0,1)):")
bin_edges = np.arange(-4, 5, 1)
for i in range(len(bin_edges) - 1):
    count = np.sum((normal_samples >= bin_edges[i]) & (normal_samples < bin_edges[i+1]))
    bar   = "#" * (count // 25)
    print(f"  [{bin_edges[i]:+d} to {bin_edges[i+1]:+d}): {bar}  ({count})")

# Real example: test scores might follow a normal distribution
scores = np.random.normal(loc=70, scale=10, size=1000)
print(f"\nSimulated test scores Normal(mean=70, std=10):")
print(f"  Mean:   {scores.mean():.1f}")
print(f"  Std:    {scores.std():.1f}")
print(f"  Min:    {scores.min():.1f}")
print(f"  Max:    {scores.max():.1f}")
print(f"  P(score > 90): {np.mean(scores > 90):.3f}  (students who scored >90)")

print("\n" + "=" * 60)
print("PART 3: SOFTMAX FUNCTION")
print("=" * 60)

# ------------------------------------------------------------------
# Problem: A neural network's last layer produces raw numbers
# (called "logits") like [2.5, -1.0, 0.8].
# These are NOT probabilities (they don't sum to 1, can be negative).
#
# Solution: Softmax converts any list of numbers into probabilities!
#
# Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
#
# Properties:
#   - All outputs are between 0 and 1
#   - All outputs SUM TO 1 (valid probability distribution)
#   - Larger input -> larger probability (preserves ordering)
#   - Exponentiation makes differences more extreme
# ------------------------------------------------------------------

# ===========================================================================
# FIRST PRINCIPLES: Deriving softmax from scratch
# ===========================================================================
# PROBLEM: We have raw scores (logits) from a neural network, e.g. [2.5, -1.0, 0.8].
# We want to convert them to PROBABILITIES. Probabilities must be:
#   1. Non-negative (>= 0)
#   2. Sum to 1
#
# STEP 1: Make everything positive.
#   The exponential function e^x is ALWAYS positive (for any x).
#   So exp([2.5, -1.0, 0.8]) = [12.18, 0.37, 2.23] -- all positive!
#
# STEP 2: Make them sum to 1.
#   Divide each by the total sum: each_i / sum_of_all
#   [12.18, 0.37, 2.23] / (12.18 + 0.37 + 2.23) = [0.824, 0.025, 0.151]
#
# RESULT: softmax(x_i) = exp(x_i) / SUM_j exp(x_j)
#
# CONNECTION TO PHYSICS (Boltzmann distribution):
#   In statistical mechanics, the probability of a system being in state i
#   with energy E_i at temperature T is:
#     P(state_i) = exp(-E_i / kT) / SUM_j exp(-E_j / kT)
#   This is EXACTLY softmax! The logits play the role of -E/kT.
#   Higher logit = lower "energy" = more probable state.
#   In ML, we can think of logits as negative energies: the network
#   assigns low "energy" to the class it thinks is correct.
#
# WHY exponential specifically? Because:
#   1. Always positive (requirement for probabilities)
#   2. Preserves ordering (larger input -> larger output)
#   3. Differences in inputs become RATIOS of probabilities:
#      softmax(a) / softmax(b) = exp(a) / exp(b) = exp(a - b)
#      This means only the DIFFERENCE between logits matters.
# ===========================================================================

def softmax(x):
    """
    Convert a vector of numbers into probabilities.
    Uses a numerical stability trick: subtract the maximum first.
    (This prevents overflow with very large numbers.)
    """
    x_shifted = x - np.max(x)   # stability trick
    exp_x     = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

print("\n--- FIRST PRINCIPLES: Building softmax step by step ---")
demo_logits = np.array([2.5, -1.0, 0.8])
print(f"  Raw logits: {demo_logits}")
demo_exp = np.exp(demo_logits)
print(f"  Step 1 - exp(logits): {np.round(demo_exp, 3)}  (all positive!)")
demo_sum = np.sum(demo_exp)
print(f"  Step 2 - sum: {demo_sum:.3f}")
demo_probs = demo_exp / demo_sum
print(f"  Step 3 - normalize: {np.round(demo_probs, 4)}  (sums to {np.sum(demo_probs):.4f})")

# Example: a model classifying into 3 categories (cat, dog, bird)
logits = np.array([2.5, -1.0, 0.8])
probs  = softmax(logits)

print("\nExample: image classifier (cat, dog, bird)")
print(f"Raw logits (model output): {logits}")
print(f"Softmax probabilities:     {np.round(probs, 4)}")
print(f"Sum of probabilities:      {np.sum(probs):.4f}  (must be 1.0)")
print(f"\nInterpretation:")
print(f"  P(cat)  = {probs[0]:.2%}")
print(f"  P(dog)  = {probs[1]:.2%}")
print(f"  P(bird) = {probs[2]:.2%}")
print(f"  -> Model thinks it's most likely a cat")

# Show how softmax responds to different logit values
print("\n--- Softmax with Different Logit Patterns ---")

cases = [
    ("Uniform (no preference)",     np.array([1.0, 1.0, 1.0])),
    ("One clearly dominant",        np.array([10.0, 1.0, 1.0])),
    ("Very confident",              np.array([100.0, 0.0, 0.0])),
    ("Close competition",           np.array([2.0, 1.8, 1.5])),
    ("All negative",                np.array([-1.0, -2.0, -3.0])),
]

print(f"\n{'Case':>30} | {'Logits':>20} | {'Probabilities'}")
print("-" * 80)
for label, logit_vec in cases:
    prob_vec = softmax(logit_vec)
    logits_str = str(logit_vec)
    probs_str  = str(np.round(prob_vec, 3))
    print(f"{label:>30} | {logits_str:>20} | {probs_str}")

print("""
Key observations:
  - When all logits are equal: uniform probabilities (1/n each)
  - One very large logit: almost all probability goes there
  - Softmax always outputs valid probabilities summing to 1
  - The relative ordering is preserved (higher logit = higher probability)
""")

print("=" * 60)
print("PART 4: CROSS-ENTROPY LOSS")
print("=" * 60)

# ------------------------------------------------------------------
# We have probabilities from softmax. Now we need to measure HOW WRONG
# the probabilities are compared to the true label.
#
# Cross-Entropy Loss formula:
#   CE = -sum( true_label * log(predicted_probability) )
#
# For ONE-HOT true labels (only one class is 1, rest are 0):
#   CE = -log( predicted_probability_of_correct_class )
#
# Intuition:
#   - If we predicted 99% for the correct class: CE = -log(0.99) ≈ 0.01 (great!)
#   - If we predicted 50% for the correct class: CE = -log(0.5)  ≈ 0.69 (ok)
#   - If we predicted 1%  for the correct class: CE = -log(0.01) ≈ 4.6  (terrible!)
#
# Goal: MINIMIZE cross-entropy loss (same as maximizing the probability
#       we assign to the correct class).
# ------------------------------------------------------------------

# ===========================================================================
# FIRST PRINCIPLES: Deriving cross-entropy from maximum likelihood
# ===========================================================================
# GOAL: We want the model to assign HIGH probability to the correct class.
#
# MAXIMUM LIKELIHOOD PRINCIPLE:
#   Given data, find model parameters that make the data most likely.
#   If the model predicts P(class=k) = p_k for each class k, and the
#   true label is one-hot y = [y_0, y_1, ..., y_K], then:
#
#   Likelihood = PRODUCT over all classes: p_k^(y_k)
#   (Only the correct class contributes, since y_k = 0 for wrong classes)
#
#   For a batch of N samples:
#   Likelihood = PRODUCT over samples: p_{correct_class_for_sample_i}
#
# WHY USE LOG?
#   1. Products become sums: log(a * b) = log(a) + log(b)
#      Sums are numerically stable (products of small numbers -> underflow)
#   2. Easier derivatives: d/dx log(f(x)) = f'(x)/f(x) (simple!)
#   3. Monotonic: maximizing log(L) is same as maximizing L
#
# Log-likelihood = SUM over classes: y_k * log(p_k)
#   (For one-hot y, this simplifies to log(p_correct))
#
# We want to MAXIMIZE log-likelihood, which is the same as
# MINIMIZING negative log-likelihood:
#
#   Loss = -SUM y_k * log(p_k)
#
# This IS cross-entropy! So cross-entropy loss = negative log-likelihood.
# ===========================================================================

print("\n--- FIRST PRINCIPLES: Cross-entropy from maximum likelihood ---")
print("  Likelihood = product of P(correct class) for each sample")
print("  Log-likelihood = SUM log(P(correct class))")
print("  Loss = -log-likelihood = -SUM y_k * log(p_k) = cross-entropy")
print("  Why log? Products -> sums (numerical stability + easier math)")

def cross_entropy_loss(predicted_probs, true_label_index):
    """
    Compute cross-entropy loss.
    predicted_probs   : numpy array of probabilities (from softmax), sum to 1
    true_label_index  : integer index of the correct class
    """
    # Add tiny epsilon to avoid log(0) = -infinity
    epsilon = 1e-12
    correct_prob = predicted_probs[true_label_index]
    return -np.log(correct_prob + epsilon)

# Demonstrate how loss changes with prediction quality
print("\n--- Cross-Entropy Loss Examples ---")
true_class = 0   # correct answer is class 0

print(f"True class: {true_class}  (cat in our cat/dog/bird example)")
print(f"\n{'Predicted P(cat)':>18} | {'P(dog)':>8} | {'P(bird)':>8} | {'CE Loss':>10} | Judgment")
print("-" * 65)

scenarios = [
    ("Very confident CORRECT",  np.array([0.99, 0.005, 0.005])),
    ("Moderately confident",    np.array([0.80, 0.10,  0.10])),
    ("Completely uncertain",    np.array([0.33, 0.33,  0.34])),
    ("Slightly wrong",          np.array([0.20, 0.50,  0.30])),
    ("Very confident WRONG",    np.array([0.01, 0.98,  0.01])),
]

for label, probs_arr in scenarios:
    ce = cross_entropy_loss(probs_arr, true_class)
    print(f"{probs_arr[0]:>18.3f} | {probs_arr[1]:>8.3f} | {probs_arr[2]:>8.3f} | {ce:>10.4f} | {label}")

print("""
Pattern:
  - High probability for correct class -> LOW loss (good)
  - Low probability for correct class  -> HIGH loss (bad)
  - Perfect prediction (prob=1)        -> loss = 0
  - Completely wrong (prob=0)          -> loss = infinity
""")

# ------------------------------------------------------------------
# Cross-entropy with one-hot encoded labels (common in practice)
# ------------------------------------------------------------------

def cross_entropy_onehot(predicted_probs, true_onehot):
    """
    Cross-entropy using one-hot true labels.
    predicted_probs : array of probabilities
    true_onehot     : one-hot encoded true label (e.g., [0, 1, 0])
    """
    epsilon = 1e-12
    return -np.sum(true_onehot * np.log(predicted_probs + epsilon))

# Example
predicted = np.array([0.7, 0.2, 0.1])
true_cat  = np.array([1, 0, 0])   # true class is cat (index 0)
true_dog  = np.array([0, 1, 0])   # true class is dog (index 1)
true_bird = np.array([0, 0, 1])   # true class is bird (index 2)

print("--- One-hot Cross-Entropy ---")
print(f"Predicted probabilities: {predicted}")
print(f"  (70% cat, 20% dog, 10% bird)")
print(f"\nLoss if true label = cat:  {cross_entropy_onehot(predicted, true_cat):.4f}")
print(f"Loss if true label = dog:  {cross_entropy_onehot(predicted, true_dog):.4f}")
print(f"Loss if true label = bird: {cross_entropy_onehot(predicted, true_bird):.4f}")

print("\n" + "=" * 60)
print("PART 5: BATCH CROSS-ENTROPY LOSS")
print("=" * 60)

# ------------------------------------------------------------------
# In practice, we compute the average cross-entropy loss over a
# BATCH of examples (many data points at once).
# ------------------------------------------------------------------

def batch_cross_entropy(predicted_probs_batch, true_labels_batch):
    """
    Average cross-entropy loss over a batch.
    predicted_probs_batch : shape (n_samples, n_classes)
    true_labels_batch     : shape (n_samples,) integer class indices
    """
    n = len(true_labels_batch)
    epsilon = 1e-12
    # For each sample, get the probability assigned to the correct class
    correct_probs = predicted_probs_batch[np.arange(n), true_labels_batch]
    return -np.mean(np.log(correct_probs + epsilon))

# Simulate a batch of 5 predictions for a 3-class problem
np.random.seed(7)
raw_logits_batch = np.random.randn(5, 3)   # 5 samples, 3 classes

# Apply softmax to each row
probs_batch = np.array([softmax(row) for row in raw_logits_batch])
true_labels = np.array([0, 1, 2, 1, 0])   # true class for each sample

print("\nBatch of 5 predictions (3 classes: cat, dog, bird):")
print(f"{'Sample':>8} | {'P(cat)':>8} | {'P(dog)':>8} | {'P(bird)':>8} | {'True Label':>12} | {'Sample Loss':>12}")
print("-" * 70)

for i in range(5):
    p = probs_batch[i]
    true = true_labels[i]
    class_names = ['cat', 'dog', 'bird']
    ce = cross_entropy_loss(p, true)
    print(f"{i:>8} | {p[0]:>8.4f} | {p[1]:>8.4f} | {p[2]:>8.4f} | {class_names[true]:>12} | {ce:>12.4f}")

batch_loss = batch_cross_entropy(probs_batch, true_labels)
print(f"\nAverage batch loss: {batch_loss:.4f}")

print("\n" + "=" * 60)
print("PART 6: FULL PIPELINE — LOGITS -> SOFTMAX -> LOSS")
print("=" * 60)

# ------------------------------------------------------------------
# In a neural network classifier, the training process is:
# 1. Forward pass: compute logits (raw output of last layer)
# 2. Apply softmax to get probabilities
# 3. Compute cross-entropy loss against true labels
# 4. Backpropagate the loss to update weights
#
# The combined gradient of softmax + cross-entropy has a beautiful
# simple form: predicted_prob - true_one_hot
# (We won't derive it, but it's elegant!)
# ------------------------------------------------------------------

print("\nFull pipeline example:\n")

# Simulate one forward pass
logits_example = np.array([1.2, 0.5, -0.8])   # raw model output (3 classes)
true_label     = 1                              # correct class is class 1 (dog)
class_names    = ['cat', 'dog', 'bird']

probs_example = softmax(logits_example)
ce_loss       = cross_entropy_loss(probs_example, true_label)

print(f"Step 1 — Model output (logits):  {logits_example}")
print(f"Step 2 — Softmax probabilities:  {np.round(probs_example, 4)}")
print(f"Step 3 — True label:             {true_label} ({class_names[true_label]})")
print(f"Step 4 — Cross-entropy loss:     {ce_loss:.4f}")
print(f"\nInterpretation: Model is {probs_example[true_label]:.1%} confident in the correct answer.")
print(f"We want to MINIMIZE this loss through gradient descent.")

# Show how the gradient simplifies beautifully
true_onehot = np.zeros(3)
true_onehot[true_label] = 1
gradient = probs_example - true_onehot

print(f"\nGradient (d_loss / d_logits) = probs - true_onehot:")
print(f"  probs      = {np.round(probs_example, 4)}")
print(f"  true_onehot= {true_onehot}")
print(f"  gradient   = {np.round(gradient, 4)}")
print(f"\nThe gradient tells us: push logits UP for the correct class,")
print(f"push them DOWN for the wrong classes. Simple!")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key ideas from this file:
  - Probability: a number in [0,1] measuring likelihood
  - Probabilities of all outcomes must sum to 1
  - Uniform distribution: all outcomes equally likely
  - Normal distribution: bell curve, most values near the mean
  - Softmax: converts any numbers to valid probabilities (sum=1)
    formula: exp(x_i) / sum(exp(x_j))
  - Cross-entropy: -log(P(correct class)) — low is good, 0 is perfect
  - In classification: model outputs logits -> softmax -> probabilities
    -> cross-entropy loss -> gradient descent -> update weights
  - These two functions (softmax + cross-entropy) are used in EVERY
    classification model: image classifiers, text classifiers, LLMs!
""")

# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test softmax: output must sum to 1
test_logits_1 = np.array([2.5, -1.0, 0.8])
sm1 = softmax(test_logits_1)
assert abs(sm1.sum() - 1.0) < 1e-6, "Softmax must sum to 1"
assert np.all(sm1 >= 0), "All softmax outputs must be non-negative"
assert np.all(sm1 <= 1), "All softmax outputs must be <= 1"

# Largest logit should get highest probability
assert sm1[0] > sm1[1] and sm1[0] > sm1[2], "Largest logit should have highest probability"

# Uniform logits -> equal probabilities
sm_uniform = softmax(np.array([1.0, 1.0, 1.0]))
assert abs(sm_uniform.sum() - 1.0) < 1e-6, "Softmax of uniform logits must sum to 1"
assert abs(sm_uniform[0] - 1/3) < 1e-6, "Softmax of uniform logits should give 1/3 each"

# Large dominant logit -> high probability
sm_dominant = softmax(np.array([100.0, 0.0, 0.0]))
assert abs(sm_dominant.sum() - 1.0) < 1e-6, "Softmax with dominant logit must sum to 1"
assert sm_dominant[0] > 0.99, "Very dominant logit should get >99% probability"

# Test cross-entropy loss
# Perfect prediction: P(correct) = ~1.0 -> loss ≈ 0
perfect_probs = np.array([0.9999, 0.0001, 0.0])
ce_perfect = cross_entropy_loss(perfect_probs, 0)
assert ce_perfect < 0.01, f"Near-perfect prediction should have small CE loss, got {ce_perfect:.4f}"

# Terrible prediction: P(correct) = ~0.01 -> loss is large
bad_probs = np.array([0.01, 0.98, 0.01])
ce_bad = cross_entropy_loss(bad_probs, 0)
assert ce_bad > 4.0, f"Very wrong prediction should have large CE loss, got {ce_bad:.4f}"

# CE loss should be larger when the model is more wrong
ce_moderate = cross_entropy_loss(np.array([0.5, 0.3, 0.2]), 0)
assert ce_perfect < ce_moderate < ce_bad, "CE loss should grow as predictions get worse"

# Test one-hot cross-entropy matches index-based CE
predicted_test = np.array([0.7, 0.2, 0.1])
true_cat_vec = np.array([1, 0, 0])
ce_from_index = cross_entropy_loss(predicted_test, 0)
ce_from_onehot = cross_entropy_onehot(predicted_test, true_cat_vec)
assert abs(ce_from_index - ce_from_onehot) < 1e-9, "Index-based and one-hot CE should match"

# Test cross-entropy values: -log(0.7) ≈ 0.3567, -log(0.2) ≈ 1.609, -log(0.1) ≈ 2.303
assert abs(cross_entropy_loss(predicted_test, 0) - (-np.log(0.7))) < 0.001, "CE for cat wrong"
assert abs(cross_entropy_loss(predicted_test, 1) - (-np.log(0.2))) < 0.001, "CE for dog wrong"
assert abs(cross_entropy_loss(predicted_test, 2) - (-np.log(0.1))) < 0.001, "CE for bird wrong"

# Test that probabilities sum to 1 in batch
for row in probs_batch:
    assert abs(row.sum() - 1.0) < 1e-6, "Each row in probs_batch must sum to 1"

# Test the full pipeline softmax probabilities
probs_example_test = softmax(logits_example)
assert abs(probs_example_test.sum() - 1.0) < 1e-6, "Pipeline softmax must sum to 1"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
# 1. Simulate 50,000 rolls of TWO dice. What is the probability that
#    the sum equals 7? (Should be about 1/6 = 0.167)
#    Hint: roll two dice arrays, add them, count times sum == 7.
#
# 2. Generate 10,000 samples from Normal(mean=65, std=15) (representing
#    heights in cm). What fraction are taller than 180cm? Shorter than 50cm?
#
# 3. Apply softmax to [1, 2, 3, 4, 5]. What happens as the numbers
#    get larger? Now try [10, 20, 30, 40, 50]. Does the relative
#    ranking change? Does the confidence change?
#
# 4. A model predicts [0.1, 0.6, 0.3] for classes [cat, dog, bird].
#    Compute the cross-entropy loss when:
#      a) The true label is dog (class 1)
#      b) The true label is cat (class 0)
#    Which is higher? Why?
#
# 5. The cross-entropy of a perfect model (probability 1.0 for correct class)
#    should be 0. Verify this: compute -log(1.0). What does -log(0.5) equal?
#    What about -log(0.01)?
#
# 6. FIRST PRINCIPLES EXERCISE: Show that cross-entropy is minimized when
#    the predicted distribution equals the true distribution.
#    Hint: For a true distribution q and predicted distribution p,
#    cross-entropy H(q, p) = -SUM q_k * log(p_k).
#    Using Lagrange multipliers or Jensen's inequality, you can show that
#    H(q, p) >= H(q, q) (entropy of q), with equality iff p = q.
#    The difference H(q, p) - H(q, q) is called KL divergence, and it's
#    always >= 0. So the best a model can do is match the true distribution.
#    Verify numerically: compute H(q, p) for q = [0.7, 0.2, 0.1] and
#    various p values. Show that H(q, q) gives the lowest value.
# =============================================================================
