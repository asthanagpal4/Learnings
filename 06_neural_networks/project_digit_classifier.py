# HOW TO RUN:
#   uv run python 06_neural_networks/project_digit_classifier.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# MINI-PROJECT: HANDWRITTEN DIGIT CLASSIFIER FROM SCRATCH
# =============================================================================
# Run this file with:  uv run python 06_neural_networks/project_digit_classifier.py
#
# WHAT WE'RE BUILDING:
# ---------------------
# A neural network that recognizes small digit-like patterns.
# We create a tiny synthetic dataset of "digits" 0-4 using 3x5 pixel grids.
# Each digit is a binary image (pixels are 0=off or 1=on).
# The network learns to look at the pixel pattern and output which digit it is.
#
# This is the SAME idea as real handwriting recognition (MNIST dataset),
# just much smaller. Real MNIST uses 28x28 images and 10 digit classes.
#
# WHAT THIS BRINGS TOGETHER:
#   - From Lesson 2: multi-layer network structure, forward pass
#   - From Lesson 3: backpropagation, chain rule
#   - From Lesson 4: training loop, batching, learning rate
#   - NEW: one-hot encoding, softmax output, cross-entropy loss
#
# ARCHITECTURE:
#   Input:   15 pixels (3x5 image, flattened to a vector)
#   Hidden1: 32 neurons
#   Hidden2: 16 neurons
#   Output:  5 neurons (one per digit 0-4)
#
# The output uses SOFTMAX — converts raw scores into probabilities that sum to 1.
# "I'm 80% sure this is a 3, 10% sure it's a 2, ..."

import numpy as np

np.random.seed(42)

print("=" * 60)
print("MINI-PROJECT: DIGIT CLASSIFIER FROM SCRATCH")
print("=" * 60)


# =============================================================================
# PART 1: THE DIGIT TEMPLATES (3x5 pixel grids)
# =============================================================================
# We design simple pixel patterns for digits 0-4.
# Each digit is a 3x5 grid (3 columns wide, 5 rows tall = 15 pixels).
# 1 = pixel is ON (dark), 0 = pixel is OFF (light).
#
# Reading each template row by row, top to bottom:
#
# Digit 0:    Digit 1:    Digit 2:    Digit 3:    Digit 4:
# # # #       . # .       # # #       # # #       # . #
# # . #       # # .       . . #       . . #       # . #
# # . #       . # .       . # .       . # #       # # #
# # . #       . # .       # . .       . . #       . . #
# # # #       # # #       # # #       # # #       . . #

DIGIT_TEMPLATES = {
    0: np.array([
        1, 1, 1,
        1, 0, 1,
        1, 0, 1,
        1, 0, 1,
        1, 1, 1,
    ], dtype=float),

    1: np.array([
        0, 1, 0,
        1, 1, 0,
        0, 1, 0,
        0, 1, 0,
        1, 1, 1,
    ], dtype=float),

    2: np.array([
        1, 1, 1,
        0, 0, 1,
        0, 1, 1,
        1, 0, 0,
        1, 1, 1,
    ], dtype=float),

    3: np.array([
        1, 1, 1,
        0, 0, 1,
        0, 1, 1,
        0, 0, 1,
        1, 1, 1,
    ], dtype=float),

    4: np.array([
        1, 0, 1,
        1, 0, 1,
        1, 1, 1,
        0, 0, 1,
        0, 0, 1,
    ], dtype=float),
}

N_DIGITS = len(DIGIT_TEMPLATES)   # 5
N_PIXELS = 15                      # 3 * 5


def print_digit(pixel_vector, label=None):
    """Print a digit template as a visual grid."""
    grid = pixel_vector[:15].reshape(5, 3)
    chars = {0.0: ".", 1.0: "#"}
    if label is not None:
        print(f"  Digit {label}:")
    for row in grid:
        print("    " + " ".join(chars.get(v, "?") for v in row))
    print()


print("\n--- Part 1: Our Digit Templates (3x5 pixels) ---")
for digit_id, template in DIGIT_TEMPLATES.items():
    print_digit(template, label=digit_id)


# =============================================================================
# PART 2: CREATE THE DATASET WITH DATA AUGMENTATION
# =============================================================================
# A single template per digit is not enough to train a network.
# We need VARIATION so the network learns the pattern, not just one exact image.
#
# DATA AUGMENTATION: create many slightly different versions of each digit
#   - Add random noise (flip a few pixels randomly)
#   - This makes the network robust to imperfect handwriting
#
# We'll generate 80 samples per digit = 400 total samples.

def augment_digit(template, noise_level=0.15):
    """
    Create a noisy version of a digit template.
    Each pixel has noise_level probability of being flipped.
    """
    augmented = template.copy()
    # Random noise: with probability noise_level, flip a pixel
    flip_mask = np.random.rand(len(template)) < noise_level
    augmented[flip_mask] = 1.0 - augmented[flip_mask]  # flip: 0->1 or 1->0
    return augmented


def create_dataset(samples_per_digit=80, noise_level=0.10):
    """
    Create training dataset with augmented digit images.
    Returns X (pixels), y_labels (class indices), y_onehot (one-hot encoded).
    """
    all_X = []
    all_y = []

    for digit_id, template in DIGIT_TEMPLATES.items():
        for _ in range(samples_per_digit):
            # Create a noisy version of the digit
            sample = augment_digit(template, noise_level)
            all_X.append(sample)
            all_y.append(digit_id)

    X = np.array(all_X)              # shape: (n_samples, 15)
    y_labels = np.array(all_y)       # shape: (n_samples,) — class index 0-4

    # ONE-HOT ENCODING: convert class index to vector
    # class 0 -> [1,0,0,0,0]
    # class 1 -> [0,1,0,0,0]
    # class 2 -> [0,0,1,0,0]  etc.
    # Why? The output layer has 5 neurons, one per digit.
    # We want neuron 3 to fire for digit "3", not neuron 0.
    y_onehot = np.zeros((len(y_labels), N_DIGITS))
    y_onehot[np.arange(len(y_labels)), y_labels] = 1.0

    # Shuffle the dataset
    idx = np.random.permutation(len(X))
    return X[idx], y_labels[idx], y_onehot[idx]


print("--- Part 2: Creating the Dataset ---")
X_train, y_labels, y_onehot = create_dataset(samples_per_digit=100, noise_level=0.10)
print(f"  X_train shape:  {X_train.shape}  (500 samples, 15 pixels each)")
print(f"  y_labels shape: {y_labels.shape}  (class indices 0-4)")
print(f"  y_onehot shape: {y_onehot.shape}  (one-hot encoded)")
print()
print("  One-hot encoding examples:")
for digit_id in range(5):
    idx = np.where(y_labels == digit_id)[0][0]
    print(f"    Digit {digit_id}: y_labels={y_labels[idx]}  y_onehot={y_onehot[idx].astype(int)}")
print()
print("  Samples per class:")
for digit_id in range(5):
    count = int(np.sum(y_labels == digit_id))
    print(f"    Digit {digit_id}: {count} samples")

# Show one augmented sample vs the clean template
print()
print("  Original digit 3 vs augmented version (noise_level=0.10):")
print("  --- Original ---")
print_digit(DIGIT_TEMPLATES[3], label=3)
noisy_sample = augment_digit(DIGIT_TEMPLATES[3], noise_level=0.25)
print("  --- Noisy version (25% noise, for visual clarity) ---")
print_digit(noisy_sample, label="3 noisy")


# =============================================================================
# PART 3: ACTIVATION FUNCTIONS FOR MULTI-CLASS CLASSIFICATION
# =============================================================================
# For BINARY classification (0 or 1), sigmoid output works fine.
# For MULTI-CLASS (digit 0, 1, 2, 3, or 4), we use SOFTMAX.
#
# SOFTMAX converts a vector of raw scores to PROBABILITIES:
#   softmax(z)[i] = exp(z[i]) / sum(exp(z[j]) for all j)
#
# Properties:
#   - All outputs are positive (exp is always positive)
#   - All outputs sum to exactly 1.0 (they're probabilities!)
#   - The largest input gets the largest probability
#   - Differences are amplified: [1,2,3] -> [0.09, 0.24, 0.67]
#
# CROSS-ENTROPY LOSS works with softmax:
#   loss = -sum( y_true * log(y_pred) )
#   This penalizes heavily when the network is CONFIDENT but WRONG.

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(a):
    return a * (1.0 - a)

def softmax(z):
    """
    Convert raw scores to probabilities.
    Subtracting max(z) first prevents numerical overflow (but doesn't change result).
    """
    # Subtract max for numerical stability (exp of large numbers overflows)
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy loss for multi-class classification.
    y_pred: softmax probabilities, shape (m, n_classes)
    y_true: one-hot labels,        shape (m, n_classes)
    """
    # Clip to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-10, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))


print("--- Part 3: Softmax Demonstration ---")
print("  Raw network output (before softmax): [2.0, 0.5, -1.0, 3.5, 1.0]")
raw_scores = np.array([[2.0, 0.5, -1.0, 3.5, 1.0]])
probs = softmax(raw_scores)
print(f"  After softmax (probabilities):       {probs[0].round(4)}")
print(f"  Sum of probabilities: {probs[0].sum():.6f}  (always 1.0)")
print(f"  Predicted digit: {np.argmax(probs[0])}  (highest probability)")
print()
print("  Note: digit 3 has highest raw score (3.5) and gets most probability ({:.1f}%)".format(
    probs[0][3] * 100))


# =============================================================================
# PART 4: THE 3-LAYER DIGIT CLASSIFIER NETWORK
# =============================================================================

class DigitClassifier:
    """
    3-layer MLP for digit classification.
    Architecture: input(15) -> hidden1(32) -> hidden2(16) -> output(5)
    Hidden layers use sigmoid. Output layer uses softmax.
    """

    def __init__(self, n_input=15, n_h1=32, n_h2=16, n_output=5):
        # He initialization for all weight matrices
        self.W1 = np.random.randn(n_input, n_h1)  * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros(n_h1)
        self.W2 = np.random.randn(n_h1, n_h2)     * np.sqrt(2.0 / n_h1)
        self.b2 = np.zeros(n_h2)
        self.W3 = np.random.randn(n_h2, n_output) * np.sqrt(2.0 / n_h2)
        self.b3 = np.zeros(n_output)

        n_params = (self.W1.size + self.b1.size + self.W2.size +
                    self.b2.size + self.W3.size + self.b3.size)
        print(f"  DigitClassifier initialized. Total parameters: {n_params}")

    def forward(self, x):
        """Forward pass. Stores intermediates for backprop."""
        self.x = x

        # Hidden layer 1
        self.z1 = x @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        # Hidden layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        # Output layer (softmax for multi-class)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = softmax(self.z3)      # probabilities for each digit

        return self.a3

    def backward(self, y_true, learning_rate):
        """
        Backward pass for softmax + cross-entropy.

        Special property: when using softmax + cross-entropy together,
        the gradient at the output simplifies beautifully to just:
            delta3 = (predicted - true)
        This is one reason softmax + cross-entropy is so popular!
        (The ugly chain rule terms cancel out.)
        """
        m = self.x.shape[0]

        # Output layer gradient (softmax + cross-entropy simplified)
        delta3 = (self.a3 - y_true) / m     # shape: (m, n_output)

        dW3 = self.a2.T @ delta3
        db3 = np.sum(delta3, axis=0)

        # Hidden layer 2 gradient
        delta2 = (delta3 @ self.W3.T) * sigmoid_deriv(self.a2)

        dW2 = self.a1.T @ delta2
        db2 = np.sum(delta2, axis=0)

        # Hidden layer 1 gradient
        delta1 = (delta2 @ self.W2.T) * sigmoid_deriv(self.a1)

        dW1 = self.x.T @ delta1
        db1 = np.sum(delta1, axis=0)

        # Update all weights
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def predict(self, x):
        """Returns the predicted digit class (0-4) for each sample."""
        probs = self.forward(x)
        return np.argmax(probs, axis=1)   # index of highest probability

    def accuracy(self, x, y_true_labels):
        """Compute classification accuracy."""
        predictions = self.predict(x)
        return np.mean(predictions == y_true_labels) * 100


# =============================================================================
# PART 5: TRAINING THE DIGIT CLASSIFIER
# =============================================================================

def train_digit_classifier(model, X, y_labels, y_onehot,
                            learning_rate=0.5, epochs=3000, print_every=500):
    """
    Full training loop for the digit classifier.
    Uses cross-entropy loss and full-batch gradient descent.
    """
    loss_history = []
    acc_history  = []

    print(f"\n  Training: lr={learning_rate}, epochs={epochs}")
    print(f"  {'Epoch':>7} | {'Loss':>8} | {'Accuracy':>10}")
    print("  " + "-" * 32)

    for epoch in range(epochs):
        # Forward pass
        y_pred = model.forward(X)

        # Loss
        loss = cross_entropy_loss(y_pred, y_onehot)
        loss_history.append(loss)

        # Backward + update
        model.backward(y_onehot, learning_rate)

        # Accuracy
        acc = model.accuracy(X, y_labels)
        acc_history.append(acc)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"  {epoch+1:>7} | {loss:>8.4f} | {acc:>9.1f}%")

    return loss_history, acc_history


# =============================================================================
# FIRST PRINCIPLES: PARAMETER COUNT, FLOPS, AND MEMORY ANALYSIS
# =============================================================================
# Architecture: input(15) -> hidden1(32) -> hidden2(16) -> output(5)
#
# PARAMETER COUNT (weights + biases per layer):
#   Layer 1 (15 -> 32):  15 * 32 weights + 32 biases = 480 + 32 = 512
#   Layer 2 (32 -> 16):  32 * 16 weights + 16 biases = 512 + 16 = 528
#   Layer 3 (16 -> 5):   16 * 5  weights + 5  biases = 80  + 5  = 85
#   -----------------------------------------------------------------
#   Total: 512 + 528 + 85 = 1,125 parameters
#
# FLOPS ANALYSIS (one forward pass, one sample):
#   A matrix multiply of (1, n_in) @ (n_in, n_out) does n_in * n_out multiply-adds.
#   Layer 1: 15 * 32 = 480 multiply-adds  + 32 bias adds + 32 sigmoid evals
#   Layer 2: 32 * 16 = 512 multiply-adds  + 16 bias adds + 16 sigmoid evals
#   Layer 3: 16 * 5  = 80  multiply-adds  + 5  bias adds + 5  softmax ops
#   Total multiply-adds: 480 + 512 + 80 = 1,072
#   (Bias adds and activations are comparatively cheap)
#
# MEMORY ANALYSIS:
#   Parameters: 1,125 values * 8 bytes (float64) = 9,000 bytes = ~8.8 KB
#   Activations stored during forward pass (for backprop, per sample):
#     a1: 32 values, a2: 16 values, a3: 5 values = 53 values
#   For a batch of 500 samples: 500 * 53 * 8 = 212,000 bytes = ~207 KB
#   Total memory (params + batch activations): ~216 KB
#
# EXERCISE: If you doubled every hidden layer (15 -> 64 -> 32 -> 5):
#   Layer 1: 15 * 64 + 64 = 1,024   (was 512, ~2x)
#   Layer 2: 64 * 32 + 32 = 2,080   (was 528, ~4x!)
#   Layer 3: 32 * 5  + 5  = 165     (was 85, ~2x)
#   Total: 3,269 parameters (was 1,125, ~2.9x)
#   FLOPs: 15*64 + 64*32 + 32*5 = 960 + 2048 + 160 = 3,168 (was 1,072, ~3x)
#   Key insight: doubling hidden layers roughly TRIPLES params and FLOPs,
#   because the MIDDLE layer pair (h1*h2) grows as the SQUARE (both sides double).
# =============================================================================

print("\n--- First Principles: Architecture Analysis ---")
arch = [15, 32, 16, 5]
total_w = sum(arch[i] * arch[i+1] for i in range(len(arch)-1))
total_b = sum(arch[i+1] for i in range(len(arch)-1))
total_p = total_w + total_b
total_flops = sum(arch[i] * arch[i+1] for i in range(len(arch)-1))
print(f"  Architecture: {arch}")
print(f"  Weights: {total_w}, Biases: {total_b}, Total parameters: {total_p}")
print(f"  Forward pass FLOPs (multiply-adds): {total_flops}")
print(f"  Parameter memory: {total_p * 8 / 1024:.1f} KB (float64)")
print()

# Show what happens if hidden layers are doubled
arch_double = [15, 64, 32, 5]
total_p2 = sum(arch_double[i]*arch_double[i+1] + arch_double[i+1] for i in range(len(arch_double)-1))
total_f2 = sum(arch_double[i]*arch_double[i+1] for i in range(len(arch_double)-1))
print(f"  If hidden layers doubled: {arch_double}")
print(f"  Total parameters: {total_p2} ({total_p2/total_p:.1f}x original)")
print(f"  Forward FLOPs: {total_f2} ({total_f2/total_flops:.1f}x original)")
print()

print("\n--- Part 4 & 5: Building and Training the Classifier ---")
classifier = DigitClassifier(n_input=15, n_h1=32, n_h2=16, n_output=5)

loss_hist, acc_hist = train_digit_classifier(
    model=classifier,
    X=X_train,
    y_labels=y_labels,
    y_onehot=y_onehot,
    learning_rate=0.3,
    epochs=3000,
    print_every=500
)


# =============================================================================
# PART 6: EVALUATION AND RESULTS
# =============================================================================
print("\n--- Part 6: Evaluation ---")
print()

# Overall accuracy
final_acc = classifier.accuracy(X_train, y_labels)
print(f"  Training Accuracy: {final_acc:.1f}%")
print()

# Per-digit accuracy
print("  Per-digit accuracy:")
for digit_id in range(5):
    mask = (y_labels == digit_id)
    digit_acc = classifier.accuracy(X_train[mask], y_labels[mask])
    bar = "#" * int(digit_acc // 5)
    print(f"    Digit {digit_id}: {digit_acc:5.1f}%  {bar}")
print()

# Confusion matrix
all_preds = classifier.predict(X_train)
print("  Confusion Matrix (rows=actual, cols=predicted):")
print("           Pred 0  Pred 1  Pred 2  Pred 3  Pred 4")
for actual in range(5):
    row = []
    for predicted in range(5):
        count = int(np.sum((y_labels == actual) & (all_preds == predicted)))
        row.append(count)
    row_str = "  ".join(f"{v:5d}" for v in row)
    print(f"  Actual {actual}:  {row_str}")
print()
print("  (Diagonal = correct predictions, off-diagonal = mistakes)")


# =============================================================================
# PART 7: TEST ON CLEAN TEMPLATES (THE PERFECT DIGITS)
# =============================================================================
print("\n--- Part 7: Testing on the Perfect Clean Templates ---")
print("  Can the network recognize the exact template it was trained on?")
print()

clean_X = np.array([DIGIT_TEMPLATES[d] for d in range(5)])
clean_preds = classifier.predict(clean_X)
clean_probs = classifier.forward(clean_X)

print(f"  {'Digit':>7} | {'Pred':>6} | {'Confidence':>11} | {'Correct?':>9}")
print("  " + "-" * 45)
for d in range(5):
    pred = clean_preds[d]
    conf = clean_probs[d, pred] * 100
    ok = "YES" if pred == d else " NO"
    print(f"    {d:>5}  |   {pred:>3}  | {conf:>9.1f}%  |    {ok}")

print()
print("  Full probability distributions for clean templates:")
print(f"  {'Digit':>7} | {'P(0)':>6} {'P(1)':>6} {'P(2)':>6} {'P(3)':>6} {'P(4)':>6}")
print("  " + "-" * 42)
for d in range(5):
    probs_row = " ".join(f"{clean_probs[d,c]*100:5.1f}%" for c in range(5))
    print(f"    {d:>5}  | {probs_row}")


# =============================================================================
# PART 8: PREDICTIONS ON NEW NOISY SAMPLES
# =============================================================================
print("\n--- Part 8: Predictions on New Noisy Samples ---")
print("  Generating 2 fresh noisy samples per digit and predicting...")
print()

np.random.seed(99)  # Different seed = different noise than training data
for d in range(5):
    print(f"  --- Testing digit {d} ---")
    for trial in range(2):
        noisy = augment_digit(DIGIT_TEMPLATES[d], noise_level=0.15)
        noisy_batch = noisy.reshape(1, -1)
        probs_out = classifier.forward(noisy_batch)
        pred = np.argmax(probs_out)
        conf = probs_out[0, pred] * 100
        ok = "CORRECT" if pred == d else f"WRONG (said {pred})"
        print(f"    Sample {trial+1}: predicted={pred} ({conf:.1f}% confident) -> {ok}")
    print()


# =============================================================================
# PART 9: WHAT THE NETWORK LEARNED
# =============================================================================
print("\n--- Part 9: What Did the Network Learn? ---")
print("""
  Each hidden neuron learns to be a "feature detector":
  - Some hidden neurons might fire for "top horizontal bar"
  - Others might fire for "vertical line on the right"
  - The output neurons combine these features to identify digits

  This is exactly how deep learning works at scale:
  - GPT (language model): hidden neurons detect grammar, meaning, context
  - ResNet (image classifier): hidden neurons detect edges, textures, objects
  - AlphaFold (protein folding): hidden neurons detect structural patterns

  The SAME backpropagation algorithm we used here trains ALL of them.
  The only differences are:
    - Scale: millions to trillions of parameters
    - Architecture: convolutions, attention, residual connections
    - Tricks: batch normalization, dropout, learning rate schedules
    - Hardware: thousands of GPUs running in parallel
""")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("=" * 60)
print("PROJECT COMPLETE! WHAT WE BUILT:")
print("=" * 60)
print(f"""
  Dataset:
    - 500 synthetic digit images (5 digits x 100 samples each)
    - 15 pixels per image (3x5 grid, flattened to vector)
    - Data augmentation: 10% random pixel noise
    - One-hot encoded labels

  Network:
    - 3 layers: 15 -> 32 -> 16 -> 5
    - Sigmoid hidden activations
    - Softmax output (probabilities for 5 digit classes)
    - Cross-entropy loss
    - Backpropagation with chain rule
    - Full-batch gradient descent

  Training:
    - {len(loss_hist)} epochs
    - Learning rate: 0.3
    - Loss: {loss_hist[0]:.4f} -> {loss_hist[-1]:.4f}

  Results:
    - Training accuracy: {final_acc:.1f}%

  This is a COMPLETE neural network implemented from scratch in pure numpy.
  You now understand the core of modern deep learning!
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- dataset shape checks ---
assert X_train.shape == (500, 15), f"X_train shape should be (500,15), got {X_train.shape}"
assert y_labels.shape == (500,),   f"y_labels shape should be (500,), got {y_labels.shape}"
assert y_onehot.shape == (500, 5), f"y_onehot shape should be (500,5), got {y_onehot.shape}"

# one-hot rows must sum to 1
assert np.all(y_onehot.sum(axis=1) == 1), "Each one-hot row must sum to 1"

# labels are in range 0-4
assert set(y_labels.tolist()).issubset({0, 1, 2, 3, 4}), "Labels must be 0-4"

# --- parameter count for DigitClassifier(15, 32, 16, 5) ---
expected_params = (15*32 + 32) + (32*16 + 16) + (16*5 + 5)  # 512 + 528 + 85 = 1125
actual_params = classifier.W1.size + classifier.b1.size + \
                classifier.W2.size + classifier.b2.size + \
                classifier.W3.size + classifier.b3.size
assert actual_params == expected_params, \
    f"Expected {expected_params} parameters, got {actual_params}"

# --- weight matrix shapes ---
assert classifier.W1.shape == (15, 32), f"W1 shape wrong: {classifier.W1.shape}"
assert classifier.W2.shape == (32, 16), f"W2 shape wrong: {classifier.W2.shape}"
assert classifier.W3.shape == (16, 5),  f"W3 shape wrong: {classifier.W3.shape}"

# --- forward pass output shape ---
out = classifier.forward(X_train[:10])
assert out.shape == (10, 5), f"Output shape should be (10,5), got {out.shape}"

# --- softmax outputs sum to 1 for each sample ---
assert np.allclose(out.sum(axis=1), 1.0), "Softmax outputs must sum to 1 per row"

# --- all softmax probabilities are positive ---
assert np.all(out > 0), "All softmax probabilities must be positive"

# --- predict returns class indices in range 0-4 ---
preds_t = classifier.predict(X_train[:20])
assert preds_t.shape == (20,), f"Predict shape should be (20,), got {preds_t.shape}"
assert np.all((preds_t >= 0) & (preds_t <= 4)), "Predictions must be in range 0-4"

# --- training reduced the loss ---
assert loss_hist[-1] < loss_hist[0], \
    f"Training should reduce loss: initial={loss_hist[0]:.4f}, final={loss_hist[-1]:.4f}"

# --- final accuracy is better than random (random = 20% for 5 classes) ---
assert final_acc > 20.0, f"Final accuracy ({final_acc:.1f}%) should exceed random chance (20%)"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("=" * 60)
print("EXERCISES — Push your understanding further!")
print("=" * 60)
print("""
Exercise 1: Add Digit 5, 6, 7 (more classes)
  Design 3x5 pixel templates for digits 5, 6, 7 by filling in
  a 5-row x 3-column grid of 1s and 0s.
  Add them to DIGIT_TEMPLATES and change N_DIGITS to 8.
  Update the output layer: DigitClassifier(n_output=8).
  Does the network still learn well with more classes?

Exercise 2: Increase Noise
  Try noise_level=0.20 and noise_level=0.35 in create_dataset().
  At what noise level does the network start struggling?
  Why does more noise make learning harder?

Exercise 3: Fewer Training Samples
  Try samples_per_digit=10 (very small dataset).
  What happens to accuracy? Does the network overfit the few samples it sees?
  Hint: with only 10 noisy samples per digit, the network might not see
  enough variation to generalize.

Exercise 4: Different Network Sizes
  Try the following architectures (change n_h1, n_h2):
    - 15 -> 8 -> 4 -> 5  (small network)
    - 15 -> 64 -> 32 -> 5 (bigger network)
  Does bigger always mean better? What is the trade-off?

Exercise 5: Print Most Confused Digits
  After training, find which digit pairs the network confuses most.
  Look at the off-diagonal elements of the confusion matrix.
  Which two digits get confused? Can you see WHY by looking at their templates?
  (Hint: digits with similar shapes will confuse the network more.)

Exercise 6: Train/Test Split (important concept!)
  Use 80% of data for training, 20% for testing.
  Train only on the 80%, evaluate on the 20% (unseen data).
  Is test accuracy lower than training accuracy? That gap is called
  "generalization error" — the key measure of how well ML models really work.
  Hint: split_idx = int(0.8 * len(X_train)); X_test = X_train[split_idx:]
""")
