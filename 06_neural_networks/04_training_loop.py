# HOW TO RUN:
#   uv run python 06_neural_networks/04_training_loop.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# LESSON 4: THE COMPLETE TRAINING LOOP
# =============================================================================
# Run this file with:  uv run python 06_neural_networks/04_training_loop.py
#
# So far we've seen:
#   Lesson 1: A single neuron (perceptron)
#   Lesson 2: Multi-layer network structure, forward pass
#   Lesson 3: Backpropagation on XOR
#
# Now we'll build a COMPLETE, CLEAN, REUSABLE training loop.
# And we'll apply it to a more interesting problem than XOR:
#   Classifying whether points lie INSIDE or OUTSIDE a circle.
#
# THE COMPLETE TRAINING RECIPE:
# --------------------------------
#   1. Create dataset
#   2. Initialize network weights
#   3. Repeat for N epochs:
#      a. Forward pass     -> get predictions
#      b. Compute loss     -> how wrong are we?
#      c. Backward pass    -> compute gradients
#      d. Update weights   -> gradient descent step
#      e. Track metrics    -> loss, accuracy
#   4. Evaluate on training data
#
# KEY NEW CONCEPTS:
#   - Batching: process multiple samples per update
#   - Learning rate and its effect on training
#   - Epochs vs iterations
#   - Tracking training metrics over time

import numpy as np

np.random.seed(42)

print("=" * 60)
print("LESSON 4: THE COMPLETE TRAINING LOOP")
print("=" * 60)


# =============================================================================
# PART 1: CREATE A DATASET — INSIDE OR OUTSIDE A CIRCLE?
# =============================================================================
# We'll create a 2D dataset:
#   - Each sample is a point (x, y) in 2D space
#   - Label = 1 if the point is inside a circle of radius 0.5
#   - Label = 0 if the point is outside the circle
#
# This is a classic classification problem that a single neuron CANNOT solve
# (why? a single neuron draws a straight line — a circle boundary is curved!)
# But a multi-layer network can approximate the curved boundary.

def make_circle_dataset(n_samples=200, noise=0.05):
    """
    Create a binary classification dataset.
    Points inside a circle (radius 0.5) get label 1.
    Points outside get label 0.

    Returns X (shape: n_samples x 2) and y (shape: n_samples x 1)
    """
    # Random points in [-1, 1] x [-1, 1]
    X = np.random.uniform(-1, 1, size=(n_samples, 2))

    # Add a tiny bit of noise to x,y before computing distance
    X_noisy = X + np.random.randn(n_samples, 2) * noise

    # Distance from origin: sqrt(x^2 + y^2)
    distances = np.sqrt(X_noisy[:, 0]**2 + X_noisy[:, 1]**2)

    # Label 1 if inside circle of radius 0.5
    y = (distances < 0.5).astype(float).reshape(-1, 1)

    return X, y


print("\n--- Part 1: Dataset ---")
X, y = make_circle_dataset(n_samples=300, noise=0.02)
print(f"  Dataset created: {X.shape[0]} samples, {X.shape[1]} features each")
print(f"  X shape: {X.shape}  (each row is [x_coordinate, y_coordinate])")
print(f"  y shape: {y.shape}  (each row is 0 or 1)")
print(f"  Class 0 (outside circle): {int(np.sum(y==0))} samples")
print(f"  Class 1 (inside circle):  {int(np.sum(y==1))} samples")
print(f"  First 5 samples:")
for i in range(5):
    dist = np.sqrt(X[i, 0]**2 + X[i, 1]**2)
    label = "INSIDE" if y[i, 0] == 1 else "outside"
    print(f"    x={X[i,0]:+.3f}, y={X[i,1]:+.3f}  dist={dist:.3f}  label={int(y[i,0])} ({label})")


# =============================================================================
# PART 2: THE NEURAL NETWORK CLASS
# =============================================================================
# Instead of loose functions, let's organize everything into a clean structure.
# We'll use a Python class to hold the weights and methods together.
# (Classes were covered in 02_programming/04_classes_basics.py)
#
# Architecture: input(2) -> hidden1(16) -> hidden2(8) -> output(1)
# This 3-layer network can learn curved boundaries!

def sigmoid(x):
    # Clip to prevent overflow in exp
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(a):
    return a * (1.0 - a)


class NeuralNetwork:
    """
    A 2-hidden-layer neural network for binary classification.
    input(n_in) -> hidden(n_h1) -> hidden(n_h2) -> output(1)
    """

    def __init__(self, n_input, n_hidden1, n_hidden2, n_output=1):
        """Initialize the network with random weights."""
        # Layer 1: input -> hidden1
        # We use "He initialization": scale by sqrt(2/n_inputs)
        # This prevents vanishing/exploding gradients early in training.
        self.W1 = np.random.randn(n_input, n_hidden1) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros(n_hidden1)

        # Layer 2: hidden1 -> hidden2
        self.W2 = np.random.randn(n_hidden1, n_hidden2) * np.sqrt(2.0 / n_hidden1)
        self.b2 = np.zeros(n_hidden2)

        # Layer 3: hidden2 -> output
        self.W3 = np.random.randn(n_hidden2, n_output) * np.sqrt(2.0 / n_hidden2)
        self.b3 = np.zeros(n_output)

        # Count total parameters
        self.n_params = (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W3.size + self.b3.size
        )

    def forward(self, x):
        """
        Forward pass through all layers.
        Stores intermediate values (needed for backprop).
        """
        self.x = x

        # Layer 1
        self.z1 = x @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        # Layer 3 (output)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = sigmoid(self.z3)

        return self.a3

    def backward(self, y, learning_rate):
        """
        Backward pass: compute gradients and update weights.
        Uses chain rule applied layer by layer, from output to input.
        """
        m = self.x.shape[0]  # batch size

        # Output layer error
        delta3 = (self.a3 - y) * sigmoid_deriv(self.a3)

        dW3 = self.a2.T @ delta3 / m
        db3 = np.mean(delta3, axis=0)

        # Hidden layer 2 error (propagate delta3 back through W3)
        delta2 = (delta3 @ self.W3.T) * sigmoid_deriv(self.a2)

        dW2 = self.a1.T @ delta2 / m
        db2 = np.mean(delta2, axis=0)

        # Hidden layer 1 error (propagate delta2 back through W2)
        delta1 = (delta2 @ self.W2.T) * sigmoid_deriv(self.a1)

        dW1 = self.x.T @ delta1 / m
        db1 = np.mean(delta1, axis=0)

        # Update all weights (gradient descent)
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def predict(self, x):
        """Returns 0 or 1 predictions (thresholded at 0.5)."""
        probs = self.forward(x)
        return (probs >= 0.5).astype(int)

    def loss(self, y_pred, y_true):
        """Mean Squared Error."""
        return np.mean((y_pred - y_true) ** 2)


print("\n--- Part 2: Network Architecture ---")
net = NeuralNetwork(n_input=2, n_hidden1=16, n_hidden2=8)
print(f"  Architecture: input(2) -> hidden(16) -> hidden(8) -> output(1)")
print(f"  Total trainable parameters: {net.n_params}")
print(f"    W1: {net.W1.shape} = {net.W1.size} weights")
print(f"    b1: {net.b1.shape} = {net.b1.size} biases")
print(f"    W2: {net.W2.shape} = {net.W2.size} weights")
print(f"    b2: {net.b2.shape} = {net.b2.size} biases")
print(f"    W3: {net.W3.shape} = {net.W3.size} weights")
print(f"    b3: {net.b3.shape} = {net.b3.size} biases")


# =============================================================================
# FIRST PRINCIPLES: BATCH vs STOCHASTIC vs MINI-BATCH (Tradeoffs)
# =============================================================================
#
# FULL BATCH GRADIENT DESCENT (batch_size = n, all data):
#   - Cost per update: O(n * d * params) — processes ALL samples
#   - Gradient noise: ZERO — exact gradient over full dataset
#   - Memory: O(n * d) — must hold all data + activations in memory
#   - Convergence: smooth but slow in wall-clock time
#   - When to use: small datasets that fit in memory
#
# STOCHASTIC GRADIENT DESCENT (SGD, batch_size = 1):
#   - Cost per update: O(d * params) — processes ONE sample
#   - Gradient noise: HIGH — gradient from one sample is very noisy
#   - Memory: O(d) — only one sample at a time
#   - Convergence: fast per step but zig-zags a lot
#   - When to use: very large datasets, online learning
#
# MINI-BATCH GRADIENT DESCENT (batch_size = B, typically 32-256):
#   - Cost per update: O(B * d * params)
#   - Gradient noise: MODERATE — averaged over B samples
#   - Memory: O(B * d) — holds one batch in memory
#   - Convergence: best of both worlds!
#   - When to use: almost always in practice
#
# =============================================================================
# WHY MINI-BATCH WORKS: THE CENTRAL LIMIT THEOREM CONNECTION
# =============================================================================
# The true gradient is the average over ALL n samples.
# A mini-batch of size B estimates this average from B samples.
#
# By the Central Limit Theorem (CLT):
#   - Variance of a sample mean = (population variance) / (sample size)
#   - So: Var(mini-batch gradient) = Var(single-sample gradient) / B
#   - Standard deviation (noise) = sigma / sqrt(B)
#
# This means:
#   - Doubling batch size halves variance (reduces noise by sqrt(2))
#   - But computation also doubles!
#   - Sweet spot: B where noise reduction per FLOP is maximized
#   - In practice: B = 32 to 256 works well for most problems
#
# =============================================================================
# COMPLEXITY ANALYSIS
# =============================================================================
# One epoch = n/B weight updates (each on a batch of B samples)
# Cost per update = O(B * d * total_params)  [forward + backward pass]
# Cost per epoch  = (n/B) * O(B * d * params) = O(n * d * params)
#
# Key insight: cost per EPOCH is the SAME regardless of batch size!
# But quality per epoch differs (more updates = more learning per epoch).
# =============================================================================

# =============================================================================
# PART 3: BATCHING — PROCESSING MULTIPLE SAMPLES AT ONCE
# =============================================================================
# Instead of updating weights once per sample (stochastic gradient descent),
# we process a BATCH of samples and average the gradients.
#
# Why batch?
#   - More stable gradients (average over many samples, less noise)
#   - Faster (matrix operations are hardware-optimized)
#   - GPU-friendly (modern ML hardware loves large batches)
#
# Common batch sizes: 32, 64, 128, 256
# Rule of thumb: larger batch = more stable but needs more memory.
#
# "Full batch" = use ALL data at once (what we'll do here, 300 samples).
# "Mini-batch" = use a subset each step (more common in practice).
# "Stochastic" = batch size of 1 (one sample at a time, very noisy).

print("\n--- Part 3: Batching Demonstration ---")
print("  Testing forward pass with different batch sizes:")
for batch_size in [1, 10, 50, 300]:
    X_batch = X[:batch_size]
    preds = net.forward(X_batch)
    print(f"  batch_size={batch_size:3d}: X_batch.shape={X_batch.shape}, "
          f"predictions.shape={preds.shape}")
print("  Same code, different batch sizes — numpy handles the shapes!")


# =============================================================================
# PART 4: THE TRAINING LOOP
# =============================================================================
# This is the core of every neural network training session.
# The loop is the same whether you're training a tiny network or GPT-4.
# (GPT-4 just has 1.7 trillion parameters and trains for months on thousands of GPUs!)

def train(model, X, y, learning_rate, epochs, print_every=200):
    """
    Complete training loop.

    model:         NeuralNetwork instance
    X, y:          training data and labels
    learning_rate: step size for gradient descent
    epochs:        number of full passes through the data
    print_every:   print metrics every N epochs
    """
    loss_history = []
    acc_history  = []

    print(f"  Training: lr={learning_rate}, epochs={epochs}, samples={len(X)}")
    print(f"  {'Epoch':>7} | {'Loss':>8} | {'Accuracy':>10}")
    print("  " + "-" * 35)

    for epoch in range(epochs):
        # STEP 1: FORWARD PASS — compute predictions
        y_pred = model.forward(X)

        # STEP 2: COMPUTE LOSS — how wrong are we?
        current_loss = model.loss(y_pred, y)
        loss_history.append(current_loss)

        # STEP 3: BACKWARD PASS + UPDATE — adjust weights
        model.backward(y, learning_rate)

        # STEP 4: TRACK ACCURACY
        preds = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(preds == y) * 100
        acc_history.append(accuracy)

        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"  {epoch+1:>7} | {current_loss:>8.4f} | {accuracy:>9.1f}%")

    return loss_history, acc_history


print("\n--- Part 4: Training the Circle Classifier ---")
print()

# Create fresh network (re-initialize so results are clean)
np.random.seed(42)
net = NeuralNetwork(n_input=2, n_hidden1=16, n_hidden2=8)

loss_hist, acc_hist = train(
    model=net,
    X=X, y=y,
    learning_rate=0.5,
    epochs=2000,
    print_every=400
)


# =============================================================================
# PART 5: EFFECT OF LEARNING RATE
# =============================================================================
# The learning rate (lr) controls how big each step is in weight space.
#
#   Too small (lr=0.001): training is very slow, needs many more epochs
#   Just right (lr=0.1 - 1.0): converges in a reasonable number of steps
#   Too large (lr=10.0): steps are so big we "jump over" the minimum
#                        the loss may OSCILLATE or even DIVERGE (get worse!)
#
# Finding the right learning rate is an art. Common strategies:
#   - Start with lr=0.1 and adjust based on how fast loss decreases
#   - Learning rate schedulers: reduce lr automatically during training
#   - Adam optimizer (we'll cover this later): adapts lr per parameter

print("\n--- Part 5: Effect of Learning Rate ---")
print("  Comparing learning rates on the same problem (500 epochs each):")
print()

learning_rates_to_test = [0.01, 0.1, 0.5, 2.0]

for lr in learning_rates_to_test:
    np.random.seed(42)  # same starting weights for fair comparison
    test_net = NeuralNetwork(n_input=2, n_hidden1=16, n_hidden2=8)
    losses = []
    for _ in range(500):
        preds = test_net.forward(X)
        losses.append(test_net.loss(preds, y))
        test_net.backward(y, lr)
    final_preds = test_net.predict(X)
    final_acc   = np.mean(final_preds == y) * 100
    trend = "IMPROVING" if losses[-1] < losses[0] else "DIVERGING!"
    print(f"  lr={lr:.3f}: initial_loss={losses[0]:.4f} -> final_loss={losses[-1]:.4f} "
          f"| acc={final_acc:.1f}% | {trend}")


# =============================================================================
# PART 6: EPOCHS VS ITERATIONS
# =============================================================================
# EPOCH: one full pass through the entire training dataset.
# ITERATION: one weight update step.
#
# When using full-batch gradient descent:
#   1 epoch = 1 iteration  (we see all data, then update once)
#
# When using mini-batches (e.g., batch_size=32 with 320 samples):
#   10 iterations = 1 epoch  (32*10 = 320 = all data seen once)
#
# Common questions:
#   "How many epochs should I train?"
#   -> Until the loss stops decreasing (convergence)
#   -> Early stopping: stop when validation loss starts increasing (overfitting)
#
# Overfitting = the network memorizes training data but fails on new data.
# We'll cover this in depth later. For now, just know it exists!

print("\n--- Part 6: Epochs, Loss Snapshots ---")
print("  Loss snapshots during training:")

# Show the loss at key checkpoints
checkpoints = [0, 99, 199, 399, 799, 1599, 1999]
print()
print(f"  {'Epoch':>6} | {'Loss':>8} | {'Accuracy':>10} | Change")
print("  " + "-" * 45)
prev_loss = loss_hist[0]
for ep in checkpoints:
    if ep < len(loss_hist):
        change = loss_hist[ep] - prev_loss
        sign = "v" if change < 0 else ("^" if change > 0 else "=")
        print(f"  {ep+1:>6} | {loss_hist[ep]:>8.4f} | {acc_hist[ep]:>9.1f}% | {sign} {abs(change):.4f}")
        prev_loss = loss_hist[ep]


# =============================================================================
# PART 7: FINAL EVALUATION
# =============================================================================
print("\n--- Part 7: Final Evaluation ---")

final_preds = net.predict(X)
final_probs = net.forward(X)
final_loss  = net.loss(final_probs, y)
final_acc   = np.mean(final_preds == y) * 100

print(f"  Final loss:     {final_loss:.4f}")
print(f"  Final accuracy: {final_acc:.1f}%")
print()

# Show some predictions
print("  Sample predictions (first 10):")
print(f"  {'x':>7} {'y':>7} | {'Prob':>6} | {'Pred':>4} | {'True':>4} | {'OK?':>3}")
print("  " + "-" * 45)
for i in range(10):
    prob = final_probs[i, 0]
    pred = int(final_preds[i, 0])
    true = int(y[i, 0])
    ok = "YES" if pred == true else " NO"
    print(f"  {X[i,0]:+.4f} {X[i,1]:+.4f} | {prob:.4f} | {pred:>4} | {true:>4} | {ok:>3}")

# Confusion matrix style breakdown
tp = int(np.sum((final_preds == 1) & (y == 1)))
tn = int(np.sum((final_preds == 0) & (y == 0)))
fp = int(np.sum((final_preds == 1) & (y == 0)))
fn = int(np.sum((final_preds == 0) & (y == 1)))
print()
print("  Confusion Matrix:")
print(f"                  Predicted 0  |  Predicted 1")
print(f"  Actual 0:    TN={tn:>4}         FP={fp:>4}")
print(f"  Actual 1:    FN={fn:>4}         TP={tp:>4}")
print()
print(f"  True Positives  (correctly predicted inside): {tp}")
print(f"  True Negatives  (correctly predicted outside): {tn}")
print(f"  False Positives (said inside, actually outside): {fp}")
print(f"  False Negatives (said outside, actually inside): {fn}")


# =============================================================================
# KEY CONCEPTS SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
Training loop:      The core of ML — repeat forever:
                    forward -> loss -> backward -> update

Batch:              A group of samples processed together in one update.
                    batch_size=1: stochastic (noisy but fast per step)
                    batch_size=N: full batch (stable but slow per step)
                    batch_size=32-256: mini-batch (best of both worlds)

Epoch:              One full pass through all training data.

Learning rate:      Step size for gradient descent.
                    Too small: slow training. Too large: diverges.

Convergence:        When the loss stops decreasing — training is done.

Accuracy:           (correct predictions) / (total predictions) * 100%

Overfitting:        Network memorizes training data, fails on new data.
                    Solution: more data, regularization, early stopping.

He initialization:  Smart weight initialization: scale by sqrt(2/n_inputs).
                    Prevents vanishing/exploding gradients at start of training.
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- dataset shape checks ---
assert X.shape == (300, 2), f"X shape should be (300,2), got {X.shape}"
assert y.shape == (300, 1), f"y shape should be (300,1), got {y.shape}"
# Labels must be 0 or 1 only
assert set(y.flatten().tolist()).issubset({0.0, 1.0}), "y must contain only 0.0 or 1.0"

# --- network parameter count ---
# input(2)->h1(16): 2*16+16=48; h1->h2(8): 16*8+8=136; h2->output(1): 8*1+1=9 => 193
expected_n_params = (2*16 + 16) + (16*8 + 8) + (8*1 + 1)
assert net.n_params == expected_n_params, \
    f"Expected {expected_n_params} params, got {net.n_params}"

# --- forward pass output shape ---
preds_test = net.forward(X[:10])
assert preds_test.shape == (10, 1), f"Forward pass output shape wrong: {preds_test.shape}"
assert np.all(preds_test > 0) and np.all(preds_test < 1), \
    "Forward pass outputs should all be in (0,1)"

# --- training loop reduces loss ---
assert loss_hist[-1] < loss_hist[0], \
    f"Training should reduce loss: initial={loss_hist[0]:.4f}, final={loss_hist[-1]:.4f}"

# --- loss history has the right length ---
assert len(loss_hist) == 2000, f"Loss history should have 2000 entries, got {len(loss_hist)}"

# --- accuracy history: final accuracy better than initial ---
assert acc_hist[-1] > acc_hist[0] or acc_hist[-1] >= 50.0, \
    f"Final accuracy ({acc_hist[-1]:.1f}%) should be reasonable after training"

# --- predict returns 0/1 integers ---
binary_preds = net.predict(X[:5])
assert binary_preds.shape == (5, 1), f"Predict shape wrong: {binary_preds.shape}"
assert set(binary_preds.flatten().tolist()).issubset({0, 1}), "Predict must return 0 or 1"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("=" * 60)
print("EXERCISES — Try these yourself!")
print("=" * 60)
print("""
Exercise 1: Try Different Circle Radii
  In make_circle_dataset, the circle radius is 0.5.
  What if you change it to 0.3 (smaller circle) or 0.7 (bigger)?
  How does the class balance change? Does training still work?
  Hint: change the 0.5 in: y = (distances < 0.5)...

Exercise 2: Smaller Network
  Try architecture input(2) -> hidden(4) -> output(1) (skip hidden2).
  Does the smaller network still solve the circle problem?
  How does the accuracy compare?
  Hint: set n_hidden2=0 and remove the W3/b3 layer, making W2 connect directly to output.
  OR simpler: try NeuralNetwork(n_input=2, n_hidden1=4, n_hidden2=2)

Exercise 3: More Epochs
  Train for 5000 epochs instead of 2000.
  Does accuracy keep improving, or does it plateau?
  Print loss at epochs 2000, 3000, 4000, 5000.

Exercise 4: Track the Best Accuracy
  Modify the train() function to track the BEST accuracy seen
  during training (not just the final one).
  Hint: best_acc = 0; if accuracy > best_acc: best_acc = accuracy

Exercise 5: Two Circles (harder!)
  Create a new dataset where label=1 if 0.2 < distance < 0.5
  (an annulus/ring shape — outside inner circle, inside outer circle).
  Can the network learn this? It requires an even more curved boundary.
  Hint: y = ((distances > 0.2) & (distances < 0.5)).astype(float)

Exercise 6 (First Principles): Batch Size and Variance
  If SGD (batch_size=1) has gradient variance sigma^2,
  what batch size B reduces the variance to sigma^2 / 16?

  Answer: By CLT, Var(batch gradient) = sigma^2 / B
  We want sigma^2 / B = sigma^2 / 16
  So B = 16.

  A batch size of 16 reduces noise by a factor of sqrt(16) = 4.
  But it costs 16x more computation per update than SGD.
  Is that worth it? Usually yes, because:
    - Matrix operations on batches are hardware-optimized (GPU parallelism)
    - 16 samples in a batch might only take 2-3x longer than 1 sample
    - The noise reduction makes convergence much smoother
""")
