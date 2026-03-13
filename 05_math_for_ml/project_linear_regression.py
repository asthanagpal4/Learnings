# HOW TO RUN:
#   uv run python 05_math_for_ml/project_linear_regression.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# PROJECT: Complete Linear Regression with Gradient Descent
# =============================================================================
# This is a MINI-PROJECT that brings together everything from files 01-05.
#
# You will build a complete linear regression system from scratch:
#   1. Generate a realistic synthetic dataset
#   2. Initialize model weights
#   3. Forward pass (make predictions)
#   4. Compute loss (MSE)
#   5. Compute gradients (calculus)
#   6. Update weights (gradient descent)
#   7. Repeat for many epochs
#   8. Evaluate and make new predictions
#
# This exact loop is the FOUNDATION of all neural network training.
# GPT, DALL-E, AlphaFold — they all use this same basic loop.
# =============================================================================

import numpy as np

print("=" * 60)
print("MINI-PROJECT: LINEAR REGRESSION FROM SCRATCH")
print("=" * 60)
print("""
Problem: Predict student exam scores from study hours.
         We generate fake (synthetic) data where we KNOW the true answer.
         Then we check if our learned model recovers the truth.
""")

# =============================================================================
# STEP 1: GENERATE SYNTHETIC DATASET
# =============================================================================

print("=" * 60)
print("STEP 1: Generate Synthetic Data")
print("=" * 60)

np.random.seed(2024)   # fixed seed so results are the same every run

# True relationship we want the model to discover:
#   score = TRUE_WEIGHT * hours + TRUE_BIAS + noise
#
# We choose these values ourselves — the model doesn't know them.
# After training, the learned values should be close to these.

TRUE_WEIGHT = 8.5    # each extra hour of study = +8.5 points
TRUE_BIAS   = 30.0   # even studying 0 hours, you'd score ~30
NOISE_STD   = 6.0    # real world has randomness

N_SAMPLES = 80

# Generate study hours uniformly between 0 and 12
hours = np.random.uniform(0, 12, N_SAMPLES)

# Generate scores with noise
noise  = np.random.randn(N_SAMPLES) * NOISE_STD
scores = TRUE_WEIGHT * hours + TRUE_BIAS + noise

# Clamp scores to [0, 100] range (can't score below 0 or above 100)
scores = np.clip(scores, 0, 100)

print(f"Generated {N_SAMPLES} student records")
print(f"True relationship: score = {TRUE_WEIGHT} * hours + {TRUE_BIAS} + noise")
print(f"Noise level (std dev): {NOISE_STD}")
print(f"\nData statistics:")
print(f"  Hours: min={hours.min():.1f}, max={hours.max():.1f}, mean={hours.mean():.1f}")
print(f"  Scores: min={scores.min():.1f}, max={scores.max():.1f}, mean={scores.mean():.1f}")

# Preview the data
print(f"\nFirst 10 data points:")
print(f"{'#':>4} | {'Hours Studied':>14} | {'Exam Score':>11}")
print("-" * 36)
for i in range(10):
    print(f"{i+1:>4} | {hours[i]:>14.2f} | {scores[i]:>11.1f}")

# Quick scatter-like text visualization
print(f"\nText scatter plot (hours vs scores):")
print(f"{'Score':>6}|")
bins = [(score_low, score_low + 10) for score_low in range(30, 101, 10)]
for low, high in reversed(bins):
    mask  = (scores >= low) & (scores < high)
    count = np.sum(mask)
    avg_hour_if_any = hours[mask].mean() if count > 0 else 0
    bar   = "*" * count
    print(f"{low:>6}|{bar}")
print(f"{'':>6}+{'---' * 15}")
print(f"{'':>6} 0  1  2  3  4  5  6  7  8  9  10  <- Hours")

# =============================================================================
# STEP 2: TRAIN / TEST SPLIT
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Split into Training and Test Sets")
print("=" * 60)

# We train on 80% of data and test on the remaining 20%.
# The test set is NEVER used during training — it's for evaluation only.
# This tells us how well the model generalizes to new, unseen data.

n_train = int(0.8 * N_SAMPLES)   # 64 samples for training
n_test  = N_SAMPLES - n_train     # 16 samples for testing

# Shuffle indices so train/test split is random
indices = np.random.permutation(N_SAMPLES)
train_idx = indices[:n_train]
test_idx  = indices[n_train:]

X_train = hours[train_idx]
y_train = scores[train_idx]
X_test  = hours[test_idx]
y_test  = scores[test_idx]

print(f"Training samples: {n_train}")
print(f"Test samples:     {n_test}")
print(f"Train hours range: {X_train.min():.1f} to {X_train.max():.1f}")
print(f"Test  hours range: {X_test.min():.1f} to {X_test.max():.1f}")

# =============================================================================
# STEP 3: INITIALIZE WEIGHTS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Initialize Model Weights")
print("=" * 60)

# Our model: score = weight * hours + bias
# We initialize with random small values.
# The model doesn't know the true weight or bias yet.

np.random.seed(1)
weight = np.random.randn()     # random starting weight
bias   = np.random.randn()     # random starting bias

print(f"Initialized weight = {weight:.4f}  (true = {TRUE_WEIGHT})")
print(f"Initialized bias   = {bias:.4f}  (true = {TRUE_BIAS})")

# =============================================================================
# STEP 4: DEFINE HELPER FUNCTIONS
# =============================================================================

def forward_pass(X, w, b):
    """Make predictions: y_pred = w * X + b"""
    return w * X + b

def compute_mse(y_pred, y_true):
    """Mean Squared Error: average of (prediction - true)^2"""
    errors = y_pred - y_true
    return np.mean(errors ** 2)

def compute_rmse(y_pred, y_true):
    """Root Mean Squared Error: sqrt(MSE) — in the same units as y"""
    return np.sqrt(compute_mse(y_pred, y_true))

def compute_mae(y_pred, y_true):
    """Mean Absolute Error: average |prediction - true|"""
    return np.mean(np.abs(y_pred - y_true))

def compute_gradients(X, y_pred, y_true):
    """
    Compute gradients of MSE loss w.r.t. weight and bias.

    Loss = (1/n) * sum((y_pred - y_true)^2)
    y_pred = w * X + b

    By calculus (chain rule):
    d_Loss/d_w = (2/n) * sum((y_pred - y_true) * X)
    d_Loss/d_b = (2/n) * sum(y_pred - y_true)
    """
    n      = len(y_true)
    errors = y_pred - y_true
    grad_w = (2 / n) * np.dot(errors, X)
    grad_b = (2 / n) * np.sum(errors)
    return grad_w, grad_b

# Check initial performance
initial_pred = forward_pass(X_train, weight, bias)
initial_loss = compute_mse(initial_pred, y_train)
print(f"\nInitial training loss (MSE): {initial_loss:.2f}")
print(f"Initial RMSE: {np.sqrt(initial_loss):.2f} points  (average prediction error)")

# =============================================================================
# STEP 5: TRAINING LOOP
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Training Loop (Gradient Descent)")
print("=" * 60)

# ===========================================================================
# FIRST PRINCIPLES: Complexity analysis of training
# ===========================================================================
# Each epoch does the following work:
#   - Forward pass:  y_pred = w * X + b          -> O(n) for n samples
#   - Loss:          mean((y_pred - y)^2)         -> O(n)
#   - Gradient w:    (2/n) * dot(errors, X)       -> O(n)  (or O(n*d) for d features)
#   - Gradient b:    (2/n) * sum(errors)           -> O(n)
#   - Update w, b:   w -= lr * grad_w              -> O(1)  (or O(d) for d features)
#
# Total per epoch: O(n * d) where n = samples, d = features
# Total training:  O(n * d * epochs)
#
# For this problem: n=64, d=1, epochs=1000 -> ~64,000 operations
#
# ALTERNATIVE: The Normal Equation (closed-form solution)
# -------------------------------------------------------
# Instead of iterating, we can solve for the optimal w directly:
#   w = (X^T X)^(-1) X^T y
#
# Complexity: O(n*d^2 + d^3)
#   - X^T X is O(n*d^2) -- n dot products of d-dimensional vectors
#   - Matrix inverse is O(d^3)
#
# When is gradient descent better than the normal equation?
#   - Normal equation: O(n*d^2 + d^3), no iterations needed
#   - Gradient descent: O(n*d*epochs)
#   - GD is better when d is large (d^3 is expensive for large d)
#   - Normal equation is better when d is small and n is moderate
#   - Rough crossover: when d > ~1000, prefer gradient descent
#   - Also: normal equation requires the matrix X^T X to be invertible
# ===========================================================================

LEARNING_RATE = 0.005
N_EPOCHS      = 1000

# Storage for tracking progress
loss_history    = []
weight_history  = []
bias_history    = []

print(f"Hyperparameters:")
print(f"  Learning rate : {LEARNING_RATE}")
print(f"  Epochs        : {N_EPOCHS}")
print(f"  Training size : {n_train} samples")
print()

print(f"{'Epoch':>7} | {'Train Loss (MSE)':>17} | {'RMSE':>8} | {'Weight':>10} | {'Bias':>10} | {'Change'}")
print("-" * 75)

prev_loss = None

for epoch in range(N_EPOCHS):
    # --- Forward pass ---
    y_pred = forward_pass(X_train, weight, bias)

    # --- Compute loss ---
    loss = compute_mse(y_pred, y_train)
    loss_history.append(loss)
    weight_history.append(weight)
    bias_history.append(bias)

    # --- Compute gradients ---
    grad_w, grad_b = compute_gradients(X_train, y_pred, y_train)

    # --- Update weights (gradient descent step) ---
    weight = weight - LEARNING_RATE * grad_w
    bias   = bias   - LEARNING_RATE * grad_b

    # --- Print progress every 100 epochs ---
    if (epoch + 1) % 100 == 0 or epoch == 0:
        rmse   = np.sqrt(loss)
        if prev_loss is not None:
            change = loss - prev_loss
            change_str = f"{change:+.2f}"
        else:
            change_str = "  start"
        prev_loss = loss
        print(f"{epoch+1:>7} | {loss:>17.4f} | {rmse:>8.4f} | {weight:>10.4f} | {bias:>10.4f} | {change_str}")

print()
print(f"Training complete!")
print(f"\n  Learned weight = {weight:.4f}   (true = {TRUE_WEIGHT})")
print(f"  Learned bias   = {bias:.4f}   (true = {TRUE_BIAS})")
print(f"  Weight error   = {abs(weight - TRUE_WEIGHT):.4f}")
print(f"  Bias error     = {abs(bias - TRUE_BIAS):.4f}")

# =============================================================================
# STEP 6: EVALUATE ON TEST SET
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Evaluate on Test Set (Unseen Data)")
print("=" * 60)

train_pred = forward_pass(X_train, weight, bias)
test_pred  = forward_pass(X_test,  weight, bias)

train_mse  = compute_mse(train_pred,  y_train)
test_mse   = compute_mse(test_pred,   y_test)
train_rmse = compute_rmse(train_pred, y_train)
test_rmse  = compute_rmse(test_pred,  y_test)
train_mae  = compute_mae(train_pred,  y_train)
test_mae   = compute_mae(test_pred,   y_test)

print(f"\n{'Metric':>25} | {'Train Set':>12} | {'Test Set':>12}")
print("-" * 55)
print(f"{'MSE (Mean Squared Error)':>25} | {train_mse:>12.2f} | {test_mse:>12.2f}")
print(f"{'RMSE (Root MSE)':>25} | {train_rmse:>12.2f} | {test_rmse:>12.2f}")
print(f"{'MAE (Mean Abs Error)':>25} | {train_mae:>12.2f} | {test_mae:>12.2f}")

print(f"""
Interpretation:
  RMSE = {test_rmse:.1f} points means our predictions are off by ~{test_rmse:.0f} points on average.
  For a 0-100 scale, that's {'good' if test_rmse < 10 else 'ok' if test_rmse < 15 else 'rough'}.

  Train vs Test RMSE should be similar — if test is much higher,
  the model is "overfitting" (memorized training data, can't generalize).
  Here they are close, which is expected for linear regression.
""")

# Show test predictions vs actual
print(f"Test Set Predictions vs Actual:")
print(f"{'#':>4} | {'Hours':>8} | {'Predicted':>12} | {'Actual':>10} | {'Error':>8}")
print("-" * 50)
for i in range(len(X_test)):
    error = test_pred[i] - y_test[i]
    print(f"{i+1:>4} | {X_test[i]:>8.2f} | {test_pred[i]:>12.1f} | {y_test[i]:>10.1f} | {error:>+8.1f}")

# =============================================================================
# STEP 7: MAKE PREDICTIONS ON NEW DATA
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Make Predictions on New Data")
print("=" * 60)

new_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
predicted = forward_pass(new_hours, weight, bias)
true_val  = TRUE_WEIGHT * new_hours + TRUE_BIAS   # what we'd get with perfect params

print(f"\n{'Hours':>8} | {'Predicted Score':>16} | {'True Score':>12} | {'Diff':>8}")
print("-" * 50)
for h, p, t in zip(new_hours, predicted, true_val):
    diff = p - t
    print(f"{h:>8} | {p:>16.1f} | {t:>12.1f} | {diff:>+8.1f}")

print(f"\nLearned equation: score = {weight:.2f} * hours + {bias:.2f}")
print(f"True  equation:   score = {TRUE_WEIGHT} * hours + {TRUE_BIAS}")

# =============================================================================
# STEP 8: LOSS CURVE ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Loss Curve Analysis")
print("=" * 60)

# Show how fast the loss decreased
print("\nLoss reduction over training:")
print(f"{'Epoch':>8} | {'MSE Loss':>12} | {'RMSE':>8} | Visual")
print("-" * 55)

sample_points = [0, 9, 24, 49, 99, 199, 399, 699, 999]
for ep in sample_points:
    loss_val = loss_history[ep]
    rmse_val = np.sqrt(loss_val)
    bar_len  = int(loss_val / 20)
    bar      = "#" * min(bar_len, 50)
    print(f"{ep+1:>8} | {loss_val:>12.2f} | {rmse_val:>8.2f} | {bar}")

# ===========================================================================
# FIRST PRINCIPLES: Normal equation comparison
# ===========================================================================
print("\n--- Normal Equation: Closed-Form Solution ---")
# w = (X^T X)^(-1) X^T y
# For 1D, we need to add a bias column of 1s
X_design = np.column_stack([X_train, np.ones(n_train)])  # shape (n, 2): [hours, 1]
# Normal equation: w = (X^T X)^(-1) X^T y
XtX = X_design.T @ X_design          # (2, 2)
XtX_inv = np.linalg.inv(XtX)         # (2, 2)
w_normal = XtX_inv @ X_design.T @ y_train  # (2,)
print(f"  Normal equation solution: weight = {w_normal[0]:.4f}, bias = {w_normal[1]:.4f}")
print(f"  Gradient descent result:  weight = {weight:.4f}, bias = {bias:.4f}")
print(f"  True values:              weight = {TRUE_WEIGHT}, bias = {TRUE_BIAS}")
print(f"  Normal equation gives the same answer instantly (no iterations)!")
print(f"  But: O(n*d^2 + d^3) vs gradient descent O(n*d*epochs)")

total_reduction = loss_history[0] - loss_history[-1]
pct_reduction   = total_reduction / loss_history[0] * 100
print(f"\nTotal loss reduction: {total_reduction:.2f}  ({pct_reduction:.1f}%)")

# How quickly did we converge?
threshold = loss_history[-1] * 1.05  # within 5% of final loss
for ep, loss_val in enumerate(loss_history):
    if loss_val <= threshold:
        print(f"Essentially converged at epoch {ep + 1}")
        break

# =============================================================================
# STEP 9: VISUALIZE WEIGHT CONVERGENCE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: How Did the Parameters Evolve?")
print("=" * 60)

print(f"\n{'Epoch':>8} | {'Weight':>10} | {'Bias':>10} | {'Dist to True Weight':>20}")
print("-" * 56)
for ep in sample_points:
    w_val  = weight_history[ep]
    b_val  = bias_history[ep]
    w_dist = abs(w_val - TRUE_WEIGHT)
    print(f"{ep+1:>8} | {w_val:>10.4f} | {b_val:>10.4f} | {w_dist:>20.4f}")

print(f"\nTrue weight: {TRUE_WEIGHT}   True bias: {TRUE_BIAS}")
print(f"The weight and bias started random and converged to the true values!")

# =============================================================================
# STEP 10: CONNECTION TO NEURAL NETWORKS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: How This Connects to Neural Networks")
print("=" * 60)

print(f"""
What you just built is called a "Linear Neuron" — the simplest possible
neural network: one input, one output, one weight, one bias.

    Input (hours) --[weight]--> sum + bias --> Output (score)

A real neural network is just MANY of these connected together:
  - Multiple inputs (features like hours, sleep, stress level)
  - Multiple outputs (or intermediate "hidden" layers)
  - Activation functions between layers (sigmoid, ReLU, tanh)
  - Same training loop: forward pass -> loss -> gradients -> update

The training loop you coded in Step 5:
  for epoch in epochs:
      y_pred = forward_pass(X, weight, bias)     <- same
      loss   = compute_mse(y_pred, y_true)        <- same (or cross-entropy)
      grad_w, grad_b = compute_gradients(...)     <- same (backpropagation)
      weight -= lr * grad_w                       <- same (SGD / Adam)
      bias   -= lr * grad_b                       <- same

This is EXACTLY how PyTorch and TensorFlow train neural networks.
The only difference: they do it automatically (autograd), support GPUs,
and handle millions of parameters instead of just 2.

You now understand the CORE of deep learning!
""")

# =============================================================================
# STEP 11: EXPERIMENTS (Summary)
# =============================================================================

print("=" * 60)
print("STEP 11: Quick Experiments")
print("=" * 60)

# Experiment: What if we used a higher learning rate?
def run_experiment(lr, n_ep, label):
    np.random.seed(1)
    w = np.random.randn()
    b = np.random.randn()
    history = []
    for _ in range(n_ep):
        y_p   = w * X_train + b
        loss  = compute_mse(y_p, y_train)
        history.append(loss)
        gw, gb = compute_gradients(X_train, y_p, y_train)
        w -= lr * gw
        b -= lr * gb
        if loss > 1e8 or np.isnan(loss):
            history.extend([float('inf')] * (n_ep - len(history)))
            break
    final = history[-1]
    final_str = f"{final:.2f}" if final != float('inf') and not np.isnan(final) else "DIVERGED"
    print(f"  {label}: final loss = {final_str}, final w = {w:.3f}, b = {b:.3f}")

print(f"\nEffect of learning rate (after {N_EPOCHS} epochs):")
run_experiment(0.0001, N_EPOCHS, "LR=0.0001 (too slow)")
run_experiment(0.005,  N_EPOCHS, "LR=0.005  (just right)")
run_experiment(0.05,   N_EPOCHS, "LR=0.05   (risky)")
run_experiment(0.5,    N_EPOCHS, "LR=0.5    (too high)")

print(f"\nEffect of dataset size (LR=0.005, {N_EPOCHS} epochs):")
for n in [10, 30, 60, 80]:
    np.random.seed(1)
    h_sub = hours[:n]
    s_sub = scores[:n]
    w, b  = np.random.randn(), np.random.randn()
    for _ in range(N_EPOCHS):
        y_p = w * h_sub + b
        gw  = (2 / n) * np.dot(y_p - s_sub, h_sub)
        gb  = (2 / n) * np.sum(y_p - s_sub)
        w  -= 0.005 * gw
        b  -= 0.005 * gb
    print(f"  n={n:>3}: learned w={w:.3f} (true={TRUE_WEIGHT}), b={b:.3f} (true={TRUE_BIAS})")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
You have built a complete machine learning pipeline:

  Data generation  -> Splits data into train/test
  Initialization   -> Random starting weights
  Forward pass     -> y_pred = w * x + b
  Loss function    -> MSE = mean((y_pred - y_true)^2)
  Gradients        -> d_loss/d_w, d_loss/d_b
  Weight update    -> w = w - lr * gradient
  Training loop    -> Repeat until convergence
  Evaluation       -> RMSE, MAE on unseen test data
  New predictions  -> Apply learned model to new inputs

Learned: weight={weight:.3f} (true={TRUE_WEIGHT}), bias={bias:.3f} (true={TRUE_BIAS})
Final test RMSE: {test_rmse:.2f} points

This is the foundation of ALL neural network training.
Next steps: multiple inputs (features), multiple layers, nonlinearities,
and you have a full neural network!
""")

# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test forward_pass function
assert forward_pass(np.array([0.0]), 2.0, 3.0)[0] == 3.0, "forward_pass(0, w=2, b=3) should be 3"
assert forward_pass(np.array([5.0]), 2.0, 0.0)[0] == 10.0, "forward_pass(5, w=2, b=0) should be 10"

# Test compute_mse
assert compute_mse(np.array([1.0]), np.array([1.0])) == 0.0, "MSE of identical arrays should be 0"
assert compute_mse(np.array([3.0, 5.0]), np.array([1.0, 3.0])) == 4.0, "MSE of errors [2,2] should be 4"

# Test compute_rmse
assert abs(compute_rmse(np.array([3.0, 5.0]), np.array([1.0, 3.0])) - 2.0) < 1e-9, "RMSE of errors [2,2] should be 2"

# Test compute_mae
assert abs(compute_mae(np.array([3.0, 5.0]), np.array([1.0, 3.0])) - 2.0) < 1e-9, "MAE of errors [2,2] should be 2"

# Test compute_gradients direction:
# if y_pred > y_true, grad_w should be positive (must decrease w)
X_g = np.array([1.0, 2.0])
yp_g = np.array([10.0, 10.0])
yt_g = np.array([1.0, 1.0])
gw_test, gb_test = compute_gradients(X_g, yp_g, yt_g)
assert gw_test > 0, "grad_w should be positive when overpredicting"
assert gb_test > 0, "grad_b should be positive when overpredicting"

# Core test: training must reduce the loss
assert loss_history[-1] < loss_history[0], "Final loss must be less than initial loss"

# After 1000 epochs, loss should be much smaller than at epoch 0
assert loss_history[-1] < loss_history[0] * 0.1, "Loss should reduce by at least 90%"

# Learned weight and bias should be close to true values
assert abs(weight - TRUE_WEIGHT) < 2.0, f"Learned weight {weight:.2f} too far from true {TRUE_WEIGHT}"
assert abs(bias - TRUE_BIAS) < 5.0, f"Learned bias {bias:.2f} too far from true {TRUE_BIAS}"

# Test set loss should be reasonably low (RMSE < 15 for a noisy dataset)
assert test_rmse < 15.0, f"Test RMSE {test_rmse:.2f} is too high, training may have failed"

# Weight and bias histories have the right length
assert len(loss_history) == N_EPOCHS, "Loss history length should equal N_EPOCHS"
assert len(weight_history) == N_EPOCHS, "Weight history length should equal N_EPOCHS"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES (extend this project)
# =============================================================================
# 1. Change TRUE_WEIGHT to 15 and TRUE_BIAS to 10. Retrain.
#    Does gradient descent still find the correct values?
#
# 2. Change NOISE_STD to 0 (no noise). What happens to the final RMSE?
#    Does the model learn the exact true parameters?
#
# 3. Change N_SAMPLES to 10 (very small dataset). Does the model still
#    recover the true weight and bias well? How does test RMSE compare?
#
# 4. Change N_SAMPLES to 500. Does the model do better?
#    More data usually means better estimates.
#
# 5. EXTENSION: Add a second feature.
#    True model: score = 8.5 * hours + 3.0 * sleep + 30
#    Hint: you'll need TWO weights (one per feature).
#    X becomes a matrix of shape (n, 2), and y_pred = X @ weights + bias
#    Gradient for weights: (2/n) * X.T @ errors
#    Gradient for bias:    (2/n) * sum(errors)
#
# 6. Plot the loss curve by printing it as a bar chart for every 50 epochs.
#    Can you see the "elbow" where loss stops decreasing quickly?
#
# 7. FIRST PRINCIPLES EXERCISE: Calculate when gradient descent becomes
#    more efficient than the normal equation.
#    Normal equation cost: O(n*d^2 + d^3)
#    Gradient descent cost: O(n*d*epochs)
#    For n=10000 samples and epochs=100, at what value of d (number of
#    features) does gradient descent become cheaper?
#    Hint: set n*d*epochs = n*d^2 + d^3 and solve for d.
#    For n >> d: n*d*100 = n*d^2 -> d = 100 features.
#    So with 100 epochs, GD becomes cheaper when d > ~100 features.
# =============================================================================
