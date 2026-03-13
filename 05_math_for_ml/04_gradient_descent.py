# HOW TO RUN:
#   uv run python 05_math_for_ml/04_gradient_descent.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE 4: Gradient Descent
# =============================================================================
# This is THE core algorithm behind all of machine learning.
# Everything — from linear regression to GPT — uses this idea.
#
# Topics:
#   - The loss (error) function: how wrong are our predictions?
#   - Mean Squared Error (MSE)
#   - What is a gradient?
#   - Gradient descent algorithm step by step
#   - Linear regression from scratch: y = mx + b
#   - Learning rate experiments
# =============================================================================

import numpy as np

print("=" * 60)
print("PART 1: THE CORE IDEA")
print("=" * 60)

print("""
Imagine you're blindfolded, standing somewhere in hilly terrain.
Your goal: reach the lowest point (the valley floor).

How do you do it?
  1. Feel the slope under your feet.
  2. Take a small step in the downhill direction.
  3. Repeat.

That's GRADIENT DESCENT!

In machine learning:
  - The "terrain" = the loss surface (a function of model parameters)
  - The "height"  = how wrong our model is (the loss/error)
  - The "slope"   = the gradient (direction of steepest increase)
  - A "step"      = updating our parameters to reduce the error
  - "Lowest point"= best possible parameters (minimum error)

We keep adjusting parameters until the error stops decreasing.
""")

print("=" * 60)
print("PART 2: LOSS FUNCTION AND MSE")
print("=" * 60)

# ------------------------------------------------------------------
# A loss function (also called cost function or error function) measures
# HOW WRONG our model's predictions are compared to the true answers.
#
# Mean Squared Error (MSE) is the most common loss for regression:
#   MSE = (1/n) * sum( (prediction - true_value)^2 )
#
# Why squared? Because:
#   1. It makes negative errors positive (we care about magnitude, not sign)
#   2. It penalizes large errors MORE heavily (2^2=4 is much larger than 1^2=1)
#   3. It has a nice smooth derivative (important for gradient descent)
# ------------------------------------------------------------------

def mse_loss(predictions, targets):
    """Mean Squared Error loss."""
    errors = predictions - targets
    return np.mean(errors ** 2)

# Simple example
targets     = np.array([10, 20, 30, 40, 50])
predictions_good = np.array([11, 19, 31, 39, 51])   # close to true
predictions_bad  = np.array([5,  30, 15, 60, 20])   # far from true

loss_good = mse_loss(predictions_good, targets)
loss_bad  = mse_loss(predictions_bad,  targets)

print("\n--- MSE Loss Examples ---")
print("True values:      ", targets)
print("Good predictions: ", predictions_good)
print("Bad predictions:  ", predictions_bad)
print(f"\nLoss (good model): {loss_good:.2f}   <- small, close to 0 is best")
print(f"Loss (bad model):  {loss_bad:.2f}   <- large, far from 0 is bad")

# Show why squaring helps — it punishes large errors more
print("\n--- Why Squaring Matters ---")
errors = np.array([0, 1, 2, 5, 10])
print(f"{'Error':>8} | {'Error^2':>8} | {'Proportional punishment'}")
print("-" * 50)
for e in errors:
    bar = "#" * (e**2 // 2)
    print(f"{e:>8} | {e**2:>8} | {bar}")

print("\n" + "=" * 60)
print("PART 3: WHAT IS A GRADIENT?")
print("=" * 60)

print("""
The GRADIENT is the derivative of the loss with respect to each parameter.
It tells us:
  "If I increase this parameter by a tiny bit, does the loss go UP or DOWN,
   and by how much?"

If gradient > 0: increasing the parameter INCREASES the loss
                 -> we should DECREASE the parameter
If gradient < 0: increasing the parameter DECREASES the loss
                 -> we should INCREASE the parameter

So the update rule is:
  parameter = parameter - learning_rate * gradient

The LEARNING RATE is a small number (like 0.01) that controls step size.
  - Too large: we overshoot the minimum, loss oscillates or diverges
  - Too small: learning is correct but very slow
  - Just right: loss decreases smoothly to minimum
""")

# ===========================================================================
# FIRST PRINCIPLES: Deriving MSE gradient by hand
# ===========================================================================
# Model: y_pred = w*x + b
# Loss:  MSE = (1/n) * SUM_i (y_i - (w*x_i + b))^2
#
# We need partial derivatives with respect to w and b.
#
# --- Partial derivative w.r.t. w ---
# Let L_i = (y_i - w*x_i - b)^2
# Using chain rule:
#   dL_i/dw = 2*(y_i - w*x_i - b) * d/dw(y_i - w*x_i - b)
#           = 2*(y_i - w*x_i - b) * (-x_i)
#
# So: dMSE/dw = (1/n) * SUM_i 2*(y_i - w*x_i - b)*(-x_i)
#             = -(2/n) * SUM_i (y_i - y_pred_i) * x_i
#             = (2/n) * SUM_i (y_pred_i - y_i) * x_i
#             = (2/n) * dot(errors, x)
#
# --- Partial derivative w.r.t. b ---
#   dL_i/db = 2*(y_i - w*x_i - b) * (-1)
#
# So: dMSE/db = -(2/n) * SUM_i (y_i - y_pred_i)
#             = (2/n) * SUM_i (y_pred_i - y_i)
#             = (2/n) * sum(errors)
#
# These are the EXACT formulas used in the training loop below.
#
# --- Why does gradient descent converge for MSE? (Informal proof) ---
# MSE is a CONVEX function of w and b (it's a sum of squares).
# Convex means the second derivative is always >= 0 (the surface curves
# upward like a bowl). For a bowl-shaped surface:
#   - The gradient points "uphill" (direction of steepest ascent)
#   - Negative gradient points "downhill"
#   - Any step in the negative gradient direction reduces the loss
#   - With a small enough learning rate, we're guaranteed to converge
#     to the unique global minimum.
#
# --- Complexity per epoch ---
# Computing y_pred = w*x + b: O(n) for n samples
# Computing errors: O(n)
# Computing grad_w = dot(errors, x): O(n)
# Computing grad_b = sum(errors): O(n)
# Total per epoch: O(n) for 1 feature, O(n*d) for d features
# Total training: O(n * d * epochs)
# ===========================================================================

print("\n--- FIRST PRINCIPLES: MSE gradient derivation ---")
print("  MSE = (1/n) * SUM (y_pred - y_true)^2")
print("  dMSE/dw = (2/n) * SUM (y_pred - y_true) * x  = (2/n) * dot(errors, x)")
print("  dMSE/db = (2/n) * SUM (y_pred - y_true)       = (2/n) * sum(errors)")
print("  MSE is convex -> gradient descent is guaranteed to converge")
print("  Complexity per epoch: O(n) for n samples with 1 feature")

# ------------------------------------------------------------------
# Simple 1D example: minimize f(w) = (w - 3)^2
# The minimum is clearly at w = 3, where f(3) = 0.
# Derivative: f'(w) = 2*(w - 3)
# ------------------------------------------------------------------

def simple_loss(w):
    return (w - 3) ** 2

def simple_gradient(w):
    return 2 * (w - 3)

print("--- Simple Example: Minimize f(w) = (w - 3)^2 ---")
print("True minimum is at w = 3 (where f = 0)\n")

w = 10.0          # start far from the minimum
learning_rate = 0.1
print(f"Starting at w = {w}, loss = {simple_loss(w):.4f}")
print(f"\n{'Step':>5} | {'w':>8} | {'loss':>10} | {'gradient':>10}")
print("-" * 45)

for step in range(1, 21):
    grad = simple_gradient(w)
    w    = w - learning_rate * grad
    loss = simple_loss(w)
    print(f"{step:>5} | {w:>8.4f} | {loss:>10.6f} | {grad:>10.4f}")

print(f"\nAfter 20 steps: w = {w:.4f}  (should be close to 3.0)")

print("\n" + "=" * 60)
print("PART 4: LINEAR REGRESSION FROM SCRATCH")
print("=" * 60)

# ------------------------------------------------------------------
# Linear regression: given input x, predict output y using
#   y_pred = m * x + b
#   m = slope (weight), b = intercept (bias)
#
# We need to FIND the best m and b from data.
# We do this with gradient descent!
#
# Loss = MSE = (1/n) * sum( (m*x + b - y_true)^2 )
#
# Gradients (from calculus):
#   dLoss/dm = (2/n) * sum( (y_pred - y_true) * x )
#   dLoss/db = (2/n) * sum( (y_pred - y_true) )
# ------------------------------------------------------------------

# --- Step 1: Generate synthetic data ---
# True relationship: score = 10 * hours_studied + 20 + noise
# So true m = 10, true b = 20
# After training, our learned m and b should be close to 10 and 20.

np.random.seed(42)   # for reproducibility (same random numbers every run)
n_samples    = 50
true_m       = 10.0
true_b       = 20.0
noise_level  = 5.0

hours_studied = np.random.uniform(0, 10, n_samples)  # 0 to 10 hours
noise         = np.random.randn(n_samples) * noise_level
test_scores   = true_m * hours_studied + true_b + noise

print(f"\nGenerated {n_samples} synthetic data points:")
print(f"  True relationship: score = {true_m} * hours + {true_b} + noise")
print(f"  Hours studied range: {hours_studied.min():.1f} to {hours_studied.max():.1f}")
print(f"  Score range: {test_scores.min():.1f} to {test_scores.max():.1f}")

# Preview first 5 data points
print("\nFirst 5 data points:")
print(f"{'Hours':>8} | {'Score':>8}")
print("-" * 20)
for i in range(5):
    print(f"{hours_studied[i]:>8.2f} | {test_scores[i]:>8.2f}")

# --- Step 2: Initialize weights randomly ---
np.random.seed(0)
m = np.random.randn()  # start with a random slope
b = np.random.randn()  # start with a random intercept
print(f"\n--- Initial (random) parameters ---")
print(f"Initial m = {m:.4f}  (true m = {true_m})")
print(f"Initial b = {b:.4f}  (true b = {true_b})")

x = hours_studied   # input features
y = test_scores     # target values

initial_pred = m * x + b
initial_loss = mse_loss(initial_pred, y)
print(f"Initial loss (MSE): {initial_loss:.2f}")

# --- Step 3: Training loop ---
learning_rate = 0.01
n_epochs      = 500
loss_history  = []

print(f"\n--- Training (learning_rate={learning_rate}, epochs={n_epochs}) ---")
print(f"\n{'Epoch':>7} | {'Loss':>12} | {'m':>8} | {'b':>8}")
print("-" * 45)

for epoch in range(n_epochs):
    # Forward pass: compute predictions
    y_pred = m * x + b

    # Compute loss
    loss = mse_loss(y_pred, y)
    loss_history.append(loss)

    # Compute gradients
    errors  = y_pred - y                  # prediction errors
    grad_m  = (2 / n_samples) * np.dot(errors, x)  # dLoss/dm
    grad_b  = (2 / n_samples) * np.sum(errors)      # dLoss/db

    # Update parameters (gradient descent step)
    m = m - learning_rate * grad_m
    b = b - learning_rate * grad_b

    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"{epoch+1:>7} | {loss:>12.4f} | {m:>8.4f} | {b:>8.4f}")

print(f"\n--- Results ---")
print(f"Learned  m = {m:.4f}   True m = {true_m}")
print(f"Learned  b = {b:.4f}   True b = {true_b}")
print(f"Final loss:  {loss:.4f}")

# --- Step 4: Make predictions on new data ---
new_hours = np.array([1, 3, 5, 7, 9])
predicted_scores = m * new_hours + b

print(f"\n--- Predictions on new data ---")
print(f"{'Hours':>8} | {'Predicted Score':>16} | {'Expected (~true)':>16}")
print("-" * 47)
for h, p in zip(new_hours, predicted_scores):
    expected = true_m * h + true_b  # what the true line would give
    print(f"{h:>8} | {p:>16.1f} | {expected:>16.1f}")

# Show loss decreasing over time
print(f"\n--- Loss over time (sample) ---")
sample_epochs = [0, 50, 100, 200, 300, 400, 499]
print(f"{'Epoch':>7} | {'Loss':>12}")
print("-" * 25)
for ep in sample_epochs:
    print(f"{ep+1:>7} | {loss_history[ep]:>12.4f}")

print("\n" + "=" * 60)
print("PART 5: LEARNING RATE EXPERIMENTS")
print("=" * 60)

# ------------------------------------------------------------------
# The learning rate is a crucial "hyperparameter" in ML.
# We will try three values and see what happens to the loss.
# ------------------------------------------------------------------

def run_gradient_descent(lr, epochs=200, seed=0):
    """Run linear regression with given learning rate, return loss history."""
    np.random.seed(seed)
    m_curr = np.random.randn()
    b_curr = np.random.randn()
    losses = []

    for _ in range(epochs):
        y_pred = m_curr * x + b_curr
        loss   = mse_loss(y_pred, y)
        losses.append(loss)

        errors  = y_pred - y
        grad_m  = (2 / n_samples) * np.dot(errors, x)
        grad_b  = (2 / n_samples) * np.sum(errors)

        m_curr -= lr * grad_m
        b_curr -= lr * grad_b

        # Stop if loss explodes (learning rate too high)
        if loss > 1e8 or np.isnan(loss):
            losses.extend([float('inf')] * (epochs - len(losses)))
            break

    return losses, m_curr, b_curr

learning_rates = {
    "Too small (0.0001)": 0.0001,
    "Just right (0.01) ": 0.01,
    "Too large  (0.5)  ": 0.5,
}

print(f"\n{'Learning Rate':>22} | {'Loss at epoch 1':>16} | {'Loss at epoch 100':>18} | {'Loss at epoch 200':>18} | {'Final m':>8}")
print("-" * 90)

for label, lr in learning_rates.items():
    losses, final_m, final_b = run_gradient_descent(lr, epochs=200)

    loss_1   = losses[0]
    loss_100 = losses[99] if losses[99] != float('inf') else float('inf')
    loss_200 = losses[199] if losses[199] != float('inf') else float('inf')

    l1_str   = f"{loss_1:.2f}"
    l100_str = f"{loss_100:.2f}" if loss_100 != float('inf') else "DIVERGED"
    l200_str = f"{loss_200:.2f}" if loss_200 != float('inf') else "DIVERGED"
    m_str    = f"{final_m:.4f}" if not np.isnan(final_m) else "NaN"

    print(f"{label:>22} | {l1_str:>16} | {l100_str:>18} | {l200_str:>18} | {m_str:>8}")

print("""
What we see:
  - Too small learning rate: loss decreases but very slowly
                             we need many more epochs to converge
  - Just right: loss decreases smoothly and quickly to a low value
  - Too large: loss may explode (diverge) or oscillate wildly
               the steps are so big we jump past the minimum

In practice, finding a good learning rate is part of the art of ML!
Common choices: 0.1, 0.01, 0.001, 0.0001
""")

print("=" * 60)
print("PART 6: GRADIENT DESCENT VARIANTS (PREVIEW)")
print("=" * 60)
print("""
What we implemented above is called "Batch Gradient Descent":
  - Use ALL training examples to compute the gradient each step
  - Accurate but slow for large datasets

In real ML, there are faster variants:
  - Stochastic Gradient Descent (SGD):
      Use ONE random sample per step. Very fast but noisy.

  - Mini-batch Gradient Descent (most common!):
      Use a small batch (e.g., 32 or 64 samples) per step.
      Balance between speed and accuracy.
      This is what frameworks like PyTorch use by default.

  - Adam, RMSProp, Adagrad:
      Smarter optimizers that adapt the learning rate automatically.
      Adam is the most popular optimizer in deep learning.

All of them are built on the same core idea:
  loss -> gradient -> update parameters -> repeat
""")

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key ideas from this file:
  - Loss function : measures how wrong the model is (lower = better)
  - MSE           : (1/n) * sum((prediction - true)^2)
  - Gradient      : derivative of loss w.r.t. each parameter
  - Gradient descent: parameter = parameter - learning_rate * gradient
  - Learning rate : step size -- too big diverges, too small is slow
  - Linear regression: y = m*x + b, find best m and b using GD
  - This exact loop (forward pass -> loss -> gradient -> update) is
    how EVERY neural network trains, including GPT!
""")

# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test MSE loss function
assert mse_loss(np.array([10.0]), np.array([10.0])) == 0.0, "MSE of identical arrays should be 0"
assert mse_loss(np.array([11.0, 9.0]), np.array([10.0, 10.0])) == 1.0, "MSE of errors [1,-1] should be 1"
# good predictions should have lower loss than bad predictions
assert loss_good < loss_bad, "Good predictions should have lower MSE than bad predictions"

# Test simple gradient: f(w) = (w-3)^2, f'(w) = 2*(w-3)
assert simple_gradient(3.0) == 0.0, "Gradient at minimum w=3 should be 0"
assert simple_gradient(5.0) == 4.0, "Gradient at w=5 should be 2*(5-3)=4"
assert simple_gradient(1.0) == -4.0, "Gradient at w=1 should be 2*(1-3)=-4"

# Test simple loss: f(w) = (w-3)^2
assert simple_loss(3.0) == 0.0, "Loss at minimum should be 0"
assert simple_loss(4.0) == 1.0, "Loss at w=4 should be 1"

# After gradient descent, w should be close to 3.0 and loss should be tiny
assert abs(w - 3.0) < 0.2, f"After GD w should be ~3.0, got {w:.4f}"
assert simple_loss(w) < 0.01, "Loss after GD should be very small"

# Test that training reduced the loss (final < initial)
assert loss_history[-1] < initial_loss, "Training should reduce loss"

# Test gradient formulas match expected direction
# If y_pred > y_true, grad_m should be positive (we need to decrease m)
x_test_arr = np.array([1.0, 2.0])
y_true_test = np.array([2.0, 3.0])
y_pred_test = np.array([5.0, 6.0])   # predicted much too high
errors_test  = y_pred_test - y_true_test  # [3.0, 3.0]
grad_m_test = (2 / 2) * np.dot(errors_test, x_test_arr)  # positive
assert grad_m_test > 0, "Gradient for m should be positive when overpredicting"

# Test learned parameters: after 500 epochs, m and b should be close to true values
assert abs(m - true_m) < 2.0, f"Learned m={m:.2f} should be within 2 of true m={true_m}"
assert abs(b - true_b) < 5.0, f"Learned b={b:.2f} should be within 5 of true b={true_b}"

# Test loss monotonically decreased across the broad checkpoints
assert loss_history[0] > loss_history[100] > loss_history[499], "Loss should decrease over training"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
# 1. Change the learning rate in the main training loop to 0.001.
#    How many more epochs does it need to reach a similar final loss?
#
# 2. Change n_samples to 200. Does the model learn better parameters?
#    Why might more data help?
#
# 3. Change noise_level to 0 (no noise). The data is now perfectly linear.
#    How close does gradient descent get to the true m=10, b=20?
#
# 4. What happens if you set learning_rate = 1.0 in the training loop?
#    Run it and observe. This demonstrates learning rate instability.
#
# 5. The gradient for m is: (2/n) * sum((y_pred - y) * x)
#    Write out what each part means in plain English:
#      (y_pred - y) = ?
#      * x          = ?
#      sum()        = ?
#      * 2/n        = ?
#
# 6. FIRST PRINCIPLES EXERCISE: Derive the gradient of MAE (Mean Absolute Error).
#    MAE = (1/n) * SUM |y_pred - y_true|
#    The derivative of |u| is:  sign(u) = +1 if u > 0, -1 if u < 0
#    So: dMAE/dw = (1/n) * SUM sign(y_pred - y_true) * x
#    Why is MAE harder to optimize than MSE?
#    Hint: the gradient of MAE has the same magnitude regardless of how
#    far off the prediction is (sign is +1 or -1). MSE gradient scales
#    with the error magnitude, giving bigger updates for bigger errors.
#    Also, MAE is not differentiable at y_pred = y_true (the kink).
# =============================================================================
