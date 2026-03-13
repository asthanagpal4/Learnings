# HOW TO RUN:
#   uv run python 06_neural_networks/03_backpropagation.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# LESSON 3: BACKPROPAGATION — Teaching the Network to Learn
# =============================================================================
# Run this file with:  uv run python 06_neural_networks/03_backpropagation.py
#
# THE CENTRAL QUESTION OF LEARNING:
# -----------------------------------
# The network made a wrong prediction. We want to fix it.
# But the network has MANY weights. Which ones should we change? By how much?
#
# Backpropagation answers this by asking:
#   "How much did each weight CONTRIBUTE to the error?"
#
# INTUITION: The Chain Rule
# --------------------------
# Imagine a chain of causes:
#   Weight W -> affects neuron activation -> affects next layer -> affects loss
#
# If A causes B, and B causes C, then to know "how does A affect C?"
# we multiply the individual effects:
#   dC/dA = (dC/dB) * (dB/dA)
#
# This is the CHAIN RULE from calculus.
# In a neural network: loss depends on output, which depends on hidden,
# which depends on weights. We apply chain rule backwards through all layers.
#
# "Back" in backpropagation means we compute gradients FROM OUTPUT BACK TO INPUT.
#
# ALGORITHM:
#   Forward pass  -> compute prediction
#   Compute loss  -> how wrong are we?
#   Backward pass -> compute gradient of loss w.r.t. every weight
#   Update step   -> adjust every weight to reduce loss

import numpy as np

np.random.seed(42)

print("=" * 60)
print("LESSON 3: BACKPROPAGATION")
print("=" * 60)


# =============================================================================
# FIRST PRINCIPLES: THE CHAIN RULE FROM SCRATCH
# =============================================================================
# The chain rule is the foundation of backpropagation. Here it is from scratch:
#
# If y = f(g(x)), meaning we first apply g to x, then apply f to the result:
#   dy/dx = dy/dg * dg/dx
#
# Intuition: A small change dx in x causes a small change dg in g(x).
# That change dg then causes a small change dy in f(g(x)).
# The TOTAL effect on y is the PRODUCT of these two effects.
#
# Example: y = (3x + 2)^2
#   Let g(x) = 3x + 2, so y = g^2
#   dg/dx = 3       (how fast does g change with x?)
#   dy/dg = 2g      (how fast does y change with g?)
#   dy/dx = 2g * 3 = 6(3x + 2)   (total effect: product of both)
#
# In a neural network with 3 layers:
#   Loss depends on output (a2), which depends on hidden (a1), which depends
#   on weights (W1). So:
#   d(Loss)/d(W1) = d(Loss)/d(a2) * d(a2)/d(a1) * d(a1)/d(W1)
#   Each factor is computed at its own layer, then multiplied together.
#   This is why we go BACKWARDS: compute d(Loss)/d(a2) first, then propagate.
#
# =============================================================================
# FIRST PRINCIPLES: SIGMOID DERIVATIVE (Step by Step Algebraic Derivation)
# =============================================================================
# Goal: find d/dx of sigma(x) = 1 / (1 + e^(-x))
#
# Step 1: Rewrite as sigma = (1 + e^(-x))^(-1)
#   Let u = 1 + e^(-x), so sigma = u^(-1)
#
# Step 2: Apply chain rule:
#   d(sigma)/dx = d(sigma)/du * du/dx
#
# Step 3: Compute d(sigma)/du:
#   d(u^(-1))/du = -u^(-2) = -1 / (1 + e^(-x))^2
#
# Step 4: Compute du/dx:
#   d(1 + e^(-x))/dx = -e^(-x)    (derivative of e^(-x) is -e^(-x))
#
# Step 5: Multiply:
#   d(sigma)/dx = [-1 / (1 + e^(-x))^2] * [-e^(-x)]
#               = e^(-x) / (1 + e^(-x))^2
#
# Step 6: Factor cleverly:
#   = [1 / (1 + e^(-x))] * [e^(-x) / (1 + e^(-x))]
#   = sigma(x) * [e^(-x) / (1 + e^(-x))]
#
# Step 7: Notice that 1 - sigma(x) = 1 - 1/(1+e^(-x)) = e^(-x)/(1+e^(-x))
#   So: d(sigma)/dx = sigma(x) * (1 - sigma(x))
#
# This is the beautiful result: the derivative of sigmoid can be computed
# from the sigmoid output itself! No need to recompute anything.
#
# =============================================================================
# FIRST PRINCIPLES: WEIGHT UPDATE FORMULAS (Chain Rule Expanded)
# =============================================================================
# For a 2-layer network: input -> hidden (W1) -> output (W2) -> loss
#
# OUTPUT LAYER WEIGHTS (W2):
#   d(Loss)/d(W2) = d(Loss)/d(output) * d(output)/d(W2)
#   Only one chain link: output directly depends on W2.
#
# HIDDEN LAYER WEIGHTS (W1):
#   d(Loss)/d(W1) = d(Loss)/d(output) * d(output)/d(hidden) * d(hidden)/d(W1)
#   Two chain links: error must propagate through the output layer first.
#
# For a 3-LAYER NETWORK (input -> h1 -> h2 -> output -> loss):
#   d(Loss)/d(W1) = d(Loss)/d(out) * d(out)/d(h2) * d(h2)/d(h1) * d(h1)/d(W1)
#   Three chain links! The chain gets LONGER for earlier layers.
#   This is why vanishing gradients are worse for deeper networks.
# =============================================================================

# =============================================================================
# PART 1: ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# =============================================================================
# To use the chain rule, we need the DERIVATIVE of each function.
# The derivative tells us: "if I change x by a tiny bit, how much does f(x) change?"
#
# SIGMOID derivative (derived step-by-step above):
#   sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
#
# This is a beautiful property — we can reuse the sigmoid value we already computed!
# If a = sigmoid(z), then da/dz = a * (1 - a).

def sigmoid(x):
    """Smooth S-shaped activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(a):
    """
    Derivative of sigmoid. Input 'a' is already the sigmoid OUTPUT.
    Formula: a * (1 - a)
    This tells us: how sensitive is the neuron output to changes in its input?
    """
    return a * (1.0 - a)


print("\n--- Part 1: Sigmoid and Its Derivative ---")
print("  Note: when a is near 0 or 1 (saturated), the derivative is SMALL.")
print("  Small derivative = small gradient = slow learning (vanishing gradient problem)")
print()
print("  a (sigmoid output) | sigmoid_derivative(a)")
print("  " + "-" * 40)
for a_val in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    print(f"        {a_val:.2f}           |      {sigmoid_derivative(a_val):.4f}")
print("  (Maximum derivative is 0.25 when a=0.5 — at the middle of the S-curve)")


# =============================================================================
# PART 2: LOSS FUNCTION — MEASURING HOW WRONG WE ARE
# =============================================================================
# We need a single number that measures "how bad is the prediction?"
# This is the LOSS (or cost, or error).
#
# Mean Squared Error (MSE):
#   loss = mean( (predicted - actual)^2 )
#
# Why squared? Two reasons:
#   1. Always positive (we don't want errors to cancel out)
#   2. Penalizes big errors more than small ones (3^2=9 >> 1^2=1)
#
# The derivative of MSE w.r.t. output a2:
#   d(loss)/d(a2) = 2 * (a2 - y) / n    (or simplified: (a2 - y))
# This tells us: if prediction > actual, gradient is positive -> decrease a2

def mse_loss(y_pred, y_true):
    """Mean Squared Error loss."""
    return np.mean((y_pred - y_true) ** 2)


print("\n--- Part 2: Mean Squared Error Loss ---")
examples = [(0.9, 1), (0.5, 1), (0.1, 1), (0.9, 0)]
print("  predicted | actual | loss   | interpretation")
print("  " + "-" * 55)
for pred, actual in examples:
    loss = (pred - actual) ** 2
    interp = "almost right" if loss < 0.05 else ("somewhat wrong" if loss < 0.25 else "very wrong")
    print(f"    {pred:.1f}     |   {actual}    | {loss:.3f}  | {interp}")


# =============================================================================
# PART 3: BACKPROPAGATION — STEP BY STEP FOR XOR
# =============================================================================
# Network architecture:
#   Input (2) -> Hidden (2) -> Output (1)
#
# Variable naming convention:
#   W1, b1 = weights/biases of hidden layer
#   W2, b2 = weights/biases of output layer
#   z1 = hidden layer pre-activation (before sigmoid)
#   a1 = hidden layer activation (after sigmoid)
#   z2 = output pre-activation
#   a2 = output activation = final prediction
#
# FORWARD PASS:
#   z1 = x  @ W1 + b1
#   a1 = sigmoid(z1)
#   z2 = a1 @ W2 + b2
#   a2 = sigmoid(z2)
#   loss = mean((a2 - y)^2)
#
# BACKWARD PASS (chain rule applied in reverse):
#   d_loss/d_a2 = (a2 - y)                    [derivative of MSE]
#   d_a2/d_z2   = sigmoid'(a2) = a2*(1-a2)    [derivative of sigmoid]
#
#   delta2 = (a2 - y) * a2 * (1 - a2)         [output layer error signal]
#
#   d_loss/d_W2 = a1.T @ delta2                [how W2 affects loss]
#   d_loss/d_b2 = sum(delta2)                  [how b2 affects loss]
#
#   delta1 = (delta2 @ W2.T) * a1 * (1 - a1)  [hidden layer error signal]
#            ^propagate back   ^sigmoid deriv
#
#   d_loss/d_W1 = x.T @ delta1                 [how W1 affects loss]
#   d_loss/d_b1 = sum(delta1)                  [how b1 affects loss]
#
# UPDATE (gradient descent):
#   W2 -= learning_rate * d_loss/d_W2
#   b2 -= learning_rate * d_loss/d_b2
#   W1 -= learning_rate * d_loss/d_W1
#   b1 -= learning_rate * d_loss/d_b1
#
# Key insight: "delta" = how much this layer contributed to the error.
# We compute delta at the output, then PROPAGATE it backwards layer by layer.

def forward(x, W1, b1, W2, b2):
    """Forward pass. Returns all intermediate values (needed for backprop)."""
    z1 = x @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def backward(x, y, a1, a2, W1, b1, W2, b2):
    """
    Backward pass. Computes gradients for all weights using chain rule.

    Returns: gradients dW1, db1, dW2, db2
    """
    m = x.shape[0]  # number of samples in batch

    # ---- OUTPUT LAYER ----
    # How much does the loss change as we change a2?
    # d_loss/d_a2 = (a2 - y)
    # But a2 = sigmoid(z2), so d_a2/d_z2 = a2 * (1 - a2)
    # Chain rule: d_loss/d_z2 = d_loss/d_a2 * d_a2/d_z2
    delta2 = (a2 - y) * sigmoid_derivative(a2)    # shape: (m, 1)

    # Gradient of W2: how much does loss change per unit change in W2?
    # d_loss/d_W2 = a1.T @ delta2
    dW2 = a1.T @ delta2 / m                        # shape: (n_hidden, 1)
    db2 = np.mean(delta2, axis=0)                   # shape: (1,)

    # ---- HIDDEN LAYER ----
    # Propagate the error signal BACK through W2
    # "If output error is delta2, and W2 connects hidden->output,
    #  how much did each hidden neuron contribute to this error?"
    delta1 = (delta2 @ W2.T) * sigmoid_derivative(a1)  # shape: (m, n_hidden)

    # Gradient of W1
    dW1 = x.T @ delta1 / m                         # shape: (n_input, n_hidden)
    db1 = np.mean(delta1, axis=0)                   # shape: (n_hidden,)

    return dW1, db1, dW2, db2


# =============================================================================
# PART 4: TRACE ONE BACKWARD PASS BY HAND
# =============================================================================
print("\n--- Part 3 & 4: Tracing One Backward Pass ---")

# Initialize network
W1 = np.random.randn(2, 2) * 0.5
b1 = np.zeros(2)
W2 = np.random.randn(2, 1) * 0.5
b2 = np.zeros(1)

# One XOR sample
x_sample = np.array([[0.0, 1.0]])   # shape (1, 2) — batch of 1
y_sample = np.array([[1.0]])         # expected output

print()
print("  Sample: input=[0,1], expected output=1")
print()

# Forward
z1, a1, z2, a2 = forward(x_sample, W1, b1, W2, b2)
loss = mse_loss(a2, y_sample)

print(f"  FORWARD PASS:")
print(f"    z1 = x @ W1 + b1 = {z1.round(4)}")
print(f"    a1 = sigmoid(z1) = {a1.round(4)}  <- hidden neuron outputs")
print(f"    z2 = a1 @ W2 + b2 = {z2.round(4)}")
print(f"    a2 = sigmoid(z2) = {a2.round(4)}  <- prediction")
print(f"    loss = (a2 - y)^2 = {loss:.4f}")
print()

# Backward
dW1, db1_grad, dW2, db2_grad = backward(x_sample, y_sample, a1, a2, W1, b1, W2, b2)

print(f"  BACKWARD PASS (gradients):")
delta2 = (a2 - y_sample) * sigmoid_derivative(a2)
print(f"    delta2 (output error signal) = {delta2.round(4)}")
print(f"    dW2 (how loss changes with W2) = {dW2.round(4)}")
print(f"    db2 = {db2_grad.round(4)}")
delta1 = (delta2 @ W2.T) * sigmoid_derivative(a1)
print(f"    delta1 (hidden error signal) = {delta1.round(4)}")
print(f"    dW1 (how loss changes with W1) =")
print(f"      {dW1.round(4)}")
print(f"    db1 = {db1_grad.round(4)}")
print()
print("  UPDATE (learning_rate=0.1, we subtract the gradient):")
lr = 0.1
print(f"    W2 before: {W2.T.round(4)}")
W2_new = W2 - lr * dW2
print(f"    W2 after:  {W2_new.T.round(4)}")
print(f"    (W2 changed — the network just took a tiny step toward correct!)")


# =============================================================================
# PART 5: TRAIN ON XOR — WATCH IT WORK!
# =============================================================================
# Now let's put it all together and actually train the network on XOR.
# Remember: a single perceptron FAILED on XOR (Lesson 1).
# With 2 layers and backprop, it should succeed!

def train_network(X, y, n_hidden=4, learning_rate=0.5, epochs=5000, print_every=500):
    """
    Full training loop: forward -> loss -> backward -> update.
    Returns trained weights.
    """
    n_input  = X.shape[1]
    n_output = y.shape[1]

    # Initialize weights (small random values)
    W1 = np.random.randn(n_input, n_hidden) * 0.5
    b1 = np.zeros(n_hidden)
    W2 = np.random.randn(n_hidden, n_output) * 0.5
    b2 = np.zeros(n_output)

    loss_history = []

    for epoch in range(epochs):
        # FORWARD PASS
        z1, a1, z2, a2 = forward(X, W1, b1, W2, b2)

        # LOSS
        loss = mse_loss(a2, y)
        loss_history.append(loss)

        # BACKWARD PASS
        dW1, db1_g, dW2, db2_g = backward(X, y, a1, a2, W1, b1, W2, b2)

        # UPDATE WEIGHTS (gradient descent)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1_g
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2_g

        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            preds = (a2 >= 0.5).astype(int)
            accuracy = np.mean(preds == y) * 100
            print(f"  Epoch {epoch+1:5d} | Loss: {loss:.4f} | Accuracy: {accuracy:.0f}%")

    return W1, b1, W2, b2, loss_history


print("\n--- Part 5: Training a 2-Layer Network on XOR ---")
print("  (A single perceptron FAILED this in Lesson 1 — now watch it succeed!)")
print()

X_xor = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
y_xor = np.array([[0.0], [1.0], [1.0], [0.0]])

W1f, b1f, W2f, b2f, losses = train_network(
    X_xor, y_xor,
    n_hidden=4,
    learning_rate=0.5,
    epochs=5000,
    print_every=1000
)

# Final predictions
_, _, _, final_preds = forward(X_xor, W1f, b1f, W2f, b2f)
print()
print("  Final Predictions vs Expected:")
print("  Input A | Input B | Prediction | Expected | Correct?")
print("  " + "-" * 52)
for i in range(len(X_xor)):
    pred_raw = final_preds[i, 0]
    pred_rounded = 1 if pred_raw >= 0.5 else 0
    expected = int(y_xor[i, 0])
    ok = "YES" if pred_rounded == expected else "NO"
    print(f"    {int(X_xor[i,0])}       {int(X_xor[i,1])}       {pred_raw:.4f}     "
          f"    {expected}       {ok}")

print()
print(f"  Initial loss: {losses[0]:.4f}  ->  Final loss: {losses[-1]:.4f}")
print(f"  Loss decreased by: {(1 - losses[-1]/losses[0])*100:.1f}%")
print()
print("  XOR is SOLVED! The 2-layer network learned what a single neuron cannot.")


# =============================================================================
# PART 6: WHY DOES BACKPROP WORK? (INTUITION)
# =============================================================================
print("\n--- Part 6: Why Does Backprop Work? ---")
print("""
  Think of it like adjusting a recipe:

  Imagine you made a cake and it was too sweet.
  - The taste depends on: sugar amount, baking time, mixing order...
  - You want to know: WHICH ingredients contributed MOST to the sweetness?
  - Backprop does exactly this: it traces the error back to find which
    "ingredients" (weights) contributed most to the wrong answer.
  - Weights that contributed a lot get changed a lot.
  - Weights that had little effect get changed a little.

  The gradient tells you: "how much does the loss change if I nudge this weight?"
  Subtracting the gradient (gradient DESCENT) moves in the downhill direction.
  Repeat thousands of times -> the network finds good weights!
""")


# =============================================================================
# KEY CONCEPTS SUMMARY
# =============================================================================
print("=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
Chain rule:     d(loss)/d(W) = d(loss)/d(output) * d(output)/d(W)
                Applied layer by layer, backwards.

Gradient:       The derivative of the loss w.r.t. a weight.
                Tells us: "which direction increases the loss?"
                We go in the OPPOSITE direction (gradient descent).

Delta:          The error signal at each layer.
                delta_output = (pred - true) * sigmoid'(pred)
                delta_hidden = (delta_output @ W_next.T) * sigmoid'(hidden)

Gradient descent: weight -= learning_rate * gradient
                  Small steps in the direction that reduces loss.

Vanishing grad: When sigmoid is saturated (output near 0 or 1),
                the gradient is near 0 -> weights barely update.
                This is why modern networks use ReLU instead of sigmoid.
                (We'll cover ReLU in future lessons!)
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- sigmoid derivative formula: a*(1-a) ---
for a_v in [0.1, 0.5, 0.9]:
    assert np.isclose(sigmoid_derivative(a_v), a_v * (1 - a_v)), \
        f"sigmoid_derivative({a_v}) should equal {a_v*(1-a_v)}"

# Maximum derivative is 0.25 at a=0.5
assert np.isclose(sigmoid_derivative(0.5), 0.25), "Max derivative should be 0.25 at a=0.5"

# --- mse_loss: perfect prediction gives 0 loss ---
y_perfect = np.array([[0.0], [1.0], [1.0], [0.0]])
assert np.isclose(mse_loss(y_perfect, y_perfect), 0.0), "Perfect prediction should have 0 loss"

# mse_loss is always non-negative
y_wrong = np.array([[1.0], [0.0], [0.0], [1.0]])
assert mse_loss(y_wrong, y_perfect) > 0, "Non-perfect prediction should have positive loss"

# --- forward pass output is a valid sigmoid output (in (0,1)) ---
x_t = np.array([[1.0, 0.0]])
y_t = np.array([[1.0]])
W1_t = np.random.randn(2, 2) * 0.5
b1_t = np.zeros(2)
W2_t = np.random.randn(2, 1) * 0.5
b2_t = np.zeros(1)
_, _, _, a2_t = forward(x_t, W1_t, b1_t, W2_t, b2_t)
assert a2_t.shape == (1, 1), f"Output shape should be (1,1), got {a2_t.shape}"
assert 0.0 < a2_t[0, 0] < 1.0, "Output should be in (0,1)"

# --- training reduces loss: final loss must be less than initial loss ---
assert losses[-1] < losses[0], \
    f"Training should reduce loss: initial={losses[0]:.4f}, final={losses[-1]:.4f}"

# --- loss history has the right length ---
assert len(losses) == 5000, f"Loss history length should be 5000, got {len(losses)}"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("=" * 60)
print("EXERCISES — Try these yourself!")
print("=" * 60)
print("""
Exercise 1: Watch the Loss Decrease
  Change print_every=100 and epochs=1000.
  Print the loss every 100 epochs and observe: does it decrease smoothly?
  Are there any big jumps or plateaus?

Exercise 2: Effect of Learning Rate
  Try learning_rate = 0.01 (slow), 0.1, 0.5, 2.0 (fast).
  For each, note: how many epochs to reach loss < 0.05?
  What happens with learning_rate = 5.0? (hint: loss might INCREASE!)

Exercise 3: Effect of Hidden Layer Size
  Try n_hidden = 2, 4, 8, 16 for XOR.
  Does a bigger hidden layer train faster or slower?
  Does it always reach 100% accuracy?

Exercise 4: Train on AND and OR
  XOR is the hardest. Try training on AND and OR:
  y_and = np.array([[0],[0],[0],[1]])
  y_or  = np.array([[0],[1],[1],[1]])
  Do they train faster than XOR? Why might that be?

Exercise 5: Plot the Loss (challenge)
  The losses list contains the loss at every epoch.
  Print every 10th loss value to see the learning curve.
  Does loss decrease fast at first and slow down later?
  This is typical of gradient descent — called "exponential decay" of loss.

Exercise 6 (First Principles): Derive Gradient for a 3-Layer Network
  For a 3-layer network: input -> h1 (W1) -> h2 (W2) -> output (W3) -> loss
  Write out the chain rule for d(Loss)/d(W1):

  d(Loss)/d(W1) = d(Loss)/d(a3)     ... how loss changes with output
                * d(a3)/d(z3)        ... sigmoid derivative at output
                * d(z3)/d(a2)        ... = W3 (weights connecting h2->output)
                * d(a2)/d(z2)        ... sigmoid derivative at h2
                * d(z2)/d(a1)        ... = W2 (weights connecting h1->h2)
                * d(a1)/d(z1)        ... sigmoid derivative at h1
                * d(z1)/d(W1)        ... = input x

  Notice: 6 terms multiplied together! For a 2-layer network it was 4 terms.
  Each additional layer adds 2 more terms (one for weights, one for activation).
  If each sigmoid derivative is at most 0.25, then after 3 layers:
    gradient scale <= 0.25^3 = 0.0156
  This is the vanishing gradient problem: gradients shrink exponentially
  with depth, making early layers learn VERY slowly.
""")
