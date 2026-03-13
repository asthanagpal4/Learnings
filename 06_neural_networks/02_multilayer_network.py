# HOW TO RUN:
#   uv run python 06_neural_networks/02_multilayer_network.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# LESSON 2: MULTI-LAYER NETWORKS — Solving XOR and Beyond
# =============================================================================
# Run this file with:  uv run python 06_neural_networks/02_multilayer_network.py
#
# WHY DO WE NEED MULTIPLE LAYERS?
# --------------------------------
# In Lesson 1, we saw that a single perceptron draws ONE straight line.
# It cannot solve XOR because XOR is not "linearly separable."
#
# But what if we use MULTIPLE neurons, each drawing its own line,
# and then combine their outputs?
#
# A multi-layer network is like a committee of committees:
#   Layer 1 (hidden): several neurons each look at the raw inputs
#   Layer 2 (output): one neuron looks at what layer 1 found
#
# With enough layers and neurons, we can solve ANY pattern!
# (This is the famous "universal approximation theorem")
#
# STRUCTURE OF A 2-LAYER NETWORK (for XOR):
#
#   Input layer:    2 neurons (the two XOR inputs)
#        |
#        v
#   Hidden layer:   2 neurons (they find intermediate patterns)
#        |
#        v
#   Output layer:   1 neuron (the final answer)
#
# Note: "2-layer network" usually means 2 layers of WEIGHTS
#       (input layer is just data, not counted as a real layer)

import numpy as np

np.random.seed(42)

print("=" * 60)
print("LESSON 2: MULTI-LAYER NETWORKS")
print("=" * 60)


# =============================================================================
# PART 1: THE SIGMOID ACTIVATION FUNCTION
# =============================================================================
# In Lesson 1 we used the step function (0 or 1 — like a light switch).
# Problem: the step function has NO gradient (it's flat everywhere except
# at 0 where it jumps). That makes learning impossible in deep networks.
#
# Solution: the SIGMOID function!
#
#   sigmoid(x) = 1 / (1 + e^(-x))
#
# Properties:
#   - Output is always between 0 and 1
#   - Smooth and S-shaped (no sudden jumps)
#   - Differentiable everywhere — this is CRITICAL for backpropagation
#   - When x is very negative -> output near 0
#   - When x is very positive -> output near 1
#   - When x = 0 -> output = 0.5 (exactly in the middle)
#
# Think of it as a "soft" version of the step function.

def sigmoid(x):
    """Smooth S-shaped function. Output always between 0 and 1."""
    return 1.0 / (1.0 + np.exp(-x))


print("\n--- Part 1: Sigmoid Activation ---")
test_values = [-10, -2, -1, 0, 1, 2, 10]
print("  x value  |  sigmoid(x)  |  interpretation")
print("  " + "-" * 45)
for x in test_values:
    s = sigmoid(x)
    if s < 0.2:
        interp = "strongly OFF (near 0)"
    elif s > 0.8:
        interp = "strongly ON  (near 1)"
    else:
        interp = "in between"
    print(f"  {x:6}   |   {s:.4f}     |  {interp}")


# =============================================================================
# PART 2: WEIGHTS AS MATRICES
# =============================================================================
# Here's the key insight that makes neural networks fast:
# We can process ALL neurons in a layer at once using MATRIX MULTIPLICATION!
#
# For a layer with:
#   - n_in  = number of inputs  (e.g. 2)
#   - n_out = number of neurons (e.g. 2)
#
# Weights matrix W has shape (n_in, n_out):
#   W[i][j] = weight connecting input i to neuron j
#
# Bias vector b has shape (n_out,):
#   b[j] = bias of neuron j
#
# Forward pass for one sample:
#   z = input @ W + b      (matrix multiply, then add bias)
#   a = sigmoid(z)         (apply activation to every element)
#
# For a BATCH of m samples, input has shape (m, n_in):
#   z = inputs @ W + b     (numpy broadcasts b automatically!)
#   a = sigmoid(z)
#
# This is why numpy is so important for neural networks — one line of code
# processes ALL neurons in ALL samples at the same time!

print("\n--- Part 2: Weights as Matrices ---")
print("  For a layer: 2 inputs -> 3 neurons:")
print()
W_example = np.array([[0.1, 0.2, 0.3],    # weights from input 0 to all 3 neurons
                       [0.4, 0.5, 0.6]])   # weights from input 1 to all 3 neurons
b_example = np.array([0.0, 0.0, 0.0])     # biases for all 3 neurons

print(f"  W shape: {W_example.shape}  (rows=inputs, cols=neurons)")
print(f"  b shape: {b_example.shape}  (one bias per neuron)")
print()
print("  Weight matrix W:")
print("            Neuron0  Neuron1  Neuron2")
print(f"  Input0:   {W_example[0]}")
print(f"  Input1:   {W_example[1]}")

sample_input = np.array([1.0, 0.5])
z = sample_input @ W_example + b_example
a = sigmoid(z)
print()
print(f"  Input: {sample_input}")
print(f"  z = input @ W + b = {z.round(4)}")
print(f"  a = sigmoid(z)    = {a.round(4)}")
print("  We computed ALL 3 neurons at once with one matrix multiply!")


# =============================================================================
# PART 3: BUILD A 2-LAYER NETWORK STRUCTURE
# =============================================================================
# Let's build a network to solve XOR:
#   Input:  2 values (x1, x2)
#   Hidden: 2 neurons
#   Output: 1 neuron
#
# Shapes:
#   W1: (2, 2)  — 2 inputs  -> 2 hidden neurons
#   b1: (2,)    — bias for each hidden neuron
#   W2: (2, 1)  — 2 hidden neurons -> 1 output neuron
#   b2: (1,)    — bias for the output neuron

print("\n--- Part 3: Building a 2-layer Network Structure ---")

# Network dimensions
n_input  = 2   # XOR has 2 inputs
n_hidden = 2   # we'll use 2 hidden neurons
n_output = 1   # XOR has 1 output

# Initialize weights with small random values
# (We'll explain why "small" matters in later lessons — large initial weights
#  can cause the sigmoid to saturate and gradients to vanish)
W1 = np.random.randn(n_input, n_hidden) * 0.5    # shape (2, 2)
b1 = np.zeros(n_hidden)                           # shape (2,)
W2 = np.random.randn(n_hidden, n_output) * 0.5   # shape (2, 1)
b2 = np.zeros(n_output)                           # shape (1,)

print(f"  W1 shape: {W1.shape}  (connects input -> hidden)")
print(f"  b1 shape: {b1.shape}  (bias for hidden neurons)")
print(f"  W2 shape: {W2.shape}  (connects hidden -> output)")
print(f"  b2 shape: {b2.shape}  (bias for output neuron)")
print()
print("  W1 (initial random values):")
print(f"    {W1.round(3)}")
print("  W2 (initial random values):")
print(f"    {W2.round(3)}")

total_params = W1.size + b1.size + W2.size + b2.size
print(f"\n  Total parameters (weights + biases): {total_params}")
print("  (Modern LLMs have billions of parameters — same idea, much bigger!)")


# =============================================================================
# PART 4: THE FORWARD PASS — STEP BY STEP
# =============================================================================
# The forward pass = feeding input through the network to get a prediction.
# Each step is simple: multiply, add bias, apply activation.
#
# For a single input x:
#   Step 1: z1 = x  @ W1 + b1   (hidden layer pre-activation)
#   Step 2: a1 = sigmoid(z1)    (hidden layer activation)
#   Step 3: z2 = a1 @ W2 + b2   (output layer pre-activation)
#   Step 4: a2 = sigmoid(z2)    (output — our prediction!)

def forward_pass(x, W1, b1, W2, b2, verbose=False):
    """
    Run one forward pass through the 2-layer network.

    x: input array, shape (n_input,) for single sample
       or shape (m, n_input) for a batch of m samples

    Returns: final output (prediction)
    """
    # Hidden layer
    z1 = x @ W1 + b1      # pre-activation (weighted sum + bias)
    a1 = sigmoid(z1)       # activation (apply sigmoid to each element)

    # Output layer
    z2 = a1 @ W2 + b2      # pre-activation
    a2 = sigmoid(z2)       # activation (final output)

    if verbose:
        print(f"    Input x:        {x}")
        print(f"    z1 = x@W1+b1:  {z1.round(4)}")
        print(f"    a1 = sig(z1):  {a1.round(4)}")
        print(f"    z2 = a1@W2+b2: {z2.round(4)}")
        print(f"    a2 = sig(z2):  {a2.round(4)}  <- PREDICTION")

    return a1, a2


print("\n--- Part 4: Forward Pass Step-by-Step ---")
print("  Let's trace one sample through the network:")
print("  Input: [0, 1] (expected XOR output: 1)")
print()
a1_out, a2_out = forward_pass(np.array([0.0, 1.0]), W1, b1, W2, b2, verbose=True)


# =============================================================================
# PART 5: INFORMATION FLOW — ALL XOR INPUTS
# =============================================================================
# Let's run ALL four XOR inputs through the untrained network.
# The predictions will be wrong (random weights!) but we can see
# how information flows through each layer.

print("\n--- Part 5: Information Flow Through the Network ---")
print("  (Network is UNTRAINED — predictions will be random)")
print()

X_xor = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
y_xor = np.array([0, 1, 1, 0])

print("  Input  | Hidden Layer Activations | Output | Expected")
print("  " + "-" * 55)
for i in range(len(X_xor)):
    a1, a2 = forward_pass(X_xor[i], W1, b1, W2, b2)
    pred_rounded = 1 if a2[0] >= 0.5 else 0
    print(f"  {X_xor[i]}  | {a1.round(3)}       | {a2[0]:.3f}  |   {y_xor[i]}")

print()
print("  The hidden layer creates NEW features (combinations of inputs)")
print("  that the output layer can separate with a straight line!")
print("  (The network needs training to find useful hidden features — see Lesson 3)")


# =============================================================================
# FIRST PRINCIPLES: FORWARD PASS COMPLEXITY
# =============================================================================
# For a network with layers of sizes [n_in, n_h1, n_h2, ..., n_out]:
#
# Each layer pair does a matrix multiply:
#   Layer i -> Layer i+1: O(n_i * n_{i+1}) multiply-add operations
#
# Total forward pass cost:
#   O(n_in * n_h1 + n_h1 * n_h2 + ... + n_hk * n_out)
#
# This is DOMINATED by the largest layer pair. If one layer has 1000 neurons
# and the next has 500, that one multiply (500,000 ops) dominates everything.
#
# Example for our XOR network [2, 2, 1]:
#   Layer 1->2: 2 * 2 = 4 multiply-adds
#   Layer 2->3: 2 * 1 = 2 multiply-adds
#   Total: 6 multiply-adds per sample
#
# =============================================================================
# FIRST PRINCIPLES: UNIVERSAL APPROXIMATION THEOREM (Intuition)
# =============================================================================
# Theorem (Cybenko 1989, Hornik 1991):
#   A single hidden layer with enough neurons can approximate ANY continuous
#   function on a compact (bounded, closed) set, to any desired accuracy.
#
# Intuition: Think of each hidden neuron as a "bump detector."
#   - Each neuron with sigmoid activation creates a smooth step in one direction.
#   - Two neurons can create a "bump" (step up + step down).
#   - Many bumps at different positions and heights can approximate any shape,
#     like building a landscape out of LEGO bricks.
#   - More neurons = more bricks = finer approximation.
#
# Caveat: The theorem says the network CAN approximate anything, but does NOT
# say it's easy to FIND the right weights. Training (backprop + gradient descent)
# might get stuck in local minima. Deeper networks often work better in practice.
#
# =============================================================================
# SPACE COMPLEXITY
# =============================================================================
# For layers [n_0, n_1, n_2, ..., n_L]:
#   Weights: sum of n_i * n_{i+1} for all adjacent layer pairs
#   Biases:  sum of n_i for i = 1 to L (one bias per neuron, excluding input)
#   Total parameters = sum(n_i * n_{i+1}) + sum(n_{i+1})
#
# Our XOR network [2, 2, 1]:
#   Weights: 2*2 + 2*1 = 4 + 2 = 6
#   Biases:  2 + 1 = 3
#   Total: 9 parameters
# =============================================================================

print("\n--- First Principles: Complexity Analysis ---")
# Demonstrate for our XOR network
layers_xor = [n_input, n_hidden, n_output]
total_weights = sum(layers_xor[i] * layers_xor[i+1] for i in range(len(layers_xor)-1))
total_biases = sum(layers_xor[i+1] for i in range(len(layers_xor)-1))
flops = sum(layers_xor[i] * layers_xor[i+1] for i in range(len(layers_xor)-1))
print(f"  XOR network {layers_xor}:")
print(f"    Forward pass FLOPs per sample: {flops} multiply-adds")
print(f"    Weights: {total_weights}, Biases: {total_biases}, Total params: {total_weights + total_biases}")
print(f"    Space: {(total_weights + total_biases) * 8} bytes (float64)")
print()

# Exercise: calculate for [784, 128, 64, 10]
layers_exercise = [784, 128, 64, 10]
ex_weights = sum(layers_exercise[i] * layers_exercise[i+1] for i in range(len(layers_exercise)-1))
ex_biases = sum(layers_exercise[i+1] for i in range(len(layers_exercise)-1))
ex_flops = sum(layers_exercise[i] * layers_exercise[i+1] for i in range(len(layers_exercise)-1))
print(f"  Exercise network {layers_exercise} (MNIST-like):")
print(f"    Weights: {ex_weights}")
print(f"    Biases:  {ex_biases}")
print(f"    Total parameters: {ex_weights + ex_biases}")
print(f"    Forward pass FLOPs per sample: {ex_flops} multiply-adds")
print(f"    Memory for parameters: {(ex_weights + ex_biases) * 8 / 1024:.1f} KB (float64)")


# =============================================================================
# PART 6: BATCH PROCESSING — WHY IT'S FASTER
# =============================================================================
# Instead of processing one sample at a time, we can feed the ENTIRE
# dataset as a matrix and get all predictions at once!

print("\n--- Part 6: Batch Processing ---")
print("  Processing all 4 XOR inputs at once (batch forward pass):")
print()

# X_xor has shape (4, 2) — 4 samples, 2 features each
a1_batch, a2_batch = forward_pass(X_xor, W1, b1, W2, b2)

print(f"  Input matrix shape:         {X_xor.shape}")
print(f"  Hidden activations shape:   {a1_batch.shape}")
print(f"  Output activations shape:   {a2_batch.shape}")
print()
print("  All 4 predictions in one shot:")
for i in range(len(X_xor)):
    pred_rounded = 1 if a2_batch[i, 0] >= 0.5 else 0
    print(f"    Input {X_xor[i]} -> hidden {a1_batch[i].round(3)} -> output {a2_batch[i,0]:.3f} -> pred: {pred_rounded}")

print()
print("  Batch processing is the same math — numpy handles the matrix sizes!")
print("  In real training, batches of 32, 64, or 256 samples are common.")


# =============================================================================
# KEY CONCEPTS SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
Hidden layer:       Neurons between input and output. They learn to
                    represent intermediate features of the data.

Sigmoid:            Smooth activation function, output in (0,1).
                    Critical property: it's differentiable (smooth curve).

Weight matrix:      All weights connecting two layers, stored as a 2D array.
                    Shape: (n_inputs_from_prev_layer, n_neurons_in_this_layer)

Bias vector:        One bias per neuron in the layer.

Forward pass:       Feeding input through all layers to get a prediction.
                    z = input @ W + b   (linear combination)
                    a = sigmoid(z)      (activation)

Batch processing:   Process multiple samples at once using matrix math.
                    Much faster than a Python loop!

Universal approx.:  A network with enough hidden neurons can approximate
                    ANY function — this is why deep learning is so powerful.
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- sigmoid output is always between 0 and 1 ---
for x_val in [-10, -1, 0, 1, 10]:
    s = sigmoid(x_val)
    assert 0.0 < s < 1.0, f"sigmoid({x_val}) should be in (0,1), got {s}"

# sigmoid(0) == 0.5 exactly
assert np.isclose(sigmoid(0), 0.5), "sigmoid(0) should be 0.5"

# --- weight matrix shapes ---
assert W1.shape == (n_input, n_hidden), f"W1 shape should be ({n_input},{n_hidden})"
assert b1.shape == (n_hidden,),         f"b1 shape should be ({n_hidden},)"
assert W2.shape == (n_hidden, n_output),f"W2 shape should be ({n_hidden},{n_output})"
assert b2.shape == (n_output,),         f"b2 shape should be ({n_output},)"

# --- single-sample forward pass output shapes ---
x_test_single = np.array([1.0, 0.0])
a1_s, a2_s = forward_pass(x_test_single, W1, b1, W2, b2)
assert a1_s.shape == (n_hidden,),  f"Hidden output shape wrong: {a1_s.shape}"
assert a2_s.shape == (n_output,),  f"Output shape wrong: {a2_s.shape}"

# --- batch forward pass output shapes ---
# X_xor has shape (4, 2)
a1_b, a2_b = forward_pass(X_xor, W1, b1, W2, b2)
assert a1_b.shape == (4, n_hidden), f"Batch hidden shape wrong: {a1_b.shape}"
assert a2_b.shape == (4, n_output), f"Batch output shape wrong: {a2_b.shape}"

# --- all outputs are valid probabilities (sigmoid output in (0,1)) ---
assert np.all(a2_b > 0) and np.all(a2_b < 1), "All outputs should be in (0,1)"

# --- total parameter count matches expected ---
expected_params = n_input * n_hidden + n_hidden + n_hidden * n_output + n_output
assert total_params == expected_params, f"Total params should be {expected_params}, got {total_params}"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("=" * 60)
print("EXERCISES — Try these yourself!")
print("=" * 60)
print("""
Exercise 1: Add More Hidden Neurons
  Change n_hidden from 2 to 4.
  Update W1 shape to (2, 4), b1 shape to (4,), W2 shape to (4, 1).
  Does the network have more parameters? Count them.
  Hint: W1 = np.random.randn(n_input, 4) * 0.5

Exercise 2: Explore Sigmoid
  Write a loop that prints sigmoid(x) for x in range(-10, 11).
  Notice how the output barely changes for very large/small x.
  This is called "saturation" — it's a problem we'll discuss later.

Exercise 3: Add a Third Layer
  Try adding a second hidden layer with 3 neurons.
  You'll need W3, b3.
  Forward pass: x -> hidden1 -> hidden2 -> output
  Hint: a2 = sigmoid(a1 @ W2 + b2) then output = sigmoid(a2 @ W3 + b3)

Exercise 4: Count Parameters
  For a network with layers [2, 4, 3, 1]:
    - 2 input neurons
    - 4 hidden neurons (layer 1)
    - 3 hidden neurons (layer 2)
    - 1 output neuron
  How many weight parameters total? How many bias parameters?
  Formula: for each pair of adjacent layers (n_in, n_out): n_in * n_out + n_out

Exercise 5: Why Not Just Stack Linear Layers?
  Without activation functions, a deep network collapses to a single
  linear transformation: W3 @ (W2 @ (W1 @ x)) = (W3@W2@W1) @ x
  This means it's no better than ONE layer!
  The activation function (sigmoid) is what makes depth useful.
  Think: what property of sigmoid "breaks" this collapsing?

Exercise 6 (First Principles): Calculate Total Parameters
  For a network with architecture [784, 128, 64, 10]:
    Layer 1->2: 784 * 128 weights + 128 biases = 100,480
    Layer 2->3: 128 * 64 weights  + 64 biases  = 8,256
    Layer 3->4: 64 * 10 weights   + 10 biases  = 650
    Total: 100,480 + 8,256 + 650 = 109,386 parameters
  Verify this by running the calculation printed above!
""")
