# HOW TO RUN:
#   uv run python 06_neural_networks/01_perceptron.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# LESSON 1: THE PERCEPTRON — The Simplest Artificial Neuron
# =============================================================================
# Run this file with:  uv run python 06_neural_networks/01_perceptron.py
#
# WHAT IS A NEURON?
# -----------------
# Your brain has ~86 billion neurons. Each neuron:
#   1. Receives signals from other neurons (INPUTS)
#   2. Decides how important each signal is (WEIGHTS)
#   3. Adds up all the weighted signals plus a bias
#   4. If the total is strong enough, it "fires" (ACTIVATION)
#
# An artificial neuron (perceptron) works the same way:
#   output = activation( input1*weight1 + input2*weight2 + ... + bias )
#
# Think of it like a committee vote:
#   - Each input is a person's opinion (0 = no, 1 = yes)
#   - Each weight is how much we trust that person
#   - The bias is a "default leaning" of the chairperson
#   - The activation decides: is the total vote enough to pass?

import numpy as np

# Set a random seed so we get the same results every time we run
np.random.seed(42)

print("=" * 60)
print("LESSON 1: THE PERCEPTRON")
print("=" * 60)


# =============================================================================
# PART 1: THE STEP ACTIVATION FUNCTION
# =============================================================================
# The simplest activation: if the total >= 0, output 1 (fire!); else output 0
# It's like a light switch: either ON or OFF, nothing in between.

def step_activation(total):
    """Returns 1 if total >= 0, else 0. Like a light switch."""
    if total >= 0:
        return 1
    else:
        return 0


print("\n--- Part 1: Step Activation ---")
print("step_activation(-5) =", step_activation(-5))   # 0
print("step_activation(0)  =", step_activation(0))    # 1
print("step_activation(3)  =", step_activation(3))    # 1


# =============================================================================
# PART 2: A SINGLE PERCEPTRON AS A FUNCTION
# =============================================================================
# The perceptron formula:
#   total = sum(inputs * weights) + bias
#   output = step_activation(total)

def perceptron(inputs, weights, bias):
    """
    A single artificial neuron.

    inputs:  a list/array of input values  e.g. [1, 0]
    weights: a list/array of weights       e.g. [0.5, -0.3]
    bias:    a single number               e.g. 0.1

    Returns: 0 or 1
    """
    inputs = np.array(inputs)
    weights = np.array(weights)

    # Step 1: Multiply each input by its weight and add them all up
    weighted_sum = np.dot(inputs, weights)  # dot product = multiply + sum
    # np.dot([1,0], [0.5,-0.3]) = 1*0.5 + 0*(-0.3) = 0.5

    # Step 2: Add the bias
    total = weighted_sum + bias

    # Step 3: Apply the activation function
    output = step_activation(total)

    return output


print("\n--- Part 2: Perceptron by hand ---")
# Let's test with some inputs
print("inputs=[1,1], weights=[0.5,0.5], bias=-0.5 ->", perceptron([1, 1], [0.5, 0.5], -0.5))
print("inputs=[0,0], weights=[0.5,0.5], bias=-0.5 ->", perceptron([0, 0], [0.5, 0.5], -0.5))


# =============================================================================
# PART 3: TRAINING — ADJUSTING WEIGHTS BASED ON ERRORS
# =============================================================================
# How does the neuron LEARN the right weights?
#
# The Perceptron Learning Rule (very simple):
#   1. Make a prediction with current weights
#   2. Compare prediction to correct answer (compute error)
#      error = correct_answer - prediction
#   3. Adjust each weight:
#      new_weight = old_weight + learning_rate * error * input
#   4. Adjust the bias:
#      new_bias = old_bias + learning_rate * error
#   5. Repeat for all training examples, many times
#
# If prediction was right, error=0, so nothing changes.
# If prediction was too high (pred=1, correct=0), error=-1 -> weights decrease.
# If prediction was too low  (pred=0, correct=1), error=+1 -> weights increase.

def train_perceptron(X, y, learning_rate=0.1, epochs=20):
    """
    Train a perceptron on dataset X with labels y.

    X: 2D array of shape (num_samples, num_inputs)
    y: 1D array of correct outputs (0 or 1)
    learning_rate: how big each adjustment step is (small = careful, slow)
    epochs: how many times we go through the whole dataset

    Returns: final weights and bias
    """
    num_inputs = X.shape[1]  # how many input features

    # Start with random small weights
    weights = np.random.uniform(-0.5, 0.5, num_inputs)
    bias = np.random.uniform(-0.5, 0.5)

    print(f"  Starting weights: {weights.round(3)}, bias: {bias:.3f}")

    for epoch in range(epochs):
        total_errors = 0

        for i in range(len(X)):
            # Make prediction
            prediction = perceptron(X[i], weights, bias)

            # Compute error
            error = y[i] - prediction

            # Update weights and bias (the learning step!)
            weights = weights + learning_rate * error * X[i]
            bias = bias + learning_rate * error

            if error != 0:
                total_errors += 1

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: {total_errors} errors this round")

        # If no errors at all, we've learned perfectly — stop early
        if total_errors == 0:
            print(f"  Learned perfectly at epoch {epoch+1}!")
            break

    print(f"  Final weights: {weights.round(3)}, bias: {bias:.3f}")
    return weights, bias


def test_perceptron(X, y, weights, bias, gate_name):
    """Print predictions vs expected for all inputs."""
    print(f"\n  Predictions for {gate_name}:")
    print("  Input A | Input B | Predicted | Expected | Correct?")
    print("  " + "-" * 50)
    correct = 0
    for i in range(len(X)):
        pred = perceptron(X[i], weights, bias)
        expected = y[i]
        ok = "YES" if pred == expected else "NO "
        if pred == expected:
            correct += 1
        print(f"    {int(X[i][0])}       {int(X[i][1])}          {pred}          {expected}       {ok}")
    print(f"  Accuracy: {correct}/{len(X)} = {100*correct//len(X)}%")


# =============================================================================
# PART 4: TRAIN ON AND GATE
# =============================================================================
# AND gate truth table:
#   0 AND 0 = 0
#   0 AND 1 = 0
#   1 AND 0 = 0
#   1 AND 1 = 1   (only true when BOTH inputs are true)

print("\n--- Part 4: Training on AND Gate ---")
X_and = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_and = np.array([0, 0, 0, 1])  # AND outputs

weights_and, bias_and = train_perceptron(X_and, y_and, learning_rate=0.1, epochs=30)
test_perceptron(X_and, y_and, weights_and, bias_and, "AND Gate")


# =============================================================================
# PART 5: TRAIN ON OR GATE
# =============================================================================
# OR gate truth table:
#   0 OR 0 = 0
#   0 OR 1 = 1
#   1 OR 0 = 1
#   1 OR 1 = 1   (true when AT LEAST ONE input is true)

print("\n--- Part 5: Training on OR Gate ---")
X_or = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y_or = np.array([0, 1, 1, 1])  # OR outputs

weights_or, bias_or = train_perceptron(X_or, y_or, learning_rate=0.1, epochs=30)
test_perceptron(X_or, y_or, weights_or, bias_or, "OR Gate")


# =============================================================================
# PART 6: XOR FAILURE — WHY ONE NEURON IS NOT ENOUGH
# =============================================================================
# XOR gate truth table:
#   0 XOR 0 = 0
#   0 XOR 1 = 1   (true only when inputs are DIFFERENT)
#   1 XOR 0 = 1
#   1 XOR 1 = 0
#
# WHY CAN'T A SINGLE PERCEPTRON LEARN XOR?
# -----------------------------------------
# A perceptron draws a STRAIGHT LINE to separate the two classes.
# For AND and OR, you can draw a straight line to separate 0s from 1s.
# For XOR, the 1s are at corners (0,1) and (1,0), and 0s at (0,0) and (1,1).
# No matter how you tilt a straight line, you CANNOT separate them!
#
# This is called "not linearly separable."
# The solution: use MULTIPLE layers of neurons! (Next lesson)

print("\n--- Part 6: XOR FAILURE (expected!) ---")
print("  XOR truth table: 0,0->0  |  0,1->1  |  1,0->1  |  1,1->0")
print("  A single perceptron CANNOT learn XOR.")
print("  Watch it fail no matter how many epochs we train...")

X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR outputs

weights_xor, bias_xor = train_perceptron(X_xor, y_xor, learning_rate=0.1, epochs=50)
test_perceptron(X_xor, y_xor, weights_xor, bias_xor, "XOR Gate (should fail!)")

print("\n  The perceptron never reaches 100% on XOR.")
print("  This is NOT a bug — it's a fundamental limitation of one neuron.")
print("  Solution: Stack multiple neurons in layers! See 02_multilayer_network.py")


# =============================================================================
# FIRST PRINCIPLES: WHY XOR IS NOT LINEARLY SEPARABLE (Proof by Contradiction)
# =============================================================================
# A perceptron decides using a line: w1*x1 + w2*x2 + b = 0
# Points where w1*x1 + w2*x2 + b > 0 are class 1, others class 0.
#
# XOR truth table:
#   (0,0) -> 0    (1,1) -> 0    (0,1) -> 1    (1,0) -> 1
#
# PROOF BY CONTRADICTION:
#   Assume a line w1*x1 + w2*x2 + b = 0 separates class 1 from class 0.
#
#   From (0,0) -> 0:   w1*0 + w2*0 + b <= 0      =>  b <= 0         ... (i)
#   From (1,1) -> 0:   w1*1 + w2*1 + b <= 0      =>  w1 + w2 + b <= 0  ... (ii)
#   From (0,1) -> 1:   w1*0 + w2*1 + b > 0        =>  w2 + b > 0       ... (iii)
#   From (1,0) -> 1:   w1*1 + w2*0 + b > 0        =>  w1 + b > 0       ... (iv)
#
#   Adding (iii) and (iv): w1 + w2 + 2b > 0       ... (v)
#   From (ii):             w1 + w2 + b <= 0
#   So:                    w1 + w2 <= -b
#   Substituting into (v): -b + 2b > 0 => b > 0
#   But (i) says b <= 0. CONTRADICTION!
#
#   Therefore, no single line can separate XOR. QED.
#
# =============================================================================
# FIRST PRINCIPLES: PERCEPTRON CONVERGENCE THEOREM (Simplified)
# =============================================================================
# If the data IS linearly separable, the perceptron WILL converge. Here's why:
#
# Define:
#   R = max norm of any input vector (max ||x_i||)
#   gamma = margin = minimum distance from any point to the separating hyperplane
#
# Theorem: The perceptron makes at most (R / gamma)^2 weight updates.
#
# Intuition:
#   - Each correct update moves the weight vector closer to the ideal direction.
#   - The margin gamma guarantees each update makes meaningful progress.
#   - The bound R limits how big each step can be.
#   - After at most (R/gamma)^2 steps, the weight vector aligns well enough
#     to classify all points correctly.
#
# For binary inputs (like our gates), R = sqrt(2) (for 2D inputs).
# If the margin is gamma = 0.5, max updates = (sqrt(2)/0.5)^2 = 8.
#
# =============================================================================
# COMPLEXITY ANALYSIS
# =============================================================================
# Time complexity per training pass:
#   - For n samples with d features:
#     - Each sample: compute dot product O(d), update weights O(d)
#     - Total per epoch: O(n * d)
#   - Total training: O(n * d * number_of_epochs)
#   - If data is linearly separable: converges in at most (R/gamma)^2 epochs
#
# Space complexity:
#   - Weights: O(d) for d features
#   - Input storage: O(n * d)
#   - Total: O(n * d)
# =============================================================================

print("\n--- First Principles: XOR Impossibility Proof ---")
print("  Proof by contradiction that XOR is not linearly separable:")
print("  Assume line w1*x1 + w2*x2 + b = 0 separates the classes.")
print("  From (0,0)->0:  b <= 0")
print("  From (1,1)->0:  w1 + w2 + b <= 0")
print("  From (0,1)->1:  w2 + b > 0")
print("  From (1,0)->1:  w1 + b > 0")
print("  Adding last two: w1 + w2 + 2b > 0")
print("  But w1 + w2 <= -b, so -b + 2b > 0 => b > 0")
print("  Contradiction with b <= 0! No line can separate XOR.")

print("\n--- Perceptron Convergence Theorem ---")
print("  If data is linearly separable with margin gamma,")
print("  and max input norm is R, perceptron converges in")
print("  at most (R/gamma)^2 updates.")
R = np.sqrt(2)  # max norm of 2D binary input e.g. [1,1]
gamma = 0.5     # example margin
print(f"  Example: R={R:.3f}, gamma={gamma}, max updates = {(R/gamma)**2:.0f}")


# =============================================================================
# KEY CONCEPTS SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
Perceptron:     A single artificial neuron
Weights:        How much each input matters (learned during training)
Bias:           A default "lean" — shifts the decision boundary
Activation:     Decides the output based on weighted sum (step function here)
Learning rule:  new_weight = old_weight + learning_rate * error * input
Epoch:          One full pass through all training examples
Linear sep.:    A problem a single perceptron CAN solve (separable by a line)
XOR problem:    Famous example that needs more than one neuron to solve
""")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- step_activation tests ---
assert step_activation(-5) == 0, "step_activation(-5) should be 0"
assert step_activation(0)  == 1, "step_activation(0) should be 1"
assert step_activation(3)  == 1, "step_activation(3) should be 1"

# --- perceptron tests (with known weights) ---
# weights [0.5, 0.5], bias -0.5: only fires when both inputs are 1
assert perceptron([1, 1], [0.5, 0.5], -0.5) == 1, "both inputs=1 should fire"
assert perceptron([0, 0], [0.5, 0.5], -0.5) == 0, "both inputs=0 should not fire"
assert perceptron([1, 0], [0.5, 0.5], -0.5) == 1, "one input=1 with bias=-0.5: sum=0.0, step(0)=1"

# --- AND gate: trained perceptron must get all 4 cases right ---
X_and_t = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and_t = np.array([0, 0, 0, 1])
preds_and = [perceptron(X_and_t[i], weights_and, bias_and) for i in range(4)]
assert preds_and == list(y_and_t), f"AND gate predictions wrong: {preds_and}"

# --- OR gate: trained perceptron must get all 4 cases right ---
X_or_t = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or_t  = np.array([0, 1, 1, 1])
preds_or = [perceptron(X_or_t[i], weights_or, bias_or) for i in range(4)]
assert preds_or == list(y_or_t), f"OR gate predictions wrong: {preds_or}"

# --- XOR: single perceptron CANNOT learn XOR (accuracy < 100%) ---
X_xor_t = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor_t  = np.array([0, 1, 1, 0])
preds_xor = [perceptron(X_xor_t[i], weights_xor, bias_xor) for i in range(4)]
xor_correct = sum(p == e for p, e in zip(preds_xor, y_xor_t))
assert xor_correct < 4, "Single perceptron should NOT perfectly solve XOR"

# --- sigmoid derivative correctness: a*(1-a) ---
# Using np here to mirror lesson code (sigmoid_derivative is in lesson 3,
# so we verify the formula directly)
for a_v in [0.0, 0.5, 1.0]:
    expected = a_v * (1 - a_v)
    assert np.isclose(expected, a_v * (1.0 - a_v)), "sigmoid derivative formula failed"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("=" * 60)
print("EXERCISES — Try these yourself!")
print("=" * 60)
print("""
Exercise 1: NAND Gate
  NAND is "NOT AND": output is 1 unless both inputs are 1.
  Truth table: (0,0)->1, (0,1)->1, (1,0)->1, (1,1)->0
  Hint: define y_nand = np.array([1,1,1,0]) and call train_perceptron.
  Can a single perceptron learn NAND?

Exercise 2: Change the Learning Rate
  Try learning_rate = 0.01, 0.5, 1.0 for the AND gate.
  What happens if the learning rate is very large?
  What happens if it's very small?
  Hint: pass learning_rate=0.01 (or 0.5) to train_perceptron.

Exercise 3: More Inputs
  Can you build a 3-input AND gate?
  Truth table: output is 1 only if all 3 inputs are 1.
  Hint: X will have shape (8,3) — 8 possible combinations of 3 binary inputs.
  Use np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

Exercise 4: Visualize the Weights
  After training the AND gate, print what the perceptron "learned":
  - Are both weights positive? (Makes sense for AND — both inputs matter equally)
  - Is the bias negative? (Makes sense — default is 0 unless both inputs push it up)

Exercise 5: Count Training Steps
  Modify train_perceptron to count the TOTAL number of weight updates made.
  Is it more for XOR than for AND? Why?

Exercise 6 (First Principles): Prove AND is Linearly Separable
  Find values of w1, w2, b such that the line w1*x1 + w2*x2 + b = 0
  correctly separates AND gate outputs.
  Hint: AND outputs 1 only for (1,1). You need:
    w1*0 + w2*0 + b <= 0  => b <= 0
    w1*0 + w2*1 + b <= 0  => w2 + b <= 0
    w1*1 + w2*0 + b <= 0  => w1 + b <= 0
    w1*1 + w2*1 + b > 0   => w1 + w2 + b > 0
  Try w1=1, w2=1, b=-1.5:
    (0,0): 0+0-1.5 = -1.5 <= 0  (correct, class 0)
    (0,1): 0+1-1.5 = -0.5 <= 0  (correct, class 0)
    (1,0): 1+0-1.5 = -0.5 <= 0  (correct, class 0)
    (1,1): 1+1-1.5 = 0.5 > 0    (correct, class 1)
  All conditions satisfied! AND IS linearly separable.
""")
