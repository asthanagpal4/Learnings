# HOW TO RUN:
#   uv run python 05_math_for_ml/03_functions_and_derivatives.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE 3: Functions and Derivatives
# =============================================================================
# This file covers the math behind how neural networks LEARN.
#
# Topics:
#   - Mathematical functions in Python (using numpy)
#   - What is a derivative? (measuring how fast things change)
#   - Numerical differentiation (computing derivatives with code)
#   - Activation functions: sigmoid, ReLU, tanh
#   - Why activation functions matter in neural networks
# =============================================================================

import numpy as np

print("=" * 60)
print("PART 1: MATHEMATICAL FUNCTIONS")
print("=" * 60)

# ------------------------------------------------------------------
# A mathematical function takes an input and produces an output.
# f(x) = x^2 means: give me a number x, I give you x squared.
#
# We can apply functions to entire numpy arrays at once.
# ------------------------------------------------------------------

x = np.array([-3, -2, -1, 0, 1, 2, 3])

print("\nx values:", x)
print("x^2     :", x**2)
print("x^3     :", x**3)
print("2*x + 1 :", 2*x + 1)
print("sqrt(|x|):", np.round(np.sqrt(np.abs(x)), 3))

# We can define our own functions
def square(x):
    return x ** 2

def linear(x):
    return 2 * x + 1

def cubic(x):
    return x ** 3 - 2 * x

print("\n--- Custom Function: f(x) = x^3 - 2x ---")
print("x     :", x)
print("f(x)  :", cubic(x))

# ------------------------------------------------------------------
# Printing a "function table" instead of a plot
# ------------------------------------------------------------------
def print_function_table(func, x_start, x_end, steps, name="f"):
    """Print a table of x -> f(x) values."""
    x_values = np.linspace(x_start, x_end, steps)
    print(f"\n{'x':>8} | {name}(x)")
    print("-" * 20)
    for x_val in x_values:
        y_val = func(x_val)
        # Visual bar to show magnitude
        bar_len = int(abs(y_val) * 3) if abs(y_val) < 10 else 30
        bar = "#" * bar_len if y_val >= 0 else "-" * bar_len
        print(f"{x_val:>8.2f} | {y_val:>7.3f}  {bar}")

print("\n--- Function Table: f(x) = x^2 ---")
print_function_table(square, -3, 3, 13, "x^2")

print("\n" + "=" * 60)
print("PART 2: WHAT IS A DERIVATIVE?")
print("=" * 60)

# ------------------------------------------------------------------
# The derivative of a function tells us: how fast is the output
# changing as we slightly increase the input?
#
# Imagine walking on a hilly road:
#   - Derivative = 0  means you're on flat ground
#   - Derivative > 0  means you're going UP (output increases as x increases)
#   - Derivative < 0  means you're going DOWN (output decreases as x increases)
#   - Large derivative means steep hill
#   - Small derivative means gentle slope
#
# In neural network training, the derivative tells us:
#   "If we increase this weight by a tiny bit, does the error go up or down?"
# That tells us which direction to move the weight.
#
# FORMAL DEFINITION:
#   f'(x) = lim(h->0) [ f(x+h) - f(x) ] / h
#
# In English: nudge x by a tiny amount h, see how much f(x) changes,
# divide by h to get the "rate of change per unit".
# ------------------------------------------------------------------

print("""
Analogy: Speed as a derivative
  If you drive 100 km in 2 hours, your average speed = 50 km/h.
  Speed = (change in distance) / (change in time)
  This is exactly the derivative idea!

  Derivative = (change in output) / (change in input)
             = (f(x+h) - f(x)) / h    for very small h
""")

# ===========================================================================
# FIRST PRINCIPLES: The limit definition of the derivative
# ===========================================================================
# The derivative is defined as a LIMIT:
#
#   f'(x) = lim     f(x + h) - f(x)
#           h -> 0  -------------------
#                          h
#
# This says: take two points on the curve, x and x+h, compute the
# slope of the line between them (rise over run), and let h shrink
# to zero. The slope of that "secant line" becomes the slope of the
# "tangent line" -- the instantaneous rate of change.
#
# Example with f(x) = x^2:
#   f(x+h) = (x+h)^2 = x^2 + 2xh + h^2
#   f(x+h) - f(x) = 2xh + h^2
#   [f(x+h) - f(x)] / h = 2x + h
#   lim(h->0) [2x + h] = 2x
#
# So f'(x) = 2x. The derivative of x^2 is 2x.
# This is the EXACT method. Below we use a NUMERICAL approximation.
# ===========================================================================

print("\n--- FIRST PRINCIPLES: Limit definition of derivative ---")
print("  f'(x) = lim(h->0) [f(x+h) - f(x)] / h")
print("  Example: f(x) = x^2")
print("  f(x+h) = x^2 + 2xh + h^2")
print("  [f(x+h) - f(x)] / h = 2x + h")
print("  As h -> 0: f'(x) = 2x")

# Show the limit converging
x_demo = 3.0
print(f"\n  Watching the limit converge at x = {x_demo}:")
print(f"  {'h':>12} | {'[f(x+h)-f(x)]/h':>18} | {'Exact (2x=6)':>14}")
print(f"  {'-'*50}")
for power in range(1, 11):
    h_val = 10 ** (-power)
    approx = (square(x_demo + h_val) - square(x_demo)) / h_val
    print(f"  {h_val:>12.1e} | {approx:>18.10f} | {2*x_demo:>14.10f}")
print(f"  As h gets smaller, the approximation converges to exactly 6.0!")

# ------------------------------------------------------------------
# Numerical derivative
# ------------------------------------------------------------------
# Instead of doing calculus algebra, we approximate the derivative
# by using a very small h (like h = 0.0001).

def numerical_derivative(func, x, h=1e-5):
    """Approximate the derivative of func at point x."""
    return (func(x + h) - func(x)) / h

# Test on f(x) = x^2
# Calculus tells us: derivative of x^2 is 2x
# So at x=3, derivative should be 2*3 = 6

f = square  # f(x) = x^2

test_points = [-3, -2, -1, 0, 1, 2, 3]
print("--- Derivative of f(x) = x^2 ---")
print(f"{'x':>5} | {'Numerical f\'(x)':>16} | {'Exact (2x)':>10} | {'Match?':>8}")
print("-" * 50)
for xp in test_points:
    numerical = numerical_derivative(f, xp)
    exact     = 2 * xp   # d/dx (x^2) = 2x
    match     = "YES" if abs(numerical - exact) < 0.001 else "NO"
    print(f"{xp:>5} | {numerical:>16.6f} | {exact:>10.6f} | {match:>8}")

# Test on f(x) = x^3 - 2x
# Derivative: 3x^2 - 2
print("\n--- Derivative of f(x) = x^3 - 2x ---")
print(f"{'x':>5} | {'Numerical f\'(x)':>16} | {'Exact (3x^2-2)':>14}")
print("-" * 44)
for xp in test_points:
    numerical = numerical_derivative(cubic, xp)
    exact     = 3 * xp**2 - 2
    print(f"{xp:>5} | {numerical:>16.6f} | {exact:>14.6f}")

print("\n" + "=" * 60)
print("PART 3: ACTIVATION FUNCTIONS")
print("=" * 60)

# ------------------------------------------------------------------
# Activation functions are used inside neural networks to introduce
# non-linearity. Without them, a neural network (no matter how deep)
# would just be one big linear function — very limited in what it
# could learn.
#
# Think of it this way:
#   WITHOUT activation functions: net can only draw straight lines
#   WITH activation functions: net can draw any shape
#
# The three most important activation functions are:
#   1. Sigmoid : squishes any number to (0, 1) — like a probability
#   2. ReLU    : keeps positive numbers, sets negatives to 0
#   3. Tanh    : squishes any number to (-1, 1)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 1. SIGMOID
# ------------------------------------------------------------------
# Formula: sigmoid(x) = 1 / (1 + e^(-x))
# Output range: (0, 1)
# At x=0: sigmoid(0) = 0.5
# Large positive x: sigmoid(x) -> 1
# Large negative x: sigmoid(x) -> 0
#
# Used in: output layer of binary classifiers (is this a cat? yes/no)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ------------------------------------------------------------------
# 2. ReLU (Rectified Linear Unit)
# ------------------------------------------------------------------
# Formula: relu(x) = max(0, x)
# Output range: [0, infinity)
# If x is positive, pass it through unchanged.
# If x is negative, output 0.
#
# Used in: hidden layers of most modern neural networks
# WHY? Simple, fast, works very well in practice.

def relu(x):
    return np.maximum(0, x)

# ------------------------------------------------------------------
# 3. Tanh (Hyperbolic Tangent)
# ------------------------------------------------------------------
# Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# Output range: (-1, 1)
# At x=0: tanh(0) = 0
# Like sigmoid, but centered at 0 (which often trains better)
#
# numpy already has np.tanh() built in

def tanh(x):
    return np.tanh(x)   # using numpy's built-in

x_vals = np.linspace(-5, 5, 11)

print("\n--- Activation Function Values ---")
print(f"\n{'x':>6} | {'sigmoid':>9} | {'relu':>9} | {'tanh':>9}")
print("-" * 45)
for x_val in x_vals:
    sig  = sigmoid(x_val)
    rel  = relu(x_val)
    tan  = tanh(x_val)
    print(f"{x_val:>6.1f} | {sig:>9.4f} | {rel:>9.4f} | {tan:>9.4f}")

print("\n--- Key Properties ---")
print("sigmoid(-5):", round(sigmoid(-5), 5), "  (very close to 0)")
print("sigmoid( 0):", round(sigmoid(0), 5),  "  (exactly 0.5)")
print("sigmoid( 5):", round(sigmoid(5), 5),  "  (very close to 1)")
print()
print("relu(-3):", relu(-3), "  (negative -> 0)")
print("relu( 0):", relu(0),  "  (zero -> 0)")
print("relu( 3):", relu(3),  "  (positive -> unchanged)")
print()
print("tanh(-5):", round(tanh(-5), 5), "  (close to -1)")
print("tanh( 0):", round(tanh(0), 5),  "  (exactly 0)")
print("tanh( 5):", round(tanh(5), 5),  "  (close to +1)")

print("\n" + "=" * 60)
print("PART 4: DERIVATIVES OF ACTIVATION FUNCTIONS")
print("=" * 60)

# ------------------------------------------------------------------
# Why do we need derivatives of activation functions?
#
# During training, neural networks use BACKPROPAGATION to learn.
# Backpropagation computes the derivative of the loss with respect to
# each weight — and the chain rule of calculus requires us to multiply
# by the derivative of the activation function at each layer.
#
# So knowing these derivatives is essential!
# ------------------------------------------------------------------

# ===========================================================================
# FIRST PRINCIPLES: Deriving sigmoid derivative step by step
# ===========================================================================
# sigma(x) = 1 / (1 + e^(-x))
#
# Step 1: Rewrite as sigma(x) = (1 + e^(-x))^(-1)
#
# Step 2: Apply chain rule: d/dx [u^(-1)] = -u^(-2) * du/dx
#   u = 1 + e^(-x)
#   du/dx = -e^(-x)       (derivative of e^(-x) is -e^(-x))
#
# Step 3: sigma'(x) = -(1 + e^(-x))^(-2) * (-e^(-x))
#                    = e^(-x) / (1 + e^(-x))^2
#
# Step 4: Factor cleverly:
#   = [1 / (1 + e^(-x))] * [e^(-x) / (1 + e^(-x))]
#   = sigma(x) * [e^(-x) / (1 + e^(-x))]
#
# Step 5: Notice that e^(-x) / (1 + e^(-x)) = 1 - 1/(1 + e^(-x)) = 1 - sigma(x)
#   (Because: 1 - sigma = 1 - 1/(1+e^-x) = (1+e^-x - 1)/(1+e^-x) = e^-x/(1+e^-x))
#
# RESULT: sigma'(x) = sigma(x) * (1 - sigma(x))
#
# This beautiful form means the derivative can be computed from the output
# alone -- no need to recompute e^(-x). Very efficient for backpropagation!
# ===========================================================================

print("\n--- FIRST PRINCIPLES: Sigmoid derivative derivation ---")
print("  sigma(x) = 1 / (1 + e^(-x))")
print("  sigma'(x) = e^(-x) / (1 + e^(-x))^2")
print("            = sigma(x) * (1 - sigma(x))")
x_check = 2.0
s = sigmoid(x_check)
print(f"  Verify at x={x_check}: sigma = {s:.6f}")
print(f"  sigma*(1-sigma) = {s*(1-s):.6f}")
print(f"  Numerical deriv = {(sigmoid(x_check+1e-5) - sigmoid(x_check))/1e-5:.6f}")

# ===========================================================================
# FIRST PRINCIPLES: Deriving ReLU derivative
# ===========================================================================
# ReLU(x) = max(0, x)
#
# This is a piecewise function:
#   ReLU(x) = x   if x > 0
#   ReLU(x) = 0   if x < 0
#
# Derivative of each piece:
#   ReLU'(x) = 1   if x > 0    (derivative of x is 1)
#   ReLU'(x) = 0   if x < 0    (derivative of 0 is 0)
#   ReLU'(0) = undefined        (there's a "kink" at x=0)
#
# By convention, we set ReLU'(0) = 0 (or sometimes 0.5).
# This works fine in practice because hitting exactly x=0 is rare.
#
# WHY ReLU is popular: its derivative is either 0 or 1 (never shrinks).
# Compare with sigmoid: its derivative is at most 0.25, so gradients
# get smaller as they flow backward through many layers (vanishing gradient).
# ===========================================================================

print("\n--- FIRST PRINCIPLES: ReLU derivative derivation ---")
print("  ReLU(x) = max(0, x)")
print("  ReLU'(x) = 1 if x > 0")
print("  ReLU'(x) = 0 if x < 0")
print("  ReLU'(0) = undefined (convention: 0)")

# Analytic (exact) derivatives:
# sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
# relu'(x)    = 1 if x > 0, else 0
# tanh'(x)    = 1 - tanh(x)^2

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)  # 1 where positive, 0 elsewhere

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

# Compare analytic vs numerical derivatives
x_test = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

print("\n--- Sigmoid Derivative ---")
print(f"{'x':>6} | {'Analytic':>10} | {'Numerical':>10} | {'Close?':>8}")
print("-" * 44)
for xp in x_test:
    analytic  = sigmoid_derivative(xp)
    numerical = numerical_derivative(sigmoid, xp)
    close     = "YES" if abs(analytic - numerical) < 1e-4 else "NO"
    print(f"{xp:>6.1f} | {analytic:>10.6f} | {numerical:>10.6f} | {close:>8}")

print("\n--- ReLU Derivative ---")
print(f"{'x':>6} | {'Analytic':>10} | {'Numerical':>10}")
print("-" * 35)
for xp in x_test:
    analytic  = relu_derivative(xp)
    numerical = numerical_derivative(relu, xp)
    print(f"{xp:>6.1f} | {analytic:>10.6f} | {numerical:>10.6f}")

print("\n--- Tanh Derivative ---")
print(f"{'x':>6} | {'Analytic':>10} | {'Numerical':>10} | {'Close?':>8}")
print("-" * 44)
for xp in x_test:
    analytic  = tanh_derivative(xp)
    numerical = numerical_derivative(tanh, xp)
    close     = "YES" if abs(analytic - numerical) < 1e-4 else "NO"
    print(f"{xp:>6.1f} | {analytic:>10.6f} | {numerical:>10.6f} | {close:>8}")

# ------------------------------------------------------------------
# Visualizing derivatives as "slope tables"
# ------------------------------------------------------------------
print("\n--- Derivative of Sigmoid (slope table) ---")
x_range = np.linspace(-6, 6, 25)
print(f"{'x':>7} | {'sigmoid(x)':>11} | {'slope':>7} | shape")
print("-" * 55)
for xp in x_range:
    sig  = sigmoid(xp)
    deriv = sigmoid_derivative(xp)
    bar_len = int(deriv * 40)
    bar = "#" * bar_len
    print(f"{xp:>7.2f} | {sig:>11.4f} | {deriv:>7.4f} | {bar}")

print("""
Notice:
  - Slope is highest at x=0 (sigmoid is steepest there, = 0.25)
  - Slope gets very small for large |x| (the "vanishing gradient" problem!)
  - ReLU avoids this: its slope is always 1 for positive x, never shrinks
""")

print("\n" + "=" * 60)
print("PART 5: WHY THIS MATTERS FOR NEURAL NETWORKS")
print("=" * 60)

print("""
In a neural network:

  1. FORWARD PASS:
     - Data goes in: input -> layer1 -> activation -> layer2 -> activation -> output
     - Each layer computes: z = W @ x + b   (matrix multiply + bias)
     - Then applies activation: a = activation(z)
     - Activation adds non-linearity so the network can learn complex patterns

  2. BACKWARD PASS (learning):
     - We compute how much each weight contributed to the error
     - This requires the DERIVATIVE of the activation function
     - Chain rule: multiply derivatives backwards through the network
     - This process is called BACKPROPAGATION

  3. WHICH ACTIVATION TO USE?
     - Hidden layers: usually ReLU (fast, avoids vanishing gradient)
     - Binary output: sigmoid (gives probability 0 to 1)
     - Multiclass output: softmax (see file 05_probability_basics.py)
     - Sometimes tanh: works well in RNNs (recurrent neural networks)
""")

# Quick demo: sigmoid saturates (derivative goes to 0) for large inputs
# This is the "vanishing gradient" problem
print("--- Vanishing Gradient Demo ---")
large_inputs = np.array([1, 5, 10, 20, 50])
print(f"\n{'Input x':>10} | {'sigmoid(x)':>11} | {'slope (d_sigmoid)':>18}")
print("-" * 45)
for xp in large_inputs:
    sig   = sigmoid(xp)
    slope = sigmoid_derivative(xp)
    print(f"{xp:>10} | {sig:>11.8f} | {slope:>18.10f}")

print("""
For x=50, the slope is essentially 0!
If your network's inputs are large, the gradient "vanishes" and the
network stops learning. This is why data normalization (File 02) matters!
""")

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key ideas from this file:
  - Functions map inputs to outputs: f(x) -> y
  - Derivative measures the SLOPE (rate of change) at each point
  - Numerical derivative: (f(x+h) - f(x)) / h  with small h
  - Sigmoid : output in (0,1)  -- used in binary classifiers
  - ReLU    : output in [0,inf) -- most common in hidden layers
  - Tanh    : output in (-1,1) -- zero-centered, used in RNNs
  - Derivatives of activations are needed for backpropagation
  - Sigmoid's slope goes to 0 for large inputs (vanishing gradient)
""")

# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test custom functions
assert square(3) == 9, "square(3) should be 9"
assert square(-4) == 16, "square(-4) should be 16"
assert linear(0) == 1, "linear(0) = 2*0+1 should be 1"
assert linear(2) == 5, "linear(2) = 2*2+1 should be 5"
assert cubic(0) == 0, "cubic(0) = 0^3 - 2*0 should be 0"
assert cubic(2) == 4, "cubic(2) = 8 - 4 should be 4"

# Test numerical derivative of x^2: at x=3, should be close to 2*3=6
deriv_sq_3 = numerical_derivative(square, 3.0)
assert abs(deriv_sq_3 - 6.0) < 0.01, f"Derivative of x^2 at x=3 should be ~6, got {deriv_sq_3}"

# Test numerical derivative of x^2 at x=-2: should be close to 2*(-2)=-4
deriv_sq_neg2 = numerical_derivative(square, -2.0)
assert abs(deriv_sq_neg2 - (-4.0)) < 0.01, "Derivative of x^2 at x=-2 should be ~-4"

# Test numerical derivative of cubic at x=1: 3*(1)^2 - 2 = 1
deriv_cubic_1 = numerical_derivative(cubic, 1.0)
assert abs(deriv_cubic_1 - 1.0) < 0.01, "Derivative of x^3-2x at x=1 should be ~1"

# Test sigmoid values
assert abs(sigmoid(0) - 0.5) < 1e-9, "sigmoid(0) should be exactly 0.5"
assert sigmoid(100) > 0.99, "sigmoid(100) should be very close to 1"
assert sigmoid(-100) < 0.01, "sigmoid(-100) should be very close to 0"

# Test ReLU
assert relu(-5) == 0, "relu(-5) should be 0"
assert relu(0) == 0, "relu(0) should be 0"
assert relu(3) == 3, "relu(3) should be 3"
assert np.array_equal(relu(np.array([-2, -1, 0, 1, 2])), np.array([0, 0, 0, 1, 2])), "relu array failed"

# Test tanh
assert abs(tanh(0) - 0.0) < 1e-9, "tanh(0) should be 0"
assert tanh(100) > 0.99, "tanh(100) should be close to 1"
assert tanh(-100) < -0.99, "tanh(-100) should be close to -1"

# Test sigmoid derivative: sigma(x) * (1 - sigma(x))
# At x=0: sigmoid(0)=0.5, so derivative = 0.5*0.5 = 0.25
assert abs(sigmoid_derivative(0) - 0.25) < 1e-9, "sigmoid_derivative(0) should be 0.25"
# Verify against numerical derivative
for xp in [-2.0, 0.0, 2.0]:
    analytic = sigmoid_derivative(xp)
    numerical = numerical_derivative(sigmoid, xp)
    assert abs(analytic - numerical) < 1e-4, f"Sigmoid derivative mismatch at x={xp}"

# Test ReLU derivative: 1 for x>0, 0 for x<0
assert relu_derivative(3.0) == 1.0, "relu_derivative(3) should be 1"
assert relu_derivative(-3.0) == 0.0, "relu_derivative(-3) should be 0"

# Test tanh derivative: 1 - tanh(x)^2
# At x=0: 1 - 0^2 = 1
assert abs(tanh_derivative(0) - 1.0) < 1e-9, "tanh_derivative(0) should be 1"
for xp in [-1.0, 1.0, 2.0]:
    analytic = tanh_derivative(xp)
    numerical = numerical_derivative(tanh, xp)
    assert abs(analytic - numerical) < 1e-4, f"Tanh derivative mismatch at x={xp}"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
# 1. Implement the "leaky ReLU" function: if x > 0, return x; else return 0.01*x
#    Compute its derivative numerically for x in [-3, -1, 0, 1, 3].
#    How is it different from regular ReLU?
#
# 2. Implement the function f(x) = x^4 - 3x^2 + x.
#    Compute its numerical derivative at x = -2, -1, 0, 1, 2.
#    The exact derivative is 4x^3 - 6x + 1. Verify your numerical answer.
#
# 3. What is sigmoid(0)? Compute it and explain why it equals 0.5
#    by looking at the formula: 1 / (1 + e^(-0))
#
# 4. Apply ReLU to this array: [-5, -3, -1, 0, 1, 3, 5].
#    Then apply the sigmoid to the same array.
#    Compare the outputs — when are they similar? When do they differ?
#
# 5. The derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x)).
#    At what value of x is this derivative LARGEST?
#    (Hint: try computing sigmoid_derivative for many values and find the max.)
#
# 6. FIRST PRINCIPLES EXERCISE: Derive the derivative of tanh(x).
#    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
#    Use the quotient rule: (u/v)' = (u'v - uv') / v^2
#    Let u = e^x - e^(-x),  v = e^x + e^(-x)
#    Then u' = e^x + e^(-x) = v,  v' = e^x - e^(-x) = u
#    So tanh'(x) = (v*v - u*u) / v^2 = 1 - (u/v)^2 = 1 - tanh(x)^2
#    Verify numerically for x = -2, -1, 0, 1, 2.
# =============================================================================
