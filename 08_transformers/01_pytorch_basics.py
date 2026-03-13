# HOW TO RUN:
#   uv run python 08_transformers/01_pytorch_basics.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# LESSON 1: PyTorch Basics
# =============================================================================
#
# WHAT IS PYTORCH?
# ----------------
# PyTorch is a library for doing math with multi-dimensional arrays (called
# "tensors") on either CPU or GPU. Think of it as NumPy, but with two killer
# features:
#
#   1. GPU acceleration — run the same code on a graphics card, 100x faster.
#   2. Autograd — PyTorch automatically computes gradients (slopes) for you.
#      This is what makes training neural networks possible without doing
#      calculus by hand.
#
# Run this file:
#   uv run python 01_pytorch_basics.py
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=" * 60)
print("PART 1: Tensors — PyTorch's version of arrays")
print("=" * 60)

# --- Creating tensors ---
# A tensor is just a grid of numbers (like a list, a table, or a 3-D cube of
# numbers). The number of dimensions is called the "rank".

# 1-D tensor (a list of numbers)
t1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("1-D tensor:", t1)
print("Shape:", t1.shape)       # torch.Size([4])  — 4 elements

# 2-D tensor (a table / matrix)
t2 = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
print("\n2-D tensor:\n", t2)
print("Shape:", t2.shape)       # torch.Size([2, 3])  — 2 rows, 3 columns

# Useful constructors
zeros = torch.zeros(3, 4)       # all zeros, shape 3x4
ones  = torch.ones(2, 2)        # all ones
rand  = torch.rand(2, 3)        # random values between 0 and 1
print("\nZeros (3x4):\n", zeros)
print("Ones  (2x2):\n", ones)
print("Random (2x3):\n", rand)

# --- Converting between NumPy and PyTorch ---
print("\n--- NumPy <-> PyTorch conversion ---")
np_array = np.array([10.0, 20.0, 30.0])
tensor_from_np = torch.from_numpy(np_array)   # numpy -> tensor
back_to_np     = tensor_from_np.numpy()       # tensor -> numpy

print("NumPy array:  ", np_array)
print("As tensor:    ", tensor_from_np)
print("Back to numpy:", back_to_np)

# --- Basic operations ---
print("\n--- Basic tensor operations ---")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("a + b  =", a + b)           # element-wise addition
print("a * b  =", a * b)           # element-wise multiplication
print("a @ b  =", a @ b)           # dot product (matrix multiply for 2-D)
print("a ** 2 =", a ** 2)          # element-wise power

# Reshape: change the shape without changing the data
c = torch.arange(12, dtype=torch.float32)   # [0, 1, 2, ..., 11]
print("\nOriginal (12 elements):", c)
c_reshaped = c.reshape(3, 4)
print("Reshaped to (3, 4):\n", c_reshaped)
c_3d = c.reshape(2, 2, 3)
print("Reshaped to (2, 2, 3):\n", c_3d)

# Indexing (same as NumPy)
print("\nFirst row of reshaped tensor:", c_reshaped[0])
print("Element at row 1, col 2:     ", c_reshaped[1, 2])
print("All rows, column 0:          ", c_reshaped[:, 0])


print("\n" + "=" * 60)
print("PART 2: Autograd — automatic differentiation")
print("=" * 60)

# =============================================================================
# FIRST PRINCIPLES: How Autograd Actually Works
# =============================================================================
#
# COMPUTATIONAL GRAPH:
#   Every operation in PyTorch builds a hidden graph behind the scenes.
#   Each node in the graph stores:
#     (a) The operation that produced it (add, multiply, power, etc.)
#     (b) Pointers to its input nodes
#   This graph records the ENTIRE sequence of math that produced the output.
#
#   Example: y = (x + 2) * x
#   Graph nodes:
#     [x] ---> [add: x+2] ---> [mul: (x+2)*x] = y
#     [x] ----------------------^
#
#   BACKWARD PASS: traverse the graph in REVERSE order.
#   At each node, apply the chain rule to compute the gradient of the
#   final output with respect to that node's inputs.
#
# REVERSE-MODE vs FORWARD-MODE AUTODIFF:
#   The Jacobian matrix J has shape (num_outputs x num_inputs).
#   - Forward-mode: computes ONE COLUMN of J per pass.
#     Good when there are FEW INPUTS (e.g., computing sensitivity to one param).
#   - Reverse-mode: computes ONE ROW of J per pass.
#     Good when there are FEW OUTPUTS.
#
#   Neural networks have a SCALAR loss function (1 output) but MILLIONS of
#   parameters (inputs). Reverse-mode computes ALL gradients in just ONE
#   backward pass! This is why PyTorch uses reverse-mode autodiff.
#
# COMPLEXITY:
#   - Forward pass: O(P) operations for P parameters (each param used once).
#   - Backward pass: ALSO O(P) — same order as the forward pass!
#     The backward pass is NOT more expensive than the forward pass.
#     (In practice, backward is ~2-3x the forward cost due to extra memory
#     access and bookkeeping, but it is the SAME asymptotic complexity.)
#
# EXERCISE: Draw the computational graph for f(x,y) = (x+y) * (x-y).
#   Nodes: a = x+y, b = x-y, f = a*b
#   Backward pass trace:
#     df/df = 1
#     df/da = b = (x-y),   df/db = a = (x+y)
#     df/dx = df/da * da/dx + df/db * db/dx = (x-y)*1 + (x+y)*1 = 2x
#     df/dy = df/da * da/dy + df/db * db/dy = (x-y)*1 + (x+y)*(-1) = -2y
#   Verify: f = x^2 - y^2, so df/dx = 2x, df/dy = -2y.  Correct!
# =============================================================================

# --- What is a gradient? ---
# Imagine y = x^2. The slope (gradient) at x=3 is dy/dx = 2*3 = 6.
# In neural networks, we have MILLIONS of such equations. Computing slopes by
# hand is impossible. PyTorch does it automatically!
#
# How: set requires_grad=True on any tensor whose gradient you want.
# After a forward computation, call .backward() to compute all gradients.
# Read the gradient from tensor.grad.

x = torch.tensor(3.0, requires_grad=True)   # x = 3, track its gradient
y = x ** 2                                  # y = x^2 = 9

print("x =", x)
print("y = x^2 =", y)

y.backward()   # compute dy/dx automatically

print("dy/dx (gradient) at x=3:", x.grad)   # should be 6.0  (= 2*3)

# A slightly more complex example
# z = 3*x^2 + 2*x + 1  =>  dz/dx = 6x + 2  =>  at x=4: 6*4+2 = 26
x2 = torch.tensor(4.0, requires_grad=True)
z  = 3 * x2**2 + 2 * x2 + 1
print("\nz = 3x^2 + 2x + 1  at x=4:", z.item())
z.backward()
print("dz/dx at x=4 (expected 26):", x2.grad.item())

# With a vector
# When the output is a vector/matrix, we call .backward(gradient) or use .sum()
x3 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y3 = (x3 ** 2).sum()   # sum so we get a scalar we can call .backward() on
y3.backward()
print("\nFor y = sum(x^2), dy/dx at x=[1,2,3]:", x3.grad)  # [2, 4, 6]


print("\n" + "=" * 60)
print("PART 3: nn.Module — the building block of neural networks")
print("=" * 60)

# --- What is nn.Module? ---
# Every neural network (or layer inside one) is a class that inherits from
# nn.Module. You define:
#   - __init__: create the layers / parameters
#   - forward:  describe how data flows through the network
#
# PyTorch handles the backward pass (gradients) automatically.

class SimpleLinearModel(nn.Module):
    """A one-layer linear model: output = W * input + b"""

    def __init__(self, input_size, output_size):
        super().__init__()
        # nn.Linear is a fully-connected layer. It holds two learnable
        # parameters: a weight matrix W and a bias vector b.
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # This is called when you do model(x).
        return self.linear(x)


model = SimpleLinearModel(input_size=3, output_size=1)
print("Model architecture:")
print(model)

# .parameters() gives all the learnable numbers inside the model
print("\nLearnable parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, values={param.data}")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal learnable numbers: {total_params}")

# Forward pass — give the model some data
sample_input = torch.tensor([[1.0, 2.0, 3.0]])   # shape (1, 3)
output = model(sample_input)
print("Sample input:", sample_input)
print("Model output:", output)   # (1, 1) tensor


print("\n" + "=" * 60)
print("PART 4: Training loop — linear regression in PyTorch")
print("=" * 60)

# --- The problem ---
# Remember linear regression from Section 5? We manually computed gradients
# and updated weights. PyTorch makes this MUCH simpler.
#
# Goal: learn y = 2*x + 1  from noisy data.

# --- Generate training data ---
torch.manual_seed(42)
X_data = torch.randn(100, 1)             # 100 random x values
y_data = 2.0 * X_data + 1.0             # true relationship: y = 2x + 1
y_data += 0.1 * torch.randn(100, 1)     # add a little noise

# --- Build model, loss function, and optimizer ---

# Model: one linear layer (learns slope and intercept)
lin_model = nn.Linear(in_features=1, out_features=1)

# Loss: Mean Squared Error — measures how far predictions are from reality
#   MSE = average of (prediction - true_value)^2
loss_fn = nn.MSELoss()

# Optimizer: Stochastic Gradient Descent — adjusts weights to reduce loss
#   lr = learning rate (step size)
optimizer = optim.SGD(lin_model.parameters(), lr=0.1)

# --- Training loop ---
print("Training: learning y = 2x + 1 from data\n")
print(f"{'Epoch':>6}  {'Loss':>10}  {'Weight':>10}  {'Bias':>10}")
print("-" * 45)

for epoch in range(50):
    # 1. Forward pass: compute predictions
    predictions = lin_model(X_data)

    # 2. Compute loss
    loss = loss_fn(predictions, y_data)

    # 3. Zero out old gradients (important! gradients accumulate by default)
    optimizer.zero_grad()

    # 4. Backward pass: compute gradients
    loss.backward()

    # 5. Update weights
    optimizer.step()

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        w = lin_model.weight.item()
        b = lin_model.bias.item()
        print(f"  {epoch+1:>4}  {loss.item():>10.4f}  {w:>10.4f}  {b:>10.4f}")

w_final = lin_model.weight.item()
b_final = lin_model.bias.item()
print(f"\nTrue relationship:    y = 2.0000*x + 1.0000")
print(f"Learned relationship: y = {w_final:.4f}*x + {b_final:.4f}")
print("\nPyTorch handled ALL the gradient math for us!")
print("Compare that to the manual version in Section 5...")


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# --- Tensor shape tests ---
assert t1.shape == torch.Size([4]), "t1 should be 1-D with 4 elements"
assert t2.shape == torch.Size([2, 3]), "t2 should be 2x3"
assert zeros.shape == torch.Size([3, 4]), "zeros should be 3x4"
assert ones.shape == torch.Size([2, 2]), "ones should be 2x2"
assert rand.shape == torch.Size([2, 3]), "rand should be 2x3"

# --- Dtype tests ---
assert t1.dtype == torch.float32, "t1 should be float32"
assert c.dtype == torch.float32, "c should be float32"

# --- Reshape tests ---
assert c_reshaped.shape == torch.Size([3, 4]), "c_reshaped should be 3x4"
assert c_3d.shape == torch.Size([2, 2, 3]), "c_3d should be 2x2x3"

# --- NumPy conversion test ---
assert tensor_from_np.shape == torch.Size([3]), "tensor_from_np should have 3 elements"
assert isinstance(back_to_np, type(np_array)), "back_to_np should be a numpy array"

# --- Basic operation tests (re-create a,b since b was overwritten by training loop) ---
_a = torch.tensor([1.0, 2.0, 3.0])
_b = torch.tensor([4.0, 5.0, 6.0])
assert torch.allclose(_a + _b, torch.tensor([5.0, 7.0, 9.0])), "a+b wrong"
assert torch.allclose(_a * _b, torch.tensor([4.0, 10.0, 18.0])), "a*b wrong"
assert torch.allclose(_a @ _b, torch.tensor(32.0)), "dot product wrong"

# --- Autograd tests ---
# dy/dx at x=3 for y=x^2 should be 6.0
assert torch.allclose(x.grad, torch.tensor(6.0)), "gradient of x^2 at x=3 should be 6.0"
# dz/dx at x=4 for z=3x^2+2x+1 should be 26.0
assert abs(x2.grad.item() - 26.0) < 1e-4, "gradient of 3x^2+2x+1 at x=4 should be 26.0"
# dy/dx for y=sum(x^2) at x=[1,2,3] should be [2,4,6]
assert torch.allclose(x3.grad, torch.tensor([2.0, 4.0, 6.0])), "vector autograd wrong"

# --- nn.Module tests ---
assert total_params == 4, "SimpleLinearModel(3->1) should have 4 params (3 weights + 1 bias)"
assert output.shape == torch.Size([1, 1]), "model output shape should be (1, 1)"

# --- Training convergence test ---
# After 50 epochs learning y=2x+1, weights should be close to true values
assert abs(w_final - 2.0) < 0.1, f"learned weight should be close to 2.0, got {w_final:.4f}"
assert abs(b_final - 1.0) < 0.1, f"learned bias should be close to 1.0, got {b_final:.4f}"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
print("\n" + "=" * 60)
print("EXERCISES (try these yourself!)")
print("=" * 60)
print("""
1. Tensor practice:
   - Create a 4x4 tensor of random numbers.
   - Extract the diagonal elements (hint: look up torch.diag).
   - Compute the sum of each row (hint: .sum(dim=1)).

2. Autograd exploration:
   - Define f(x) = x^3 - 5*x^2 + 3
   - Compute df/dx at x = 2.0 using autograd.
   - Verify by hand: df/dx = 3x^2 - 10x, so at x=2 it should be 3*4 - 20 = -8.

3. Custom nn.Module:
   - Create a TwoLayerNet class with:
       - Layer 1: Linear(4, 8) followed by ReLU (use nn.ReLU())
       - Layer 2: Linear(8, 1)
   - Print the number of parameters.
   - Pass a random (5, 4) input through it and print the output shape.

4. Training challenge:
   - Modify the training loop above to learn y = -3*x + 5.
   - Try different learning rates (0.01, 0.1, 0.5).
   - What happens when the learning rate is too large?
   (Hint: just change the true relationship in the data generation step.)
""")
