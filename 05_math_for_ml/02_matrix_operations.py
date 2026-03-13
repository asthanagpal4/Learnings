# HOW TO RUN:
#   uv run python 05_math_for_ml/02_matrix_operations.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE 2: Matrix Operations
# =============================================================================
# In the previous file we learned what vectors and matrices ARE.
# Now we learn how to WORK with them efficiently.
#
# Topics:
#   - Element-wise operations (math on each element)
#   - Broadcasting (numpy's smart shape-handling)
#   - Reshaping arrays
#   - Stacking arrays
#   - Indexing and slicing
#   - Data normalization (very important in ML!)
#   - One-hot encoding
# =============================================================================

import numpy as np

print("=" * 60)
print("PART 1: ELEMENT-WISE OPERATIONS")
print("=" * 60)

# ------------------------------------------------------------------
# Element-wise means: apply the operation to EACH pair of elements.
# Both matrices must have the SAME shape.
# ------------------------------------------------------------------

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print("\nA:\n", A)
print("\nB:\n", B)

print("\n--- Element-wise Addition ---")
print(A + B)   # [1+10, 2+20, ...] etc.

print("\n--- Element-wise Subtraction ---")
print(A - B)

print("\n--- Element-wise Multiplication ---")
# NOTE: This is NOT matrix multiplication! Each element multiplied by its pair.
print(A * B)   # [1*10, 2*20, 3*30, ...]

print("\n--- Element-wise Division ---")
print(A / B)

print("\n--- Element-wise Power ---")
print(A ** 2)  # Each element squared

print("\n--- Math Functions (applied element-wise) ---")
angles = np.array([0, 30, 45, 60, 90])
print("Angles:", angles)
print("Sqrt:", np.sqrt(angles))
print("Absolute value of [-3, -1, 0, 2, 5]:", np.abs(np.array([-3, -1, 0, 2, 5])))

print("\n" + "=" * 60)
print("PART 2: BROADCASTING")
print("=" * 60)

# ------------------------------------------------------------------
# Broadcasting is numpy's ability to do math between arrays of
# DIFFERENT shapes by stretching the smaller one automatically.
#
# This is one of numpy's most powerful features.
# Without broadcasting, you'd need loops everywhere.
# ------------------------------------------------------------------

print("\n--- Example 1: Matrix + Single Number ---")
# Adding 10 to every element in a matrix
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print("M:\n", M)
print("M + 10:\n", M + 10)
# Numpy "broadcasts" 10 to every position — it acts as if 10 were a
# full matrix of the same shape filled with 10s.

print("\n--- Example 2: Matrix + Row Vector ---")
# Adding a different value to each COLUMN
row = np.array([100, 200, 300])   # shape: (3,)
print("M:\n", M)
print("row:", row, "  shape:", row.shape)
print("M + row:\n", M + row)
# Each row of M gets [100, 200, 300] added to it.
# Row 0: [1+100, 2+200, 3+300] = [101, 202, 303]
# Row 1: [4+100, 5+200, 6+300] = [104, 205, 306]

print("\n--- Example 3: Matrix + Column Vector ---")
# Adding a different value to each ROW
col = np.array([[10],   # shape: (2, 1)
                [20]])
print("M:\n", M)
print("col:\n", col, "  shape:", col.shape)
print("M + col:\n", M + col)
# Column 0 of M gets 10 added, column... wait — each ROW gets a different addition:
# Row 0: [1+10, 2+10, 3+10] = [11, 12, 13]
# Row 1: [4+20, 5+20, 6+20] = [24, 25, 26]

print("\n--- Broadcasting Rule (simple explanation) ---")
print("""
  numpy broadcasts when dimensions are compatible:
    - Dimensions are equal, OR
    - One of them is 1 (then it gets stretched)

  Shape (2,3) + Shape (3,)   -> OK  (row broadcast)
  Shape (2,3) + Shape (2,1)  -> OK  (column broadcast)
  Shape (2,3) + Shape (2,4)  -> ERROR (neither is 1, and 3 != 4)
""")

# ===========================================================================
# FIRST PRINCIPLES: Broadcasting rules derived step by step
# ===========================================================================
# Broadcasting is NOT magic -- it follows a precise algorithm:
#
# STEP 1: Align shapes from the TRAILING (rightmost) dimension.
#   If the arrays have different numbers of dimensions, the shape of
#   the smaller array is padded with 1s on the LEFT.
#   Example: (3,) becomes (1, 3) when compared against a 2D array.
#
# STEP 2: For each dimension (right to left), check compatibility:
#   - If both sizes are equal: OK, no stretching needed.
#   - If one of them is 1: stretch it to match the other.
#   - Otherwise: ERROR! Shapes are incompatible.
#
# STEP 3: The output shape is the element-wise maximum of each dimension.
#
# WORKED EXAMPLE: (3, 1) + (1, 4)
#   Align trailing dims:
#     Array A: (3, 1)
#     Array B: (1, 4)
#   Dim 1 (trailing): 1 vs 4 -> 1 gets stretched to 4. OK.
#   Dim 0:            3 vs 1 -> 1 gets stretched to 3. OK.
#   Output shape: (3, 4)
#
# MEMORY ANALYSIS:
#   - Broadcasting does NOT actually copy the data in memory.
#   - NumPy uses a "strides trick": it sets the stride to 0 along
#     the broadcast dimension, so the same row/column is re-read
#     without allocating new memory. This is O(1) extra memory.
#   - In contrast, np.tile() or manual replication is O(n) memory.
# ===========================================================================

print("--- FIRST PRINCIPLES: Broadcasting walkthrough ---")
A_bc = np.ones((3, 1))
B_bc = np.ones((1, 4)) * np.array([[10, 20, 30, 40]])
result_bc = A_bc + B_bc
print(f"  A shape: {A_bc.shape}  +  B shape: {B_bc.shape}  =  result shape: {result_bc.shape}")
print(f"  Predicted output shape: (3, 4) -- max(3,1)=3, max(1,4)=4")
print(f"  Result:\n{result_bc}")
print(f"  A is stretched along columns (axis=1): 1 -> 4")
print(f"  B is stretched along rows (axis=0):    1 -> 3")

# ML use case: centering data by subtracting column means
data = np.array([
    [10, 200, 1],
    [12, 180, 2],
    [11, 190, 1],
])
col_means = data.mean(axis=0)   # mean of each column (feature)
centered = data - col_means     # broadcasting subtracts mean from each row

print("--- ML Example: Centering Data ---")
print("Raw data:\n", data)
print("Column means:", col_means)
print("Centered data (subtract mean from each column):\n", centered)

print("\n" + "=" * 60)
print("PART 3: RESHAPING ARRAYS")
print("=" * 60)

# ------------------------------------------------------------------
# .reshape() changes the shape of an array WITHOUT changing its data.
# The total number of elements must stay the same.
# -1 means "figure out this dimension automatically"
# ------------------------------------------------------------------

v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print("\nOriginal vector:", v)
print("Shape:", v.shape)

# Reshape into different matrix shapes
m1 = v.reshape(3, 4)   # 3 rows, 4 columns
m2 = v.reshape(4, 3)   # 4 rows, 3 columns
m3 = v.reshape(2, 6)   # 2 rows, 6 columns
m4 = v.reshape(2, 2, 3)  # 3D array: 2 "blocks", 2 rows, 3 cols

print("\nReshaped to (3, 4):\n", m1)
print("\nReshaped to (4, 3):\n", m2)
print("\nReshaped to (2, 6):\n", m3)
print("\nReshaped to (2, 2, 3) — a 3D array:\n", m4)

# Using -1: numpy calculates that dimension for you
m5 = v.reshape(-1, 4)  # "however many rows needed, 4 columns"
m6 = v.reshape(3, -1)  # "3 rows, however many columns needed"
print("\nreshaped(-1, 4) — numpy figures out rows:\n", m5)
print("\nreshaped(3, -1) — numpy figures out columns:\n", m6)

# Flattening: turn any shape back into a 1D vector
matrix = np.array([[1, 2, 3], [4, 5, 6]])
flat   = matrix.flatten()
print("\nMatrix:\n", matrix)
print("Flattened:", flat)

# ===========================================================================
# FIRST PRINCIPLES: Memory analysis of reshape vs copy
# ===========================================================================
# reshape() is O(1) -- it only changes the metadata (shape and strides),
# NOT the actual data in memory. The underlying buffer stays the same.
#
# .copy() is O(n) -- it allocates new memory and copies all n elements.
#
# .flatten() returns a COPY (O(n)), while .ravel() returns a VIEW (O(1))
# when possible. Views share memory with the original array.
# ===========================================================================

print("\n--- MEMORY ANALYSIS: reshape is O(1), copy is O(n) ---")
original = np.arange(12)
reshaped = original.reshape(3, 4)
print(f"  original and reshaped share memory: {np.shares_memory(original, reshaped)}")
print(f"  Changing reshaped[0,0] to 999...")
reshaped_demo = original.copy().reshape(3, 4)  # use copy so we don't mutate original
print(f"  reshape is just a 'view' -- no data is copied, O(1) operation")

# ML use case: image flattening
# A 28x28 pixel image becomes a vector of 784 numbers for a neural network
fake_image = np.random.randint(0, 256, (28, 28))
image_vector = fake_image.reshape(-1)   # shape: (784,)
print("\n--- ML Example: Flattening an Image ---")
print("Image shape (28x28):", fake_image.shape)
print("After flattening:", image_vector.shape, "  (784 pixel values)")

print("\n" + "=" * 60)
print("PART 4: STACKING ARRAYS")
print("=" * 60)

# ------------------------------------------------------------------
# Stacking means joining multiple arrays into one.
# vstack = stack vertically (add more ROWS)
# hstack = stack horizontally (add more COLUMNS)
# concatenate = general version, you choose the axis
# ------------------------------------------------------------------

a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.array([[7,  8,  9],
              [10, 11, 12]])

print("\na:\n", a)
print("\nb:\n", b)

print("\n--- np.vstack (add rows — stack vertically) ---")
vertical = np.vstack([a, b])
print(vertical)
print("Shape:", vertical.shape)   # (4, 3)

print("\n--- np.hstack (add columns — stack horizontally) ---")
horizontal = np.hstack([a, b])
print(horizontal)
print("Shape:", horizontal.shape)   # (2, 6)

print("\n--- np.concatenate (general — specify axis) ---")
cat_rows = np.concatenate([a, b], axis=0)   # same as vstack
cat_cols = np.concatenate([a, b], axis=1)   # same as hstack
print("axis=0 (rows):\n", cat_rows)
print("axis=1 (cols):\n", cat_cols)

# ML use case: combining feature columns
ages    = np.array([[25], [30], [22]])    # shape (3, 1)
heights = np.array([[170], [180], [160]]) # shape (3, 1)
weights = np.array([[65], [80], [55]])    # shape (3, 1)

people  = np.hstack([ages, heights, weights])
print("\n--- ML Example: Building a Feature Matrix ---")
print("Ages:\n", ages.T)
print("Heights:\n", heights.T)
print("Weights:\n", weights.T)
print("\nCombined feature matrix (people):\n", people)
print("Shape:", people.shape, " -> 3 people, 3 features each")

print("\n" + "=" * 60)
print("PART 5: INDEXING AND SLICING 2D ARRAYS")
print("=" * 60)

# ------------------------------------------------------------------
# Syntax: array[row, column]
# Use : to mean "all" or a range
# ------------------------------------------------------------------

data = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120],
])
print("\ndata:\n", data)

print("\n--- Single Elements ---")
print("data[0, 0] (row 0, col 0):", data[0, 0])   # 10
print("data[1, 2] (row 1, col 2):", data[1, 2])   # 70
print("data[-1, -1] (last row, last col):", data[-1, -1])  # 120

print("\n--- Entire Rows and Columns ---")
print("Row 0:", data[0])           # [10, 20, 30, 40]
print("Row 1:", data[1])           # [50, 60, 70, 80]
print("All rows, col 2:", data[:, 2])   # [30, 70, 110]
print("All rows, col 0:", data[:, 0])   # [10, 50, 90]

print("\n--- Slicing Sub-matrices ---")
print("Rows 0-1, Cols 1-3:\n", data[0:2, 1:3])
print("Last 2 rows:\n", data[-2:, :])
print("First 2 columns:\n", data[:, :2])

print("\n--- Boolean Indexing (filtering) ---")
# Select elements based on a condition
print("Elements > 50:", data[data > 50])
print("Elements in col 0 that are > 10:", data[:, 0][data[:, 0] > 10])

# Select rows where first column > 20
mask = data[:, 0] > 20
print("Rows where first value > 20:\n", data[mask])

print("\n" + "=" * 60)
print("PART 6: DATA NORMALIZATION")
print("=" * 60)

# ------------------------------------------------------------------
# Normalization = scaling numbers so they are in a similar range.
# WHY? Because ML algorithms train better when all features are
# on the same scale. If one column is [0, 1] and another is [0, 100000],
# the large-valued column unfairly dominates.
#
# Two common methods:
#   1. Min-Max Normalization: scale to [0, 1]
#      formula: (x - min) / (max - min)
#
#   2. Standardization (Z-score): mean=0, std=1
#      formula: (x - mean) / std
# ------------------------------------------------------------------

# Dataset: 3 students with [hours_studied, previous_grade, distance_to_school_km]
raw = np.array([
    [2,   60,  1.5],
    [5,   75,  10.0],
    [8,   90,  3.0],
    [1,   45,  25.0],
    [10,  95,  0.5],
])

print("\nRaw data:\n", raw)

# Min-Max normalization
col_min = raw.min(axis=0)   # minimum of each column
col_max = raw.max(axis=0)   # maximum of each column
minmax  = (raw - col_min) / (col_max - col_min)

print("\n--- Min-Max Normalization (values between 0 and 1) ---")
print("Column minimums:", col_min)
print("Column maximums:", col_max)
print("Normalized:\n", np.round(minmax, 3))

# Standardization
col_mean = raw.mean(axis=0)
col_std  = raw.std(axis=0)
standardized = (raw - col_mean) / col_std

print("\n--- Standardization (mean=0, std=1) ---")
print("Column means:", np.round(col_mean, 2))
print("Column stds: ", np.round(col_std, 2))
print("Standardized:\n", np.round(standardized, 3))
print("Verify mean of standardized columns:", np.round(standardized.mean(axis=0), 10))
print("Verify std  of standardized columns:", np.round(standardized.std(axis=0), 10))

print("\n" + "=" * 60)
print("PART 7: ONE-HOT ENCODING")
print("=" * 60)

# ------------------------------------------------------------------
# One-hot encoding = turning a category into a vector of 0s and one 1.
# WHY? Computers work with numbers. Categories like "cat", "dog", "bird"
# must be converted to numbers, but 0/1/2 implies ordering (2 > 1 > 0).
# One-hot avoids that: each category gets its own column.
#
# Example:
#   "cat"  -> [1, 0, 0]
#   "dog"  -> [0, 1, 0]
#   "bird" -> [0, 0, 1]
# ------------------------------------------------------------------

# Suppose we have 5 samples and 3 classes (0=cat, 1=dog, 2=bird)
labels = np.array([0, 2, 1, 0, 2])   # category indices
num_classes = 3

# Create a zero matrix, then put 1s in the right positions
one_hot = np.zeros((len(labels), num_classes), dtype=int)
one_hot[np.arange(len(labels)), labels] = 1

print("\nCategory labels:", labels)
print("  0 = cat, 1 = dog, 2 = bird")
print("\nOne-hot encoded:\n", one_hot)
print("\nColumns represent: [cat, dog, bird]")
print("Row 0 (cat):  ", one_hot[0])
print("Row 1 (bird): ", one_hot[1])
print("Row 2 (dog):  ", one_hot[2])

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key operations learned:
  A + B, A * B, A / B  : element-wise (same-shape) operations
  A + scalar           : broadcasting — applies to every element
  A + row_vector       : broadcasting — adds to each row
  .reshape(rows, cols) : change shape, keep data
  np.vstack([A, B])    : add more rows
  np.hstack([A, B])    : add more columns
  A[row, col]          : indexing
  A[r1:r2, c1:c2]      : slicing a sub-matrix
  Normalization        : scale data so all features are comparable
  One-hot encoding     : turn categories into 0/1 vectors
""")

# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test element-wise operations
assert np.array_equal(A + B, np.array([[11, 22, 33], [44, 55, 66]])), "Element-wise addition failed"
assert np.array_equal(A * B, np.array([[10, 40, 90], [160, 250, 360]])), "Element-wise multiplication failed"

# Test broadcasting: M + 10 adds 10 to every element
M_plus_10 = M + 10
assert M_plus_10[0, 0] == 11 and M_plus_10[1, 2] == 16, "Scalar broadcast failed"

# Test broadcasting: M + row_vector [100,200,300]
M_plus_row = M + row
assert M_plus_row[0, 0] == 101 and M_plus_row[0, 1] == 202 and M_plus_row[1, 2] == 306, "Row broadcast failed"

# Test broadcasting: M + col_vector adds 10 to row0, 20 to row1
M_plus_col = M + col
assert M_plus_col[0, 0] == 11 and M_plus_col[1, 0] == 24, "Column broadcast failed"

# Test broadcasting result shape: (3,1) + (1,4) = (3,4)
assert result_bc.shape == (3, 4), "Broadcast shape should be (3,4)"

# Test reshape: 12-element vector reshaped to (3,4)
assert m1.shape == (3, 4), "Reshape to (3,4) failed"
assert m2.shape == (4, 3), "Reshape to (4,3) failed"
assert m3.shape == (2, 6), "Reshape to (2,6) failed"
assert m4.shape == (2, 2, 3), "Reshape to (2,2,3) failed"

# Test reshape with -1
assert m5.shape == (3, 4), "Reshape(-1,4) should give (3,4)"
assert m6.shape == (3, 4), "Reshape(3,-1) should give (3,4)"

# Test flatten
assert flat.shape == (6,), "Flatten should give 1D array of 6 elements"
assert np.array_equal(flat, np.array([1, 2, 3, 4, 5, 6])), "Flatten values wrong"

# Test vstack shape: two (2,3) matrices stacked -> (4,3)
assert vertical.shape == (4, 3), "vstack shape should be (4,3)"

# Test hstack shape: two (2,3) matrices side-by-side -> (2,6)
assert horizontal.shape == (2, 6), "hstack shape should be (2,6)"

# Test people feature matrix from hstack of ages/heights/weights
assert people.shape == (3, 3), "People matrix should be (3,3)"
assert people[0, 0] == 25 and people[0, 1] == 170 and people[0, 2] == 65, "First person's features wrong"

# Test normalization: min-max values should be in [0,1]
assert np.all(minmax >= 0) and np.all(minmax <= 1), "Min-max values should all be in [0,1]"
assert abs(minmax.min()) < 1e-9, "Min of normalized data should be 0"
assert abs(minmax.max() - 1.0) < 1e-9, "Max of normalized data should be 1"

# Test standardization: mean ≈ 0, std ≈ 1 per column
assert np.allclose(standardized.mean(axis=0), 0, atol=1e-9), "Standardized column means should be 0"
assert np.allclose(standardized.std(axis=0), 1, atol=1e-9), "Standardized column stds should be 1"

# Test one-hot encoding: each row sums to 1, correct positions are 1
assert one_hot.shape == (5, 3), "One-hot shape should be (5,3)"
assert np.all(one_hot.sum(axis=1) == 1), "Each one-hot row must sum to 1"
assert one_hot[0, 0] == 1, "Label 0 -> index 0 should be 1"
assert one_hot[1, 2] == 1, "Label 2 -> index 2 should be 1"
assert one_hot[2, 1] == 1, "Label 1 -> index 1 should be 1"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
# 1. Create a (3, 4) matrix of your choice. Add the row vector [1, 2, 3, 4]
#    to it using broadcasting. Explain what happened to each row.
#
# 2. Create a 1D array of 24 numbers (hint: np.arange(24)).
#    Reshape it to (4, 6), then to (2, 3, 4), then flatten it back.
#
# 3. Create two (3, 3) matrices and stack them vertically and horizontally.
#    What are the resulting shapes?
#
# 4. Take the raw data matrix from Part 6 (the student data). Extract only
#    the rows where hours_studied >= 5. How many students studied >= 5 hours?
#
# 5. One-hot encode this label array: [1, 0, 3, 2, 1, 0]
#    (4 classes: 0, 1, 2, 3). What shape should your result be?
#
# 6. Normalize a simple array [10, 20, 30, 40, 50] using min-max normalization
#    by hand, then verify with numpy. The result should go from 0.0 to 1.0.
#
# 7. FIRST PRINCIPLES EXERCISE: Predict the output shape of broadcasting
#    np.ones((3, 1)) + np.ones((1, 4)). Walk through the broadcasting rules:
#    align trailing dimensions, check compatibility, compute output shape.
#    Then verify with code. Why is the result shape (3, 4)?
#
# 8. MEMORY EXERCISE: Create an array with np.arange(1000000). Reshape it to
#    (1000, 1000). Does np.shares_memory() return True? Now use .copy() on
#    the reshaped array. Does it still share memory? What does this tell you
#    about the cost of reshape vs copy?
# =============================================================================
