# HOW TO RUN:
#   uv run python 05_math_for_ml/01_vectors_and_matrices.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# FILE 1: Vectors and Matrices
# =============================================================================
# Before we can understand machine learning, we need to understand how data
# is stored and manipulated mathematically.
#
# The two most important structures are:
#   - VECTORS: a list of numbers (like a single data point)
#   - MATRICES: a table of numbers (like a whole dataset)
#
# NumPy is the Python library that makes working with these fast and easy.
# =============================================================================

import numpy as np

print("=" * 60)
print("PART 1: VECTORS")
print("=" * 60)

# ------------------------------------------------------------------
# What is a vector?
# ------------------------------------------------------------------
# A vector is just an ordered list of numbers.
# Example: a person's features -> [age, height_cm, weight_kg]
# Example: a 2D point on a map -> [x, y]
# Example: word counts in a sentence -> [3, 0, 1, 2, ...]
#
# In NumPy, we create vectors using np.array()

person = np.array([25, 170, 65])   # age=25, height=170cm, weight=65kg
print("\nA person described as a vector:", person)
print("Shape (how many elements):", person.shape)   # (3,) means 3 elements
print("Data type:", person.dtype)

point_2d = np.array([3.0, 4.0])
print("\nA point in 2D space:", point_2d)

# ------------------------------------------------------------------
# Vector addition
# ------------------------------------------------------------------
# Adding two vectors adds element by element.
# Think of it as combining two sets of measurements.

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

v_sum = v1 + v2   # [1+4, 2+5, 3+6] = [5, 7, 9]
print("\n--- Vector Addition ---")
print("v1 =", v1)
print("v2 =", v2)
print("v1 + v2 =", v_sum)

# ------------------------------------------------------------------
# Scalar multiplication
# ------------------------------------------------------------------
# Multiplying a vector by a single number (scalar) scales all elements.
# Example: doubling all features of a data point

v3 = np.array([2, 4, 6])
scaled = 3 * v3   # [6, 12, 18]
print("\n--- Scalar Multiplication ---")
print("v3 =", v3)
print("3 * v3 =", scaled)

# ------------------------------------------------------------------
# Dot product — what does it mean?
# ------------------------------------------------------------------
# The dot product of two vectors gives ONE number.
# Formula: dot(a, b) = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + ...
#
# What does it tell us?
#   - If two vectors point in the SAME direction -> large positive dot product
#   - If they are PERPENDICULAR (unrelated) -> dot product is 0
#   - If they point in OPPOSITE directions -> negative dot product
#
# In ML, dot products measure SIMILARITY between two vectors.
# They are used in almost every neural network layer!

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_manual  = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]   # 1*4 + 2*5 + 3*6 = 32
dot_numpy   = np.dot(a, b)
dot_operator = a @ b   # @ is the dot/matrix product operator in Python

print("\n--- Dot Product ---")
print("a =", a)
print("b =", b)
print("Manual calculation (1*4 + 2*5 + 3*6):", dot_manual)
print("np.dot(a, b):", dot_numpy)
print("a @ b:", dot_operator)
print("All three give the same answer:", dot_manual == dot_numpy == dot_operator)

# ===========================================================================
# FIRST PRINCIPLES: Deriving dot product from geometry
# ===========================================================================
# The GEOMETRIC definition of dot product is:
#   a . b = |a| |b| cos(theta)
#
# where theta is the angle between vectors a and b.
#
# Rearranging: cos(theta) = (a . b) / (|a| |b|)
#
# This IS cosine similarity! It measures the angle between two vectors,
# ignoring their magnitudes.
#
# WHY does dot product measure similarity? (Projection interpretation)
# -------------------------------------------------------------------
# The dot product a . b equals |b| times the LENGTH of a's projection
# onto b. In other words, it asks: "how much of a points in the same
# direction as b?" If a and b point the same way, the projection is
# large and positive. If perpendicular, zero. If opposite, negative.
#
# This is why dot products appear everywhere in ML:
#   - In a neural network layer, z = w . x + b asks "how much does
#     the input x align with the learned pattern w?"
#   - In attention mechanisms (Transformers/LLMs), similarity between
#     queries and keys is computed as a dot product.
#
# COMPLEXITY ANALYSIS:
#   - Dot product of two n-dimensional vectors: O(n)
#     (n multiplications + n-1 additions)
#   - Matrix multiply C = A @ B where A is (m x n), B is (n x p): O(m * n * p)
#     Each of the m*p entries in C requires a dot product of length n.
#     For square matrices (m = n = p), this is O(n^3).
# ===========================================================================

# --- Demo: Cosine similarity from first principles ---
print("\n--- FIRST PRINCIPLES: Dot Product = Cosine Similarity ---")
vec_a = np.array([3.0, 4.0])
vec_b = np.array([4.0, 3.0])

dot_ab = np.dot(vec_a, vec_b)
mag_a = np.linalg.norm(vec_a)
mag_b = np.linalg.norm(vec_b)
cos_theta = dot_ab / (mag_a * mag_b)
theta_radians = np.arccos(np.clip(cos_theta, -1, 1))
theta_degrees = np.degrees(theta_radians)

print(f"  a = {vec_a},  b = {vec_b}")
print(f"  a . b = {dot_ab}")
print(f"  |a| = {mag_a},  |b| = {mag_b}")
print(f"  cos(theta) = a.b / (|a||b|) = {dot_ab} / ({mag_a} * {mag_b}) = {cos_theta:.6f}")
print(f"  theta = {theta_degrees:.2f} degrees")
print(f"  (Small angle -> vectors are similar!)")

# --- Demo: For UNIT vectors, dot product IS the cosine ---
print("\n--- EXERCISE DEMO: Unit vectors ---")
print("  For unit vectors (length = 1), |a| = |b| = 1, so:")
print("  a . b = 1 * 1 * cos(theta) = cos(theta)")
print("  The dot product directly gives the cosine of the angle!")
unit_a = vec_a / np.linalg.norm(vec_a)
unit_b = vec_b / np.linalg.norm(vec_b)
print(f"  unit_a = a / |a| = {np.round(unit_a, 4)}, length = {np.linalg.norm(unit_a):.4f}")
print(f"  unit_b = b / |b| = {np.round(unit_b, 4)}, length = {np.linalg.norm(unit_b):.4f}")
print(f"  unit_a . unit_b = {np.dot(unit_a, unit_b):.6f}")
print(f"  cos(theta)      = {cos_theta:.6f}")
print(f"  They match! QED: for unit vectors, dot product = cos(angle)")

# --- Complexity demo ---
print("\n--- COMPLEXITY ANALYSIS DEMO ---")
print("  Dot product: O(n) for n-dimensional vectors")
print("  Matrix multiply (m x n) @ (n x p) = O(m * n * p)")
print(f"  Example: (3,2) @ (2,3) = O(3*2*3) = O(18) multiplications")
print(f"  Square matrices (n x n) @ (n x n) = O(n^3)")

# Similarity example
same_direction   = np.array([1, 0])
similar          = np.array([0.9, 0.1])
perpendicular    = np.array([0, 1])
opposite         = np.array([-1, 0])

print("\nDot product similarity examples:")
print("  [1,0] . [0.9,0.1] (similar)     =", np.dot(same_direction, similar))
print("  [1,0] . [0,1]     (unrelated)   =", np.dot(same_direction, perpendicular))
print("  [1,0] . [-1,0]    (opposite)    =", np.dot(same_direction, opposite))

print("\n" + "=" * 60)
print("PART 2: MATRICES")
print("=" * 60)

# ------------------------------------------------------------------
# What is a matrix?
# ------------------------------------------------------------------
# A matrix is a 2D table of numbers — rows and columns.
#
# In machine learning, a matrix often represents a DATASET where:
#   - Each ROW    = one data sample (one person, one image, one sentence)
#   - Each COLUMN = one feature (age, height, weight, pixel value...)
#
# Example: 3 students, each described by [hours_studied, sleep_hours, score]

students = np.array([
    [5, 7, 80],   # student 1: studied 5 hrs, slept 7 hrs, scored 80
    [2, 5, 55],   # student 2: studied 2 hrs, slept 5 hrs, scored 55
    [8, 8, 92],   # student 3: studied 8 hrs, slept 8 hrs, scored 92
])

print("\nStudent dataset as a matrix:")
print(students)
print("\nShape (rows, columns):", students.shape)   # (3, 3)
print("Number of students (rows):", students.shape[0])
print("Number of features (columns):", students.shape[1])

# Accessing rows and columns
print("\nFirst student's data (row 0):", students[0])
print("All students' hours studied (column 0):", students[:, 0])
print("All students' scores (column 2):", students[:, 2])

# ------------------------------------------------------------------
# Creating matrices in different ways
# ------------------------------------------------------------------

zeros_matrix = np.zeros((3, 4))     # 3 rows, 4 columns, all zeros
ones_matrix  = np.ones((2, 3))      # 2 rows, 3 columns, all ones
eye_matrix   = np.eye(3)            # 3x3 identity matrix (diagonal = 1)
random_matrix = np.random.rand(2, 3) # random values between 0 and 1

print("\n--- Special Matrices ---")
print("Zeros matrix (3x4):\n", zeros_matrix)
print("\nOnes matrix (2x3):\n", ones_matrix)
print("\nIdentity matrix (3x3):\n", eye_matrix)
print("\nRandom matrix (2x3):\n", np.round(random_matrix, 3))

# ------------------------------------------------------------------
# Matrix multiplication
# ------------------------------------------------------------------
# This is one of the most important operations in deep learning!
# A neural network layer is basically: output = input_matrix @ weight_matrix
#
# Rule: (m x n) @ (n x p) = (m x p)
# The INNER dimensions must match!
#
# Think of it as: each output value is a dot product of a row and a column.

A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])  # shape: (3, 2)

B = np.array([
    [7, 8, 9],
    [10, 11, 12]
])  # shape: (2, 3)

C = A @ B   # (3,2) @ (2,3) = (3,3)

print("\n--- Matrix Multiplication ---")
print("A (3x2):\n", A)
print("\nB (2x3):\n", B)
print("\nA @ B (3x3):\n", C)
print("\nA.shape:", A.shape, "  B.shape:", B.shape, "  result shape:", C.shape)
print("\nnp.dot(A, B) gives same result:", np.all(np.dot(A, B) == C))

# Manual check for top-left element: row0(A) . col0(B) = 1*7 + 2*10 = 27
print("Manual check [0,0]: 1*7 + 2*10 =", 1*7 + 2*10, "  Got:", C[0, 0])

# ------------------------------------------------------------------
# Transpose
# ------------------------------------------------------------------
# Transposing a matrix flips it: rows become columns, columns become rows.
# If A has shape (3, 2), then A.T has shape (2, 3).
#
# This is used constantly in ML — for example, when computing gradients.

M = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("\n--- Transpose ---")
print("M (2x3):\n", M)
print("\nM.T (3x2):\n", M.T)
print("M.shape:", M.shape, "  M.T.shape:", M.T.shape)

# ------------------------------------------------------------------
# Real-world ML connection: features matrix
# ------------------------------------------------------------------
# In ML, your data is almost always stored as a matrix.
# Rows = samples, Columns = features.
# This is called a "feature matrix" and usually named X.

print("\n--- Real-world ML Connection ---")

# Imagine 4 houses, each described by [size_m2, num_rooms, age_years]
X = np.array([
    [85,  3, 10],
    [120, 4,  5],
    [60,  2, 20],
    [200, 5,  2],
])

# True prices (what we want to predict)
y = np.array([300000, 450000, 200000, 750000])

print("House feature matrix X (4 houses, 3 features):")
print(X)
print("\nTarget prices y:", y)
print("\nX.shape:", X.shape, "  -> 4 samples, 3 features")
print("\nFirst house features:", X[0], "  -> 85m2, 3 rooms, 10 years old")
print("First house price:", y[0])

# ------------------------------------------------------------------
# Vector length (magnitude / norm)
# ------------------------------------------------------------------
# The length of a vector is computed with np.linalg.norm()
# For a vector [3, 4], length = sqrt(3^2 + 4^2) = sqrt(25) = 5

v = np.array([3, 4])
length = np.linalg.norm(v)
print("\n--- Vector Length (Norm) ---")
print("Vector:", v)
print("Length (sqrt(3^2 + 4^2)):", length)
print("Manual calculation:", np.sqrt(3**2 + 4**2))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key ideas from this file:
  - Vector  : a list of numbers (one data sample, or one feature set)
  - Matrix  : a 2D table of numbers (a whole dataset)
  - np.dot() or @  : multiply vectors/matrices
  - .T             : transpose (flip rows and columns)
  - Dot product    : measures similarity between two vectors
  - In ML: X has shape (num_samples, num_features)
""")

# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test vector addition
assert np.array_equal(v1 + v2, np.array([5, 7, 9])), "Vector addition failed"

# Test scalar multiplication
assert np.array_equal(3 * v3, np.array([6, 12, 18])), "Scalar multiplication failed"

# Test dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
assert np.dot(a, b) == 32, "Dot product np.dot failed"
assert (a @ b) == 32, "Dot product @ operator failed"
assert dot_manual == 32, "Manual dot product failed"

# Test matrix multiplication shape: (3,2) @ (2,3) = (3,3)
assert C.shape == (3, 3), "Matrix multiply shape should be (3,3)"
assert C[0, 0] == 27, "Top-left element of A@B should be 27"

# Test transpose: M is (2,3), M.T should be (3,2)
assert M.shape == (2, 3), "M shape should be (2,3)"
assert M.T.shape == (3, 2), "M.T shape should be (3,2)"

# Test vector norm: length of [3,4] = 5.0
assert abs(np.linalg.norm(v) - 5.0) < 1e-9, "Vector norm of [3,4] should be 5.0"

# Test cosine similarity: unit vectors have norm 1
assert abs(np.linalg.norm(unit_a) - 1.0) < 1e-9, "unit_a should have norm 1"
assert abs(np.linalg.norm(unit_b) - 1.0) < 1e-9, "unit_b should have norm 1"
# dot product of unit vectors equals cos(theta), which must be in [-1, 1]
assert -1.0 <= np.dot(unit_a, unit_b) <= 1.0, "Cosine similarity out of range"

# Test that dot product with perpendicular vector is 0
assert np.dot(same_direction, perpendicular) == 0, "Perpendicular dot product should be 0"

# Test dataset matrix shape: 4 houses, 3 features
assert X.shape == (4, 3), "House matrix should be shape (4,3)"
assert len(y) == 4, "Target prices array should have 4 elements"

print("\nAll tests passed!")

# =============================================================================
# EXERCISES (try these yourself!)
# =============================================================================
# 1. Create a vector representing yourself: [your_age, height_cm, years_studying]
#    Multiply it by 2. What does that mean conceptually?
#
# 2. Create two vectors of length 3. Compute their dot product manually
#    (with arithmetic), then verify with np.dot(). Do they match?
#
# 3. Create a matrix of shape (4, 3) representing 4 students and
#    3 test scores each. Find the average score per student (hint: np.mean
#    with axis=1).
#
# 4. Create a (2, 3) matrix A and a (3, 2) matrix B. Compute A @ B.
#    What is the shape of the result? Now try B @ A. What shape is that?
#
# 5. Transpose the students matrix from this file. What do the rows
#    represent now? What do the columns represent?
#
# 6. FIRST PRINCIPLES EXERCISE: Derive that for unit vectors (length=1),
#    the dot product equals the cosine of the angle between them.
#    Start from: a . b = |a||b|cos(theta). If |a| = |b| = 1, what
#    simplification occurs? Verify numerically with two unit vectors
#    of your choice.
# =============================================================================
