# HOW TO RUN:
#   uv run python 04_algorithms/05_numpy_intro.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- NUMPY INTRODUCTION ---
# NumPy (Numerical Python) is THE most important library for ML in Python.
# It gives you fast arrays that can do math on millions of numbers at once.
#
# Why does this matter for ML?
#   - ML models are all about numbers: weights, inputs, outputs, gradients
#   - Training an LLM means doing math on BILLIONS of numbers
#   - NumPy is the foundation that PyTorch, TensorFlow, and scikit-learn are built on
#
# Think of NumPy arrays like supercharged lists, designed for math.

# Try to import numpy — if not installed, show a helpful message
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("=" * 60)
    print("NumPy is not installed!")
    print("To install it, run this command in your terminal:")
    print()
    print("    pip install numpy")
    print()
    print("or if you use pip3:")
    print()
    print("    pip3 install numpy")
    print()
    print("After installing, run this file again.")
    print("=" * 60)

import time

if NUMPY_AVAILABLE:
    print("=" * 60)
    print("SECTION 4.5 — NUMPY INTRODUCTION")
    print("=" * 60)


    # === CONCEPT 1: ARRAYS vs LISTS ===
    # A Python list can hold anything: [1, "hello", True, 3.14]
    # A NumPy array holds ONE type of data (usually numbers).
    # This restriction is what makes it so fast!

    print("\n=== ARRAYS vs LISTS ===\n")

    # Creating arrays
    my_list = [1, 2, 3, 4, 5]
    my_array = np.array([1, 2, 3, 4, 5])

    print(f"Python list:  {my_list}     type: {type(my_list)}")
    print(f"NumPy array:  {my_array}  type: {type(my_array)}")
    # Output:
    # Python list:  [1, 2, 3, 4, 5]     type: <class 'list'>
    # NumPy array:  [1 2 3 4 5]  type: <class 'numpy.ndarray'>

    # Notice: NumPy arrays print WITHOUT commas

    # Arrays have a data type (dtype)
    print(f"\nArray dtype:  {my_array.dtype}")
    # Output: Array dtype:  int64

    float_array = np.array([1.5, 2.7, 3.2])
    print(f"Float array:  {float_array}, dtype: {float_array.dtype}")
    # Output: Float array:  [1.5 2.7 3.2], dtype: float64

    # Useful ways to create arrays
    zeros = np.zeros(5)           # 5 zeros
    ones = np.ones(5)             # 5 ones
    sequence = np.arange(0, 10, 2)  # Like range(): 0, 2, 4, 6, 8
    spaced = np.linspace(0, 1, 5)   # 5 evenly spaced values from 0 to 1

    print(f"\nnp.zeros(5):          {zeros}")
    print(f"np.ones(5):           {ones}")
    print(f"np.arange(0, 10, 2):  {sequence}")
    print(f"np.linspace(0, 1, 5): {spaced}")
    # Output:
    # np.zeros(5):          [0. 0. 0. 0. 0.]
    # np.ones(5):           [1. 1. 1. 1. 1.]
    # np.arange(0, 10, 2):  [0 2 4 6 8]
    # np.linspace(0, 1, 5): [0.   0.25 0.5  0.75 1.  ]


    # === CONCEPT 2: VECTORIZED OPERATIONS ===
    # The BIGGEST advantage of NumPy: do math on ALL elements at once.
    # With a list, you'd need a loop. With NumPy, just write the math directly.
    # This is called "vectorized" operations — and it's MUCH faster.

    print("\n=== VECTORIZED OPERATIONS ===\n")

    a = np.array([1, 2, 3, 4, 5])
    print(f"a = {a}")

    # Math with a single number (applied to every element)
    print(f"a + 10  = {a + 10}")      # Add 10 to each
    print(f"a * 3   = {a * 3}")       # Multiply each by 3
    print(f"a ** 2  = {a ** 2}")      # Square each
    print(f"a / 2   = {a / 2}")       # Divide each by 2
    # Output:
    # a + 10  = [11 12 13 14 15]
    # a * 3   = [ 3  6  9 12 15]
    # a ** 2  = [ 1  4  9 16 25]
    # a / 2   = [0.5 1.  1.5 2.  2.5]

    # Math between two arrays (element by element)
    b = np.array([10, 20, 30, 40, 50])
    print(f"\na = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    # Output:
    # a + b = [11 22 33 44 55]
    # a * b = [ 10  40  90 160 250]

    # Compare with doing this with regular lists:
    print("\nWith a regular list, you'd need a loop or comprehension:")
    list_a = [1, 2, 3, 4, 5]
    list_b = [10, 20, 30, 40, 50]
    list_sum = [x + y for x, y in zip(list_a, list_b)]
    print(f"  List comprehension: {list_sum}")
    print(f"  NumPy:              {a + b}")
    print("  Same result, but NumPy is simpler and MUCH faster on big data!")


    # === CONCEPT 3: SPEED COMPARISON ===
    # Let's see how much faster NumPy is compared to regular Python.

    print("\n=== SPEED: NUMPY vs PYTHON LISTS ===\n")

    size = 1000000  # One million numbers!

    # Create data
    py_list = list(range(size))
    np_array = np.arange(size)

    # Task: multiply every number by 2

    # Python list (using a loop)
    start = time.time()
    py_result = [x * 2 for x in py_list]
    py_time = time.time() - start

    # NumPy array (vectorized)
    start = time.time()
    np_result = np_array * 2
    np_time = time.time() - start

    print(f"  Multiply {size:,} numbers by 2:")
    print(f"    Python list comprehension: {py_time:.4f} seconds")
    print(f"    NumPy vectorized:          {np_time:.4f} seconds")
    print(f"    NumPy is ~{py_time / max(np_time, 0.000001):.0f}x faster!")

    # Task: sum all numbers
    start = time.time()
    py_sum = sum(py_list)
    py_time = time.time() - start

    start = time.time()
    np_sum = np.sum(np_array)
    np_time = time.time() - start

    print(f"\n  Sum {size:,} numbers:")
    print(f"    Python sum():  {py_time:.4f} seconds")
    print(f"    NumPy sum():   {np_time:.4f} seconds")
    print(f"    NumPy is ~{py_time / max(np_time, 0.000001):.0f}x faster!")


    # === CONCEPT 4: ARRAY SHAPE AND DIMENSIONS ===
    # In ML, data often has multiple dimensions:
    #   - 1D array: a single row of numbers (like a feature vector)
    #   - 2D array: a table of numbers (like a spreadsheet / dataset)
    #   - 3D+ array: used for images, video, etc.

    print("\n=== ARRAY SHAPE AND DIMENSIONS ===\n")

    # 1D array
    one_d = np.array([1, 2, 3, 4, 5])
    print(f"1D array:  {one_d}")
    print(f"  Shape: {one_d.shape}, Dimensions: {one_d.ndim}")
    # Output: Shape: (5,), Dimensions: 1

    # 2D array (like a table with rows and columns)
    two_d = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print(f"\n2D array:\n{two_d}")
    print(f"  Shape: {two_d.shape}, Dimensions: {two_d.ndim}")
    # Output: Shape: (3, 3), Dimensions: 2
    # Shape (3, 3) means 3 rows and 3 columns

    # In ML, your dataset might look like this:
    # Each row = one data point, each column = one feature
    dataset = np.array([
        [5.1, 3.5, 1.4],   # sample 1: feature1=5.1, feature2=3.5, feature3=1.4
        [4.9, 3.0, 1.4],   # sample 2
        [7.0, 3.2, 4.7],   # sample 3
        [6.3, 3.3, 6.0],   # sample 4
    ])
    print(f"\nML dataset example:\n{dataset}")
    print(f"  Shape: {dataset.shape}")
    print(f"  {dataset.shape[0]} samples, {dataset.shape[1]} features each")
    # Output: Shape: (4, 3) -> 4 samples, 3 features

    # Reshape arrays (very common in ML)
    flat = np.array([1, 2, 3, 4, 5, 6])
    reshaped = flat.reshape(2, 3)  # 2 rows, 3 columns
    print(f"\nFlat:     {flat}")
    print(f"Reshaped (2x3):\n{reshaped}")


    # === CONCEPT 5: INDEXING AND SLICING ===
    # Access parts of arrays — very similar to list slicing.

    print("\n=== INDEXING AND SLICING ===\n")

    arr = np.array([10, 20, 30, 40, 50, 60, 70])

    print(f"Array:      {arr}")
    print(f"arr[0]:     {arr[0]}")       # First element
    print(f"arr[-1]:    {arr[-1]}")      # Last element
    print(f"arr[2:5]:   {arr[2:5]}")     # Elements 2, 3, 4
    print(f"arr[::2]:   {arr[::2]}")     # Every other element
    # Output:
    # arr[0]:     10
    # arr[-1]:    70
    # arr[2:5]:   [30 40 50]
    # arr[::2]:   [10 30 50 70]

    # Boolean indexing — super useful in ML for filtering data!
    scores = np.array([85, 42, 91, 67, 73, 55, 88])
    print(f"\nScores:           {scores}")
    print(f"Scores > 70:      {scores > 70}")         # True/False for each
    print(f"High scores:      {scores[scores > 70]}")  # Only the high scores!
    # Output:
    # Scores > 70:      [ True False  True False  True False  True]
    # High scores:      [85 91 73 88]

    # 2D indexing
    grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print(f"\nGrid:\n{grid}")
    print(f"Row 0:        {grid[0]}")           # First row
    print(f"Column 1:     {grid[:, 1]}")        # Second column (all rows, column 1)
    print(f"Element [1,2]: {grid[1, 2]}")       # Row 1, Column 2
    # Output:
    # Row 0:        [1 2 3]
    # Column 1:     [2 5 8]
    # Element [1,2]: 6


    # === CONCEPT 6: USEFUL NUMPY FUNCTIONS ===
    # NumPy has tons of built-in math functions — all vectorized and fast.

    print("\n=== USEFUL NUMPY FUNCTIONS ===\n")

    data = np.array([14, 27, 8, 35, 19, 42, 11, 23])
    print(f"Data:     {data}")
    print(f"Mean:     {np.mean(data)}")       # Average
    print(f"Median:   {np.median(data)}")     # Middle value
    print(f"Std Dev:  {np.std(data):.2f}")    # Standard deviation (spread)
    print(f"Min:      {np.min(data)}")        # Smallest
    print(f"Max:      {np.max(data)}")        # Largest
    print(f"Sum:      {np.sum(data)}")        # Total
    print(f"Argmin:   {np.argmin(data)}")     # INDEX of smallest (useful in ML!)
    print(f"Argmax:   {np.argmax(data)}")     # INDEX of largest
    # Output:
    # Mean:     22.375
    # Median:   21.0
    # Std Dev:  10.87
    # Min:      8
    # Max:      42
    # Sum:      179
    # Argmin:   2
    # Argmax:   5

    # Sorting
    print(f"\nSorted:   {np.sort(data)}")
    # Output: Sorted:   [ 8 11 14 19 23 27 35 42]

    # Random numbers (very common in ML for initialization)
    np.random.seed(42)  # Makes random numbers reproducible
    random_floats = np.random.rand(5)       # 5 random numbers between 0 and 1
    random_ints = np.random.randint(1, 100, size=5)  # 5 random ints 1-99
    random_normal = np.random.randn(5)      # 5 numbers from a bell curve

    print(f"\nRandom floats [0,1):  {random_floats}")
    print(f"Random ints [1,100):  {random_ints}")
    print(f"Random normal:        {np.round(random_normal, 2)}")


    # === CONCEPT 7: DOT PRODUCT — THE BUILDING BLOCK OF ML ===
    # The dot product is the most important operation in ML.
    # It's how neural networks process inputs:
    #   output = input dot weights + bias
    # We'll just see the basics here.

    print("\n=== DOT PRODUCT (THE HEART OF ML) ===\n")

    # Dot product of two vectors: multiply matching elements, then sum
    weights = np.array([0.5, -0.3, 0.8])
    inputs = np.array([1.0, 2.0, 3.0])

    # Manual calculation:
    # 0.5*1.0 + (-0.3)*2.0 + 0.8*3.0 = 0.5 - 0.6 + 2.4 = 2.3
    dot_manual = sum(w * x for w, x in zip(weights, inputs))
    dot_numpy = np.dot(weights, inputs)

    print(f"Weights: {weights}")
    print(f"Inputs:  {inputs}")
    print(f"Dot product (manual): {dot_manual}")
    print(f"Dot product (NumPy):  {dot_numpy}")
    # Output: Dot product: 2.3

    print("\nThis is literally what happens in a neural network neuron!")
    print("  output = dot(inputs, weights) + bias")
    print("  In LLMs, this happens billions of times per prediction.")


    # === TEST CASES ===
    # These asserts verify the code above works correctly.
    # If any assert fails, Python will tell you which one!

    # Array creation
    assert np.array_equal(np.zeros(5), np.array([0., 0., 0., 0., 0.]))
    assert np.array_equal(np.ones(5), np.array([1., 1., 1., 1., 1.]))
    assert np.array_equal(np.arange(0, 10, 2), np.array([0, 2, 4, 6, 8]))
    assert np.allclose(np.linspace(0, 1, 5), np.array([0., 0.25, 0.5, 0.75, 1.]))

    # Vectorized operations
    a = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(a + 10, np.array([11, 12, 13, 14, 15]))
    assert np.array_equal(a * 3, np.array([3, 6, 9, 12, 15]))
    assert np.array_equal(a ** 2, np.array([1, 4, 9, 16, 25]))
    assert np.allclose(a / 2, np.array([0.5, 1.0, 1.5, 2.0, 2.5]))

    b = np.array([10, 20, 30, 40, 50])
    assert np.array_equal(a + b, np.array([11, 22, 33, 44, 55]))
    assert np.array_equal(a * b, np.array([10, 40, 90, 160, 250]))

    # Array shape and dimensions
    one_d = np.array([1, 2, 3, 4, 5])
    assert one_d.shape == (5,)
    assert one_d.ndim == 1

    two_d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert two_d.shape == (3, 3)
    assert two_d.ndim == 2

    flat = np.array([1, 2, 3, 4, 5, 6])
    reshaped = flat.reshape(2, 3)
    assert reshaped.shape == (2, 3)

    # Indexing and slicing
    arr = np.array([10, 20, 30, 40, 50, 60, 70])
    assert arr[0] == 10
    assert arr[-1] == 70
    assert np.array_equal(arr[2:5], np.array([30, 40, 50]))
    assert np.array_equal(arr[::2], np.array([10, 30, 50, 70]))

    # Boolean indexing
    scores = np.array([85, 42, 91, 67, 73, 55, 88])
    assert np.array_equal(scores[scores > 70], np.array([85, 91, 73, 88]))

    # 2D indexing
    grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert np.array_equal(grid[0], np.array([1, 2, 3]))
    assert np.array_equal(grid[:, 1], np.array([2, 5, 8]))
    assert grid[1, 2] == 6

    # Statistical functions
    data = np.array([14, 27, 8, 35, 19, 42, 11, 23])
    assert np.allclose(np.mean(data), 22.375)
    assert np.allclose(np.median(data), 21.0)
    assert np.min(data) == 8
    assert np.max(data) == 42
    assert np.sum(data) == 179
    assert np.argmin(data) == 2
    assert np.argmax(data) == 5
    assert np.array_equal(np.sort(data), np.array([8, 11, 14, 19, 23, 27, 35, 42]))

    # Dot product
    weights = np.array([0.5, -0.3, 0.8])
    inputs = np.array([1.0, 2.0, 3.0])
    assert np.allclose(np.dot(weights, inputs), 2.3)

    print("\nAll tests passed!")

    # === EXERCISES ===
    print("\n" + "=" * 60)
    print("EXERCISES")
    print("=" * 60)

    # 1. Create a NumPy array of numbers from 1 to 20. Then:
    #    a) Print all even numbers (hint: use boolean indexing with % 2 == 0)
    #    b) Print the mean and standard deviation
    #    c) Print all numbers greater than 15
    #    Expected: evens = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # Your code here:


    # 2. Create two 3x3 arrays and:
    #    a) Add them together
    #    b) Multiply them element-wise
    #    c) Find the dot product using np.dot()
    #    Hint: np.array([[1,2,3],[4,5,6],[7,8,9]])

    # Your code here:


    # 3. Simulate ML data: Create an array of 100 random "exam scores" between
    #    0 and 100. Then:
    #    a) Print the average score
    #    b) Print how many students scored above 70
    #    c) Print the percentage of students who passed (score >= 50)
    #    Hint: np.random.randint(0, 101, size=100)
    #    Hint: np.sum(scores > 70) counts True values

    # Your code here:


    # 4. Normalize an array (make values go from 0 to 1).
    #    This is a super common ML preprocessing step!
    #    Formula: normalized = (array - min) / (max - min)
    #    Test with: data = np.array([10, 20, 30, 40, 50])
    #    Expected: [0.0, 0.25, 0.5, 0.75, 1.0]

    # Your code here:


    print("\nDone! Try the exercises above by writing code and re-running this file.")

else:
    print("\nPlease install NumPy first, then re-run this file.")
