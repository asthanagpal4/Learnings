# HOW TO RUN:
#   uv run python 01_foundations/03_functions.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- FUNCTIONS ---
# A function is a reusable block of code that does a specific job.
# Instead of writing the same code over and over, you put it in a function
# and call it whenever you need it.
#
# Why this matters for ML:
# ML code is built from functions -- training functions, loss functions,
# data processing functions, evaluation functions. Learning to write
# clean functions is a fundamental skill.


# === DEFINING A FUNCTION ===
# Use the "def" keyword, give it a name, add parentheses, and a colon.
# The code inside the function must be indented.

def say_hello():
    print("Hello, Astha!")

# To use (or "call") the function, write its name with parentheses:
say_hello()
# Output: Hello, Astha!

# You can call it as many times as you want:
say_hello()
say_hello()
# Output:
# Hello, Astha!
# Hello, Astha!


# === PARAMETERS (Inputs to a function) ===
# You can pass information into a function using parameters.
# Parameters go inside the parentheses.

def greet(name):
    print(f"Hello, {name}!")

greet("Astha")     # Output: Hello, Astha!
greet("Python")    # Output: Hello, Python!

# Multiple parameters -- separate them with commas
def add(a, b):
    print(f"{a} + {b} = {a + b}")

add(3, 5)      # Output: 3 + 5 = 8
add(10, 20)    # Output: 10 + 20 = 30


# === RETURN VALUES (Getting results back) ===
# print() shows something on screen, but return sends a value back
# so you can store it and use it later. This is a crucial difference!

def multiply(a, b):
    return a * b

result = multiply(4, 5)
print("4 times 5 is:", result)
# Output: 4 times 5 is: 20

# You can use the returned value directly in expressions
total = multiply(3, 7) + multiply(2, 8)
print("Total:", total)
# Output: Total: 37    (because 21 + 16 = 37)

# A function without return gives back "None"
def just_prints(x):
    print(x)

result = just_prints("hi")   # Output: hi
print("Return value:", result)
# Output: Return value: None


# === DEFAULT PARAMETERS ===
# You can give parameters a default value.
# If the caller doesn't provide that argument, the default is used.

def power(base, exponent=2):
    return base ** exponent

print(power(5))       # Output: 25   (5 squared, used default exponent=2)
print(power(5, 3))    # Output: 125  (5 cubed, we provided exponent=3)
print(power(2, 10))   # Output: 1024 (2 to the power of 10)


# === FUNCTIONS THAT WORK WITH LISTS ===
# Functions are great for processing lists

def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

test_scores = [85, 92, 78, 95, 88]
avg = calculate_average(test_scores)
print(f"Average score: {avg}")
# Output: Average score: 87.6

# Another example: filter a list
def get_passing_scores(scores, passing_mark=60):
    passing = []
    for score in scores:
        if score >= passing_mark:
            passing.append(score)
    return passing

all_scores = [45, 82, 55, 91, 73, 38, 67]
passed = get_passing_scores(all_scores)
print("Passed (60+):", passed)
# Output: Passed (60+): [82, 91, 73, 67]

passed_high = get_passing_scores(all_scores, 75)
print("Passed (75+):", passed_high)
# Output: Passed (75+): [82, 91]


# === RETURNING MULTIPLE VALUES ===
# Python lets you return more than one value (using a tuple, covered later)

def get_stats(numbers):
    smallest = min(numbers)
    largest = max(numbers)
    average = sum(numbers) / len(numbers)
    return smallest, largest, average

low, high, avg = get_stats([10, 20, 30, 40, 50])
print(f"Low: {low}, High: {high}, Average: {avg}")
# Output: Low: 10, High: 50, Average: 30.0


# === SCOPE: Where variables live ===
# Variables created INSIDE a function only exist inside that function.
# Variables created OUTSIDE a function are available everywhere.

message = "I am outside"   # This is a "global" variable

def show_scope():
    local_var = "I am inside"   # This is a "local" variable
    print(local_var)
    print(message)   # Can see the global variable

show_scope()
# Output:
# I am inside
# I am outside

# If you try to use local_var outside the function, you'd get an error:
# print(local_var)   # This would crash! local_var doesn't exist here.

# IMPORTANT: it's best practice to pass data into functions as parameters
# rather than relying on global variables. This makes your code cleaner.


# === FUNCTIONS CALLING OTHER FUNCTIONS ===
# Functions can call other functions -- this is how you build bigger programs

def is_even(number):
    return number % 2 == 0

def count_evens(numbers):
    count = 0
    for num in numbers:
        if is_even(num):   # Calling our other function!
            count += 1
    return count

my_list = [1, 2, 3, 4, 5, 6, 7, 8]
print(f"Even numbers in list: {count_evens(my_list)}")
# Output: Even numbers in list: 4


# === A PRACTICAL EXAMPLE ===
# Let's write a function that could be useful for data processing

def clean_text(text):
    """Remove extra spaces and convert to lowercase."""
    cleaned = text.strip()       # Remove spaces from start and end
    cleaned = cleaned.lower()    # Convert to lowercase
    return cleaned

raw_data = "   Hello World   "
clean_data = clean_text(raw_data)
print(f"Before: '{raw_data}'")
print(f"After:  '{clean_data}'")
# Output:
# Before: '   Hello World   '
# After:  'hello world'


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test: multiply function
assert multiply(4, 5) == 20, "multiply(4, 5) should be 20"
assert multiply(0, 100) == 0, "multiply(0, 100) should be 0"
assert multiply(3, 7) == 21, "multiply(3, 7) should be 21"

# Test: power function with default parameter
assert power(5) == 25, "power(5) should be 25 (default exponent=2)"
assert power(5, 3) == 125, "power(5, 3) should be 125"
assert power(2, 10) == 1024, "power(2, 10) should be 1024"

# Test: calculate_average function
assert calculate_average([85, 92, 78, 95, 88]) == 87.6, "average of test_scores is wrong"
assert calculate_average([10, 20, 30]) == 20.0, "average of [10, 20, 30] should be 20.0"

# Test: get_passing_scores function
assert get_passing_scores([45, 82, 55, 91, 73, 38, 67]) == [82, 91, 73, 67], "passing scores (60+) are wrong"
assert get_passing_scores([45, 82, 55, 91, 73, 38, 67], 75) == [82, 91], "passing scores (75+) are wrong"
assert get_passing_scores([10, 20, 30], 50) == [], "no scores pass 50 from [10, 20, 30]"

# Test: get_stats function
assert get_stats([10, 20, 30, 40, 50]) == (10, 50, 30.0), "get_stats is wrong"

# Test: is_even and count_evens functions
assert is_even(4) == True, "4 should be even"
assert is_even(7) == False, "7 should not be even"
assert count_evens([1, 2, 3, 4, 5, 6, 7, 8]) == 4, "count_evens should return 4"
assert count_evens([1, 3, 5]) == 0, "no evens in [1, 3, 5]"

# Test: clean_text function
assert clean_text("   Hello World   ") == "hello world", "clean_text should strip and lowercase"
assert clean_text("  PYTHON  ") == "python", "clean_text should handle all caps"

print("\nAll tests passed!")

# === EXERCISES ===

# 1. Write a function called "double" that takes a number and returns
#    that number multiplied by 2. Test it with a few values.
#    Expected:
#    double(5) should give 10
#    double(0) should give 0
#    Hint: def double(x): ...

def double(x):
    return x*2

print(double(5))
print(double(0))



# 2. Write a function called "is_positive" that takes a number and returns
#    True if it's greater than 0, and False otherwise.
#    Expected:
#    is_positive(5) should give True
#    is_positive(-3) should give False
#    is_positive(0) should give False
#    Hint: return number > 0

def is_positive(x):
    return x > 0

print(is_positive(5))
print(is_positive(-3))
print(is_positive(0))


# 3. Write a function called "find_longest" that takes a list of strings
#    and returns the longest string.
#    Expected: find_longest(["cat", "elephant", "dog"]) should give "elephant"
#    Hint: loop through the list, keep track of the longest one so far

def find_longest(strings):
    longest = strings[0]
    for s in strings:
        if len(s) > len(longest):
            longest = s 
    return longest

animals =  ["cat", "elephant", "dog"]
print(find_longest(animals))


# 4. Write a function called "count_word" that takes two arguments:
#    a list of words and a target word, and returns how many times
#    the target word appears in the list.
#    Expected: count_word(["hi", "bye", "hi", "hello"], "hi") should give 2
#    Hint: use a counter variable, loop, and if

def count_word(words, target):
    repeat = 0
    for word in words:
        if word == target:
            repeat = repeat + 1
    return repeat

words = ["hi", "bye", "hi", "hello"]
print(count_word(words, "hi"))




# 5. (Challenge) Write a function called "apply_to_all" that takes a list
#    of numbers and a function, and returns a new list with that function
#    applied to every item. Test it with the double function from exercise 1.
#    Expected: apply_to_all([1, 2, 3], double) should give [2, 4, 6]
#    Hint: yes, you can pass a function as an argument to another function!
#    This is a key idea in Python and ML.

def apply_to_all(numbers, func):
    result = []
    for num in numbers:
        result.append(func(num))
    return result

numbers = [1, 2, 3]
print(apply_to_all(numbers, double))