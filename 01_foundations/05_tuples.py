# HOW TO RUN:
#   uv run python 01_foundations/05_tuples.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- TUPLES ---
# A tuple is like a list, but it CANNOT be changed after creation.
# It is "immutable" (just like strings).
#
# Why tuples matter:
# - They signal "this data should not change"
# - They're used for returning multiple values from functions
# - They're used as dictionary keys (lists can't do this!)
# - In ML, you'll see tuples for shapes of data: (rows, columns)


# === CREATING TUPLES ===
# Use parentheses () instead of square brackets []

point = (3, 5)
print("Point:", point)
# Output: Point: (3, 5)

colors = ("red", "green", "blue")
print("Colors:", colors)
# Output: Colors: ('red', 'green', 'blue')

# A tuple with just one item needs a trailing comma (this is a gotcha!)
single = (42,)     # This IS a tuple
not_tuple = (42)   # This is just the number 42 in parentheses!
print(type(single))       # Output: <class 'tuple'>
print(type(not_tuple))    # Output: <class 'int'>

# You can also create a tuple without parentheses (but it's less clear)
another = 1, 2, 3
print("Another tuple:", another)
# Output: Another tuple: (1, 2, 3)


# === INDEXING AND SLICING (same as lists) ===

coordinates = (10, 20, 30, 40, 50)

print(coordinates[0])     # Output: 10
print(coordinates[-1])    # Output: 50
print(coordinates[1:4])   # Output: (20, 30, 40)
print(len(coordinates))   # Output: 5


# === IMMUTABILITY: Tuples CANNOT be changed ===

point = (3, 5)
# point[0] = 10    # This would CRASH! TypeError!
# point.append(7)  # This would CRASH too! No append method.

# If you need to change something, create a new tuple
new_point = (10, point[1])
print("New point:", new_point)
# Output: New point: (10, 5)


# === PACKING AND UNPACKING ===
# This is one of the most useful features of tuples!

# Packing: putting values into a tuple
person = ("Astha", 25, "Delhi")

# Unpacking: pulling values out into separate variables
name, age, city = person
print(f"{name} is {age} years old, lives in {city}")
# Output: Astha is 25 years old, lives in Delhi

# This is why functions can return multiple values!
def get_min_max(numbers):
    return min(numbers), max(numbers)   # Returns a tuple

data = [15, 8, 42, 3, 27]
low, high = get_min_max(data)     # Unpacking the returned tuple
print(f"Min: {low}, Max: {high}")
# Output: Min: 3, Max: 42

# Swap two variables (Python trick using tuple unpacking!)
a = 10
b = 20
a, b = b, a    # Swap!
print(f"a = {a}, b = {b}")
# Output: a = 20, b = 10


# === TUPLES IN LOOPS ===

# A list of tuples -- very common pattern for structured data
students = [
    ("Astha", 85),
    ("Raj", 92),
    ("Priya", 78),
]

print("Student scores:")
for name, score in students:   # Unpacking each tuple in the loop
    print(f"  {name}: {score}")
# Output:
# Student scores:
#   Astha: 85
#   Raj: 92
#   Priya: 78

# enumerate() actually returns tuples!
fruits = ["apple", "banana", "cherry"]
for pair in enumerate(fruits):
    print(pair)   # Each pair is a tuple
# Output:
# (0, 'apple')
# (1, 'banana')
# (2, 'cherry')


# === TUPLE METHODS ===
# Tuples have only 2 methods (because they can't be changed)

numbers = (1, 3, 5, 3, 7, 3)

print("Count of 3:", numbers.count(3))   # Output: Count of 3: 3
print("Index of 5:", numbers.index(5))   # Output: Index of 5: 2


# === WHEN TO USE TUPLES VS LISTS ===

# Use a LIST when:
# - You need to add, remove, or change items
# - The collection will grow or shrink
# - Example: a shopping list, a list of user inputs

# Use a TUPLE when:
# - The data should NOT change
# - You want to protect the data from accidental changes
# - You're returning multiple values from a function
# - You need to use it as a dictionary key (more on this later)
# - Example: coordinates (x, y), a date (year, month, day)

# In ML, you'll see tuples used for:
# - Data shapes: (1000, 28, 28) means 1000 images of 28x28 pixels
# - Function return values: (loss, accuracy)
# - Configuration that shouldn't change: (learning_rate, batch_size)

# Quick comparison:
my_list = [1, 2, 3]
my_tuple = (1, 2, 3)

my_list[0] = 99     # Works! Lists are mutable.
# my_tuple[0] = 99  # CRASH! Tuples are immutable.

my_list.append(4)   # Works!
# my_tuple.append(4)  # CRASH! No such method.

print("List:", my_list)    # Output: List: [99, 2, 3, 4]
print("Tuple:", my_tuple)  # Output: Tuple: (1, 2, 3)


# === CONVERTING BETWEEN LISTS AND TUPLES ===

my_list = [1, 2, 3]
my_tuple = tuple(my_list)    # List to tuple
print("As tuple:", my_tuple)
# Output: As tuple: (1, 2, 3)

my_tuple = (4, 5, 6)
my_list = list(my_tuple)     # Tuple to list
print("As list:", my_list)
# Output: As list: [4, 5, 6]


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test: basic tuple creation
assert point == (3, 5), "point should be (3, 5)"
assert colors == ("red", "green", "blue"), "colors tuple is wrong"
assert type(single) == tuple, "single should be a tuple"
assert type(not_tuple) == int, "not_tuple should be an int, not a tuple"
assert another == (1, 2, 3), "another should be (1, 2, 3)"

# Test: indexing and slicing
assert coordinates[0] == 10, "coordinates[0] should be 10"
assert coordinates[-1] == 50, "coordinates[-1] should be 50"
assert coordinates[1:4] == (20, 30, 40), "coordinates[1:4] should be (20, 30, 40)"
assert len(coordinates) == 5, "coordinates should have 5 items"

# Test: immutability — new_point
assert new_point == (10, 5), "new_point should be (10, 5)"

# Test: unpacking (note: name is overwritten by the for loop to "Priya", the last student)
assert age == 25, "unpacked age should be 25"
assert city == "Delhi", "unpacked city should be 'Delhi'"

# Test: get_min_max function
assert get_min_max([15, 8, 42, 3, 27]) == (3, 42), "get_min_max is wrong"
assert low == 3 and high == 42, "low and high from unpacking are wrong"

# Test: swap trick
assert a == 20 and b == 10, "swap trick: a should be 20, b should be 10"

# Test: tuple methods (numbers = (1, 3, 5, 3, 7, 3) at this point)
assert numbers.count(3) == 3, "count of 3 in (1,3,5,3,7,3) should be 3"
assert numbers.index(5) == 2, "index of 5 in (1,3,5,3,7,3) should be 2"

# Test: conversion between list and tuple
# At this point: my_tuple = (4, 5, 6), my_list = [4, 5, 6]
assert my_tuple == (4, 5, 6), "my_tuple should be (4, 5, 6) after last reassignment"
assert my_list == [4, 5, 6], "my_list should be [4, 5, 6] after list(my_tuple)"
assert tuple([1, 2, 3]) == (1, 2, 3), "tuple([1,2,3]) should be (1,2,3)"
assert list((4, 5, 6)) == [4, 5, 6], "list((4,5,6)) should be [4,5,6]"

print("\nAll tests passed!")

# === EXERCISES ===

# 1. Create a tuple called "rgb" with three values: 255, 128, 0 (an orange color).
#    Unpack it into three variables: red, green, blue.
#    Print each one.
#    Expected output:
#    Red: 255
#    Green: 128
#    Blue: 0

rgb = (255, 128, 0)
red, green, blue = rgb
print(f"Red: {red}")
print(f"Green: {green}")
print(f"Blue: {blue}")



# 2. Given the list of (name, score) tuples below, loop through them
#    and print only the names of students who scored above 80.
#    Expected output:
#    Bob
#    Diana
#    Hint: unpack each tuple in the for loop

results = [("Alice", 72), ("Bob", 88), ("Charlie", 65), ("Diana", 91)]

for name, score in results:
    if score > 80:
        print(name)


# 3. Write a function called "divide" that takes two numbers and returns
#    BOTH the quotient and the remainder as a tuple.
#    Test: divide(17, 5) should return (3, 2) because 17/5 = 3 remainder 2
#    Hint: use // for integer division and % for remainder

def divide(num1, num2):
    return num1 // num2, num1 % num2

quotient, remainder = divide(17, 5)
print(f"17/5 = {quotient} remainder {remainder}")



# 4. Use the swap trick to swap the values of x and y without using
#    a temporary variable.
#    Expected output:
#    Before: x=100, y=200
#    After: x=200, y=100

x = 100
y = 200
print(f"Before: x={x}, y={y}")

x, y = y, x
print(f"After: x={x}, y={y}")


# 5. (Challenge) Given a list of (city, temperature) tuples,
#    find the city with the highest temperature.
#    Expected output: Chennai with 38 degrees
#    Hint: loop through, keep track of the best one

weather = [("Delhi", 35), ("Mumbai", 32), ("Chennai", 38), ("Kolkata", 30)]

def highest_temperature(weather):
    max_temp = 0
    max_temp_city = ""
    for city, temperature in weather:
        if temperature > max_temp:
            max_temp = temperature
            max_temp_city = city
    return max_temp_city, max_temp

            

city, temperature = highest_temperature(weather)
print(f"{city} with {temperature} degrees")

rating_list = [("Delhi", 4.5), ("Lucknow", 3.76), ("Mumbai", 7.8), ("Gwalior", 9.9), ("Raipur", 0.1)]

def mean(all_ratings: list[tuple[str, float]]) -> list[str]:
    mean_rating = 0
    total = 0
    highest_rating_city = []
    for city, rating in all_ratings:
        total += rating
    mean_rating = total / len(all_ratings)
    print(mean_rating)
    for city, rating in all_ratings:
        if rating > mean_rating:
            highest_rating_city.append(city)
    return highest_rating_city

city = mean(rating_list)
print(city)
        




