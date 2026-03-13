# HOW TO RUN:
#   uv run python 02_programming/04_classes_basics.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- CLASSES & OOP BASICS ---
# A class is a blueprint for creating objects (things with data and behavior).
# Think of it like a cookie cutter — the class is the cutter, objects are the cookies.
# In ML, classes are everywhere: models, datasets, trainers are all classes.
# OOP = Object-Oriented Programming — organizing code around objects.


# === CONCEPT 1: YOUR FIRST CLASS ===
# A class groups together related data (attributes) and actions (methods).
# "self" refers to the specific object being used.
# "__init__" is a special method that runs when you create a new object.

print("=" * 50)
print("CONCEPT 1: Your first class")
print("=" * 50)

class Dog:
    """A simple Dog class."""

    def __init__(self, name, breed):
        # These are attributes — data stored in the object
        self.name = name
        self.breed = breed
        self.energy = 100  # default value, not passed in

    def bark(self):
        # This is a method — a function that belongs to the object
        print(f"{self.name} says: Woof!")

    def info(self):
        print(f"{self.name} is a {self.breed} with {self.energy} energy")

# Create objects (also called "instances") from the class
dog1 = Dog("Buddy", "Golden Retriever")
dog2 = Dog("Luna", "Labrador")

dog1.bark()       # Output: Buddy says: Woof!
dog2.bark()       # Output: Luna says: Woof!
dog1.info()       # Output: Buddy is a Golden Retriever with 100 energy
dog2.info()       # Output: Luna is a Labrador with 100 energy

# Each object has its own separate data
print(f"dog1.name = {dog1.name}")   # Output: dog1.name = Buddy
print(f"dog2.name = {dog2.name}")   # Output: dog2.name = Luna


# === CONCEPT 2: METHODS THAT CHANGE DATA ===
# Methods can modify the object's attributes.

print("\n" + "=" * 50)
print("CONCEPT 2: Methods that change data")
print("=" * 50)

class BankAccount:
    """A simple bank account."""

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"  Deposited {amount}. New balance: {self.balance}")
        else:
            print("  Amount must be positive!")

    def withdraw(self, amount):
        if amount > self.balance:
            print(f"  Not enough funds! Balance: {self.balance}")
        elif amount <= 0:
            print("  Amount must be positive!")
        else:
            self.balance -= amount
            print(f"  Withdrew {amount}. New balance: {self.balance}")

    def show_balance(self):
        print(f"  {self.owner}'s balance: {self.balance}")

# Use the class
account = BankAccount("Astha", 1000)
account.show_balance()    # Output:   Astha's balance: 1000
account.deposit(500)      # Output:   Deposited 500. New balance: 1500
account.withdraw(200)     # Output:   Withdrew 200. New balance: 1300
account.withdraw(5000)    # Output:   Not enough funds! Balance: 1300


# === CONCEPT 3: WHY "self" MATTERS ===
# "self" is how the object refers to itself.
# When you call dog1.bark(), Python automatically passes dog1 as "self".
# Without self, the method wouldn't know WHICH dog's name to use.

print("\n" + "=" * 50)
print("CONCEPT 3: Understanding self")
print("=" * 50)

class Student:
    def __init__(self, name):
        self.name = name          # self.name belongs to THIS specific student
        self.scores = []          # each student gets their own empty list

    def add_score(self, score):
        self.scores.append(score)  # adds to THIS student's scores

    def average(self):
        if not self.scores:
            return 0
        return sum(self.scores) / len(self.scores)

# Two separate students with separate data
s1 = Student("Astha")
s2 = Student("Priya")

s1.add_score(85)
s1.add_score(92)

s2.add_score(78)
s2.add_score(88)
s2.add_score(95)

print(f"{s1.name}'s scores: {s1.scores}, average: {s1.average()}")
# Output: Astha's scores: [85, 92], average: 88.5

print(f"{s2.name}'s scores: {s2.scores}, average: {s2.average()}")
# Output: Priya's scores: [78, 88, 95], average: 87.0


# === CONCEPT 4: ATTRIBUTES VS METHODS ===
# Attributes = data (nouns): self.name, self.balance
# Methods = actions (verbs): .deposit(), .withdraw(), .bark()
# Access attributes with dot notation: object.attribute
# Call methods with parentheses: object.method()

print("\n" + "=" * 50)
print("CONCEPT 4: Attributes vs Methods")
print("=" * 50)

class Rectangle:
    def __init__(self, width, height):
        self.width = width        # attribute
        self.height = height      # attribute

    def area(self):               # method
        return self.width * self.height

    def perimeter(self):          # method
        return 2 * (self.width + self.height)

    def is_square(self):          # method
        return self.width == self.height

r1 = Rectangle(5, 3)
r2 = Rectangle(4, 4)

# Accessing attributes (no parentheses)
print(f"r1 dimensions: {r1.width} x {r1.height}")
# Output: r1 dimensions: 5 x 3

# Calling methods (with parentheses)
print(f"r1 area: {r1.area()}")
# Output: r1 area: 15
print(f"r1 perimeter: {r1.perimeter()}")
# Output: r1 perimeter: 16
print(f"r1 is square? {r1.is_square()}")
# Output: r1 is square? False
print(f"r2 is square? {r2.is_square()}")
# Output: r2 is square? True


# === CONCEPT 5: DEFAULT VALUES AND OPTIONAL PARAMETERS ===
# You can give __init__ parameters default values.

print("\n" + "=" * 50)
print("CONCEPT 5: Default values")
print("=" * 50)

class DataPoint:
    """Represents a single data point — like one row in a dataset."""

    def __init__(self, x, y, label="unknown"):
        self.x = x
        self.y = y
        self.label = label

    def display(self):
        print(f"  Point({self.x}, {self.y}) — label: {self.label}")

# With label
p1 = DataPoint(3.5, 7.2, "positive")
p1.display()
# Output:   Point(3.5, 7.2) — label: positive

# Without label (uses default "unknown")
p2 = DataPoint(1.0, 2.0)
p2.display()
# Output:   Point(1.0, 2.0) — label: unknown


# === CONCEPT 6: METHODS THAT RETURN VALUES ===
# Not all methods just print — many return values you can use later.

print("\n" + "=" * 50)
print("CONCEPT 6: Methods that return values")
print("=" * 50)

class ScoreTracker:
    """Tracks scores and computes statistics."""

    def __init__(self):
        self.scores = []

    def add(self, score):
        self.scores.append(score)

    def count(self):
        return len(self.scores)

    def total(self):
        return sum(self.scores)

    def average(self):
        if self.count() == 0:
            return 0
        return self.total() / self.count()

    def highest(self):
        if not self.scores:
            return None
        return max(self.scores)

    def summary(self):
        return {
            "count": self.count(),
            "total": self.total(),
            "average": round(self.average(), 2),
            "highest": self.highest()
        }

tracker = ScoreTracker()
tracker.add(85)
tracker.add(92)
tracker.add(78)
tracker.add(95)
tracker.add(88)

print(f"Count: {tracker.count()}")       # Output: Count: 5
print(f"Average: {tracker.average()}")   # Output: Average: 87.6
print(f"Summary: {tracker.summary()}")
# Output: Summary: {'count': 5, 'total': 438, 'average': 87.6, 'highest': 95}


# === CONCEPT 7: OBJECTS IN LISTS AND DICTS ===
# Objects can be stored in any data structure — just like numbers or strings.

print("\n" + "=" * 50)
print("CONCEPT 7: Objects in collections")
print("=" * 50)

class Item:
    def __init__(self, name, price):
        self.name = name
        self.price = price

# A list of Item objects (like a shopping cart)
cart = [
    Item("Notebook", 120),
    Item("Pen", 15),
    Item("Eraser", 5),
]

total = 0
for item in cart:
    print(f"  {item.name}: Rs {item.price}")
    total += item.price

print(f"  Total: Rs {total}")
# Output:
#   Notebook: Rs 120
#   Pen: Rs 15
#   Eraser: Rs 5
#   Total: Rs 140


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test 1: Dog class — attributes and default energy
_d = Dog("Rex", "Poodle")
assert _d.name == "Rex", "Dog name should be 'Rex'"
assert _d.breed == "Poodle", "Dog breed should be 'Poodle'"
assert _d.energy == 100, "Default energy should be 100"

# Test 2: BankAccount — deposit and withdraw change balance correctly
_acc = BankAccount("Tester", 500)
assert _acc.balance == 500, "Initial balance should be 500"
_acc.deposit(200)
assert _acc.balance == 700, "After deposit 200, balance should be 700"
_acc.withdraw(100)
assert _acc.balance == 600, "After withdraw 100, balance should be 600"
_acc.withdraw(9999)
assert _acc.balance == 600, "Over-withdrawal should not change balance"

# Test 3: Student class — separate score lists per instance
_s1 = Student("Alice")
_s2 = Student("Bob")
_s1.add_score(90)
_s1.add_score(80)
_s2.add_score(70)
assert _s1.scores == [90, 80], "Alice's scores wrong"
assert _s2.scores == [70], "Bob's scores wrong — lists must be independent"
assert _s1.average() == 85.0, "Alice's average should be 85.0"
assert _s2.average() == 70.0, "Bob's average should be 70.0"

# Empty scores average should be 0
_s_empty = Student("Empty")
assert _s_empty.average() == 0, "Empty scores average should be 0"

# Test 4: Rectangle — area, perimeter, is_square
_r = Rectangle(5, 3)
assert _r.area() == 15, "Area of 5x3 should be 15"
assert _r.perimeter() == 16, "Perimeter of 5x3 should be 16"
assert _r.is_square() == False, "5x3 rectangle is not a square"
_sq = Rectangle(4, 4)
assert _sq.is_square() == True, "4x4 rectangle is a square"

# Test 5: DataPoint — default label
_p = DataPoint(1.0, 2.0)
assert _p.label == "unknown", "Default label should be 'unknown'"
_p2 = DataPoint(0, 0, "negative")
assert _p2.label == "negative", "Custom label should be 'negative'"

# Test 6: ScoreTracker — count, total, average, highest, summary
_t = ScoreTracker()
assert _t.count() == 0, "New tracker should have 0 scores"
assert _t.average() == 0, "Empty tracker average should be 0"
assert _t.highest() is None, "Empty tracker highest should be None"
_t.add(80)
_t.add(100)
assert _t.count() == 2, "After 2 adds, count should be 2"
assert _t.total() == 180, "Total should be 180"
assert _t.average() == 90.0, "Average should be 90.0"
assert _t.highest() == 100, "Highest should be 100"
_summary = _t.summary()
assert _summary["count"] == 2, "Summary count wrong"
assert _summary["highest"] == 100, "Summary highest wrong"

# Test 7: Item objects in a list
_cart = [Item("Book", 50), Item("Pen", 10)]
_total = sum(item.price for item in _cart)
assert _total == 60, "Cart total should be 60"
assert _cart[0].name == "Book", "First item name should be 'Book'"

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Create a class called "Book" with:
#    - Attributes: title, author, pages
#    - A method called "description()" that prints:
#      "'{title}' by {author}, {pages} pages"
#    Create 2 Book objects and call description() on each.
#    Hint: very similar to the Dog class above

# YOUR CODE HERE:


# 2. Create a class called "Counter" with:
#    - An attribute "count" that starts at 0
#    - Methods: increment() adds 1, decrement() subtracts 1, reset() sets to 0
#    - A method value() that returns the current count
#    Test: increment 5 times, decrement 2 times, print value (should be 3)
#    Hint: modify self.count in each method

# YOUR CODE HERE:


# 3. Create a class called "TemperatureConverter" with:
#    - __init__ takes a celsius value
#    - A method to_fahrenheit() that returns celsius * 9/5 + 32
#    - A method to_kelvin() that returns celsius + 273.15
#    - A method summary() that prints all three values
#    Test with 100 degrees Celsius.
#    Expected: 100C = 212.0F = 373.15K

# YOUR CODE HERE:
