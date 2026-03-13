# HOW TO RUN:
#   uv run python 02_programming/05_classes_advanced.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- CLASSES ADVANCED: INHERITANCE & SPECIAL METHODS ---
# Now that you know the basics of classes, let's learn two powerful ideas:
# 1. Inheritance — making new classes based on existing ones
# 2. Special methods — making your objects work with print(), len(), +, etc.
# In ML, PyTorch models use inheritance (class MyModel(nn.Module)).


# === CONCEPT 1: INHERITANCE — BUILDING ON EXISTING CLASSES ===
# Inheritance lets you create a new class that gets everything from a parent class.
# The child class can add new stuff or change existing behavior.
# Think: Animal (parent) -> Dog, Cat (children)

print("=" * 50)
print("CONCEPT 1: Inheritance basics")
print("=" * 50)

class Animal:
    """Parent class (also called 'base class' or 'superclass')."""

    def __init__(self, name, sound):
        self.name = name
        self.sound = sound

    def speak(self):
        print(f"{self.name} says {self.sound}!")

    def info(self):
        print(f"I'm {self.name}, an animal")

# Child classes — they inherit everything from Animal
class Dog(Animal):
    """Dog inherits from Animal. Passes 'Animal' in the parentheses."""

    def __init__(self, name, breed):
        # super() calls the parent's __init__
        super().__init__(name, sound="Woof")
        self.breed = breed       # new attribute only for dogs

    def fetch(self):             # new method only for dogs
        print(f"{self.name} fetches the ball!")

class Cat(Animal):
    def __init__(self, name, indoor=True):
        super().__init__(name, sound="Meow")
        self.indoor = indoor

    def purr(self):
        print(f"{self.name} purrs...")

# Create objects
buddy = Dog("Buddy", "Golden Retriever")
whiskers = Cat("Whiskers")

buddy.speak()       # Output: Buddy says Woof!  (inherited from Animal)
buddy.fetch()       # Output: Buddy fetches the ball!  (Dog's own method)

whiskers.speak()    # Output: Whiskers says Meow!  (inherited from Animal)
whiskers.purr()     # Output: Whiskers purrs...  (Cat's own method)

# Dog has .breed, Cat does not. Cat has .indoor, Dog does not.
print(f"Buddy's breed: {buddy.breed}")
print(f"Whiskers indoor? {whiskers.indoor}")


# === CONCEPT 2: OVERRIDING METHODS ===
# A child class can replace (override) a parent's method.

print("\n" + "=" * 50)
print("CONCEPT 2: Overriding methods")
print("=" * 50)

class Shape:
    def __init__(self, name):
        self.name = name

    def area(self):
        return 0  # default — child classes will override this

    def describe(self):
        print(f"{self.name}: area = {self.area()}")

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius

    def area(self):  # overrides Shape's area()
        return 3.14159 * self.radius ** 2

class Square(Shape):
    def __init__(self, side):
        super().__init__("Square")
        self.side = side

    def area(self):  # overrides Shape's area()
        return self.side ** 2

shapes = [Circle(5), Square(4), Circle(3)]
for shape in shapes:
    shape.describe()
# Output:
# Circle: area = 78.53975
# Square: area = 16
# Circle: area = 28.27431


# === CONCEPT 3: __str__ — MAKING PRINT() WORK NICELY ===
# By default, print(object) shows something ugly like <__main__.Dog object at 0x...>.
# Define __str__ to control what print() shows.

print("\n" + "=" * 50)
print("CONCEPT 3: __str__ method")
print("=" * 50)

class StudentBad:
    """Without __str__"""
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

class StudentGood:
    """With __str__"""
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def __str__(self):
        return f"Student({self.name}, grade={self.grade})"

bad = StudentBad("Astha", "A")
good = StudentGood("Astha", "A")

print(f"Without __str__: {bad}")
# Output: Without __str__: <__main__.StudentBad object at 0x...>

print(f"With __str__:    {good}")
# Output: With __str__:    Student(Astha, grade=A)


# === CONCEPT 4: __repr__ — FOR DEVELOPERS ===
# __repr__ is like __str__ but meant for debugging.
# If __str__ is not defined, Python falls back to __repr__.

print("\n" + "=" * 50)
print("CONCEPT 4: __repr__ method")
print("=" * 50)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

p = Point(3, 7)
print(f"str:  {str(p)}")     # Output: str:  (3, 7)
print(f"repr: {repr(p)}")    # Output: repr: Point(x=3, y=7)

# In a list, Python uses __repr__
points = [Point(1, 2), Point(3, 4)]
print(f"List: {points}")
# Output: List: [Point(x=1, y=2), Point(x=3, y=4)]


# === CONCEPT 5: __len__ — MAKING len() WORK ===
# Define __len__ so len(your_object) returns something meaningful.

print("\n" + "=" * 50)
print("CONCEPT 5: __len__ method")
print("=" * 50)

class Playlist:
    def __init__(self, name):
        self.name = name
        self.songs = []

    def add(self, song):
        self.songs.append(song)

    def __len__(self):
        return len(self.songs)

    def __str__(self):
        return f"Playlist '{self.name}' ({len(self)} songs)"

my_playlist = Playlist("Study Music")
my_playlist.add("Lofi Beat 1")
my_playlist.add("Lofi Beat 2")
my_playlist.add("Lofi Beat 3")

print(f"Number of songs: {len(my_playlist)}")
# Output: Number of songs: 3

print(my_playlist)
# Output: Playlist 'Study Music' (3 songs)


# === CONCEPT 6: __eq__ — COMPARING OBJECTS ===
# By default, two objects are equal only if they're the same object.
# Define __eq__ to compare based on values.

print("\n" + "=" * 50)
print("CONCEPT 6: __eq__ for comparison")
print("=" * 50)

class Color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __eq__(self, other):
        if not isinstance(other, Color):
            return False
        return self.r == other.r and self.g == other.g and self.b == other.b

    def __str__(self):
        return f"Color({self.r}, {self.g}, {self.b})"

red1 = Color(255, 0, 0)
red2 = Color(255, 0, 0)
blue = Color(0, 0, 255)

print(f"red1 == red2? {red1 == red2}")  # Output: True  (same values)
print(f"red1 == blue? {red1 == blue}")  # Output: False (different values)


# === CONCEPT 7: __add__ AND __getitem__ ===
# __add__ lets you use + with your objects.
# __getitem__ lets you use [] indexing.

print("\n" + "=" * 50)
print("CONCEPT 7: __add__ and __getitem__")
print("=" * 50)

class Scores:
    def __init__(self, values):
        self.values = list(values)

    def __add__(self, other):
        # Combine two Scores objects
        return Scores(self.values + other.values)

    def __getitem__(self, index):
        # Allow indexing like scores[0]
        return self.values[index]

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return f"Scores({self.values})"

math_scores = Scores([85, 92, 78])
science_scores = Scores([90, 88])

all_scores = math_scores + science_scores  # uses __add__
print(f"Combined: {all_scores}")
# Output: Combined: Scores([85, 92, 78, 90, 88])

print(f"First score: {all_scores[0]}")    # uses __getitem__
# Output: First score: 85

print(f"Total scores: {len(all_scores)}") # uses __len__
# Output: Total scores: 5


# === CONCEPT 8: PRACTICAL EXAMPLE — A DATASET CLASS ===
# This is similar to how ML frameworks organize data.

print("\n" + "=" * 50)
print("CONCEPT 8: Practical Dataset class")
print("=" * 50)

class Dataset:
    """A simple dataset class — similar to what you'd see in ML."""

    def __init__(self, name):
        self.name = name
        self.data = []        # list of (input, label) pairs

    def add_sample(self, features, label):
        self.data.append((features, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __str__(self):
        return f"Dataset('{self.name}', {len(self)} samples)"

    def get_labels(self):
        return [label for _, label in self.data]

    def summary(self):
        labels = self.get_labels()
        unique = set(labels)
        print(f"  Dataset: {self.name}")
        print(f"  Samples: {len(self)}")
        print(f"  Labels:  {unique}")
        for label in unique:
            count = labels.count(label)
            print(f"    '{label}': {count} samples")

# Build a tiny dataset
ds = Dataset("Sentiment")
ds.add_sample(["I", "love", "this"], "positive")
ds.add_sample(["Great", "movie"], "positive")
ds.add_sample(["Terrible", "food"], "negative")
ds.add_sample(["Bad", "service"], "negative")
ds.add_sample(["Amazing", "book"], "positive")

print(ds)               # Output: Dataset('Sentiment', 5 samples)
print(f"Sample 0: {ds[0]}")  # Output: Sample 0: (['I', 'love', 'this'], 'positive')
ds.summary()
# Output:
#   Dataset: Sentiment
#   Samples: 5
#   Labels:  {'positive', 'negative'}
#     'positive': 3 samples
#     'negative': 2 samples


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test 1: Inheritance — Dog and Cat inherit from Animal
_buddy = Dog("Buddy", "Golden Retriever")
assert _buddy.name == "Buddy", "Dog name should be 'Buddy'"
assert _buddy.breed == "Golden Retriever", "Dog breed wrong"
assert _buddy.sound == "Woof", "Dog sound should be 'Woof' (set by super().__init__)"

_whiskers = Cat("Whiskers")
assert _whiskers.name == "Whiskers", "Cat name wrong"
assert _whiskers.sound == "Meow", "Cat sound should be 'Meow'"
assert _whiskers.indoor == True, "Default indoor should be True"

# Test 2: Method overriding — Circle and Square override Shape.area()
_c = Circle(5)
assert round(_c.area(), 2) == 78.54, "Circle area of radius 5 should be ~78.54"
_sq = Square(4)
assert _sq.area() == 16, "Square area of side 4 should be 16"

# Default Shape area is 0
_base = Shape("base")
assert _base.area() == 0, "Default Shape area should be 0"

# Test 3: __str__ method
_good = StudentGood("Astha", "A")
assert str(_good) == "Student(Astha, grade=A)", "__str__ output wrong"

# Test 4: __repr__ method
_p = Point(3, 7)
assert str(_p) == "(3, 7)", "__str__ of Point wrong"
assert repr(_p) == "Point(x=3, y=7)", "__repr__ of Point wrong"

# Test 5: __len__ method
_pl = Playlist("Test")
assert len(_pl) == 0, "Empty playlist should have len 0"
_pl.add("Song A")
_pl.add("Song B")
assert len(_pl) == 2, "Playlist with 2 songs should have len 2"
assert str(_pl) == "Playlist 'Test' (2 songs)", "__str__ of Playlist wrong"

# Test 6: __eq__ method
_red1 = Color(255, 0, 0)
_red2 = Color(255, 0, 0)
_blue = Color(0, 0, 255)
assert _red1 == _red2, "Two identical Colors should be equal"
assert not (_red1 == _blue), "Different Colors should not be equal"
assert not (_red1 == "not a color"), "Color should not equal a non-Color"

# Test 7: __add__ and __getitem__
_s1 = Scores([10, 20])
_s2 = Scores([30, 40])
_combined = _s1 + _s2
assert len(_combined) == 4, "Combined Scores should have 4 elements"
assert _combined[0] == 10, "First element of combined should be 10"
assert _combined[3] == 40, "Last element of combined should be 40"

# Test 8: Dataset class
_ds = Dataset("TestSet")
assert len(_ds) == 0, "New dataset should have 0 samples"
_ds.add_sample(["word1"], "positive")
_ds.add_sample(["word2"], "negative")
assert len(_ds) == 2, "After 2 adds, dataset should have 2 samples"
assert _ds[0] == (["word1"], "positive"), "First sample wrong"
assert _ds.get_labels() == ["positive", "negative"], "Labels wrong"
assert str(_ds) == "Dataset('TestSet', 2 samples)", "__str__ of Dataset wrong"

print("\nAll tests passed!")

# === EXERCISES ===
# Try these yourself! Write your code below each exercise.

# 1. Create a class hierarchy for vehicles:
#    - Parent class: Vehicle with name and top_speed
#    - Child class: Car with num_doors (default 4)
#    - Child class: Motorcycle with sidecar (default False)
#    - Each child should have a describe() method that prints its details.
#    Hint: use super().__init__() in each child class

# YOUR CODE HERE:


# 2. Create a class called "WordCounter" that:
#    - Takes a string of text in __init__
#    - Has __len__ that returns the number of words
#    - Has __str__ that returns "WordCounter: X words"
#    - Has a method most_common() that returns the most frequent word
#    Test: WordCounter("the cat sat on the mat the cat")
#    Expected: len = 8, most_common = "the"
#    Hint: use .split() and .count()

# YOUR CODE HERE:


# 3. Create a class "Vector" that represents a 2D point (x, y) with:
#    - __str__ returns "(x, y)"
#    - __add__ adds two vectors: Vector(1,2) + Vector(3,4) = Vector(4,6)
#    - __eq__ compares two vectors
#    - A method magnitude() that returns the length: (x**2 + y**2) ** 0.5
#    Test: Vector(3, 4).magnitude() should return 5.0
#    Hint: __add__ should return a new Vector object

# YOUR CODE HERE:
