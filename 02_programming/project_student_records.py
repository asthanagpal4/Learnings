# HOW TO RUN:
#   uv run python 02_programming/project_student_records.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- MINI-PROJECT: STUDENT GRADE TRACKER ---
# This project brings together everything from Section 2:
#   - Classes (to represent students)
#   - File I/O (to save and load records)
#   - Dictionaries (to organize data)
#   - Error handling (to handle bad input)
#   - List comprehensions (to process data)
#
# It works like a mini-database for student grades.
# The program runs as-is, but there are TODOs for you to fill in!


import os
import json


# ======================================================
# PART 1: THE STUDENT CLASS
# ======================================================

class Student:
    """Represents a single student with their scores."""

    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.scores = {}  # subject -> score, e.g. {"Math": 85, "Science": 92}

    def add_score(self, subject, score):
        """Add or update a score for a subject."""
        if not isinstance(score, (int, float)):
            raise TypeError(f"Score must be a number, got {type(score).__name__}")
        if score < 0 or score > 100:
            raise ValueError(f"Score must be 0-100, got {score}")
        self.scores[subject] = score

    def average(self):
        """Calculate the average score across all subjects."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def highest_subject(self):
        """Return the subject with the highest score."""
        if not self.scores:
            return None
        return max(self.scores, key=self.scores.get)

    def grade_letter(self):
        """Convert average score to a letter grade."""
        avg = self.average()
        if avg >= 90:
            return "A"
        elif avg >= 80:
            return "B"
        elif avg >= 70:
            return "C"
        elif avg >= 60:
            return "D"
        else:
            return "F"

    def __str__(self):
        avg = round(self.average(), 1)
        return f"Student({self.name}, ID={self.student_id}, avg={avg}, grade={self.grade_letter()})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        """Convert student to a dictionary (for saving to file)."""
        return {
            "name": self.name,
            "student_id": self.student_id,
            "scores": self.scores
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Student from a dictionary (for loading from file)."""
        student = cls(data["name"], data["student_id"])
        student.scores = data["scores"]
        return student


# ======================================================
# PART 2: THE GRADEBOOK CLASS
# ======================================================

class GradeBook:
    """Manages a collection of students. Like a mini-database."""

    def __init__(self, filename="gradebook.json"):
        self.students = {}    # student_id -> Student object
        self.filename = filename

    def add_student(self, name, student_id):
        """Add a new student to the gradebook."""
        if student_id in self.students:
            print(f"  Student ID {student_id} already exists!")
            return None
        student = Student(name, student_id)
        self.students[student_id] = student
        print(f"  Added: {student.name} (ID: {student_id})")
        return student

    def get_student(self, student_id):
        """Look up a student by their ID."""
        if student_id not in self.students:
            print(f"  Student ID {student_id} not found!")
            return None
        return self.students[student_id]

    def add_score(self, student_id, subject, score):
        """Add a score for a student."""
        student = self.get_student(student_id)
        if student is None:
            return
        try:
            student.add_score(subject, score)
            print(f"  {student.name}: {subject} = {score}")
        except (TypeError, ValueError) as e:
            print(f"  Error: {e}")

    def class_average(self):
        """Calculate the average score across ALL students."""
        if not self.students:
            return 0.0
        averages = [s.average() for s in self.students.values() if s.scores]
        if not averages:
            return 0.0
        return sum(averages) / len(averages)

    def top_students(self, n=3):
        """Return the top N students by average score."""
        ranked = sorted(
            self.students.values(),
            key=lambda s: s.average(),
            reverse=True
        )
        return ranked[:n]

    # TODO 1: Write a method called "failing_students" that returns a list
    # of students whose average is below 60.
    # Hint: use a list comprehension to filter self.students.values()
    # where student.average() < 60
    def failing_students(self):
        """Return students with average below 60."""
        # YOUR CODE HERE — replace the line below
        return []  # placeholder — should return list of Student objects

    def show_report(self):
        """Print a summary report of all students."""
        print("\n" + "=" * 60)
        print("GRADE BOOK REPORT")
        print("=" * 60)

        if not self.students:
            print("  No students in the gradebook.")
            return

        print(f"  Total students: {len(self.students)}")
        print(f"  Class average:  {round(self.class_average(), 1)}")
        print()

        # Print each student
        print(f"  {'Name':<15} {'ID':<8} {'Avg':<8} {'Grade':<6} {'Best Subject'}")
        print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*6} {'-'*15}")

        for student in sorted(self.students.values(), key=lambda s: s.average(), reverse=True):
            best = student.highest_subject() or "N/A"
            print(f"  {student.name:<15} {student.student_id:<8} "
                  f"{round(student.average(), 1):<8} {student.grade_letter():<6} {best}")

        print()

        # Top students
        top = self.top_students(3)
        if top:
            print("  Top students:")
            for i, s in enumerate(top, 1):
                print(f"    {i}. {s.name} (avg: {round(s.average(), 1)})")

        # Failing students
        failing = self.failing_students()
        if failing:
            print(f"\n  Students needing help ({len(failing)}):")
            for s in failing:
                print(f"    - {s.name} (avg: {round(s.average(), 1)})")

        print("=" * 60)

    # --- FILE OPERATIONS ---

    def save(self):
        """Save all student data to a JSON file."""
        data = {sid: student.to_dict() for sid, student in self.students.items()}
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {len(self.students)} students to {self.filename}")

    def load(self):
        """Load student data from a JSON file."""
        if not os.path.exists(self.filename):
            print(f"  No saved data found ({self.filename})")
            return

        try:
            with open(self.filename, "r") as f:
                data = json.load(f)
            self.students = {
                sid: Student.from_dict(sdata)
                for sid, sdata in data.items()
            }
            print(f"  Loaded {len(self.students)} students from {self.filename}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Error loading data: {e}")

    # TODO 2: Write a method called "export_csv" that saves the data as
    # a CSV file. Each row should have: name, student_id, average, grade_letter
    # The first row should be the header: "Name,ID,Average,Grade"
    # Hint: open a file with "w", write the header, then loop through students
    def export_csv(self, csv_filename="gradebook.csv"):
        """Export gradebook to CSV format."""
        # YOUR CODE HERE — the structure is:
        # with open(csv_filename, "w") as f:
        #     f.write("Name,ID,Average,Grade\n")
        #     for student in self.students.values():
        #         ... write each student as a row ...
        print(f"  (TODO: implement CSV export to {csv_filename})")


# ======================================================
# PART 3: SUBJECT STATISTICS
# ======================================================

def subject_stats(gradebook):
    """Calculate per-subject statistics across all students."""
    # Collect all scores by subject
    by_subject = {}
    for student in gradebook.students.values():
        for subject, score in student.scores.items():
            if subject not in by_subject:
                by_subject[subject] = []
            by_subject[subject].append(score)

    print("\n  Subject Statistics:")
    print(f"  {'Subject':<15} {'Avg':<8} {'Min':<8} {'Max':<8} {'Students'}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for subject, scores in sorted(by_subject.items()):
        avg = round(sum(scores) / len(scores), 1)
        print(f"  {subject:<15} {avg:<8} {min(scores):<8} {max(scores):<8} {len(scores)}")


# TODO 3: Write a function called "find_students_by_subject" that:
# - Takes a gradebook and a subject name
# - Returns a list of (student_name, score) tuples for that subject
# - Sorted by score from highest to lowest
# Hint: loop through gradebook.students.values(),
#        check if subject is in student.scores,
#        collect (student.name, student.scores[subject])
#        then sort with sorted(..., key=lambda x: x[1], reverse=True)

def find_students_by_subject(gradebook, subject):
    """Find all students who have a score for the given subject."""
    # YOUR CODE HERE — replace the line below
    return []  # placeholder


# ======================================================
# PART 4: MAIN PROGRAM
# ======================================================

def main():
    """Run the student grade tracker demo."""

    print("=" * 60)
    print("   STUDENT GRADE TRACKER")
    print("   A mini-project using classes, files, and dicts")
    print("=" * 60)

    # Create the gradebook
    gb = GradeBook()

    # --- Add students ---
    print("\nAdding students:")
    gb.add_student("Astha", "S001")
    gb.add_student("Priya", "S002")
    gb.add_student("Rahul", "S003")
    gb.add_student("Neha", "S004")
    gb.add_student("Amit", "S005")

    # --- Add scores ---
    print("\nAdding scores:")
    # Astha — strong student
    gb.add_score("S001", "Math", 88)
    gb.add_score("S001", "Science", 92)
    gb.add_score("S001", "English", 85)
    gb.add_score("S001", "Python", 95)

    # Priya — good student
    gb.add_score("S002", "Math", 76)
    gb.add_score("S002", "Science", 82)
    gb.add_score("S002", "English", 90)
    gb.add_score("S002", "Python", 78)

    # Rahul — average student
    gb.add_score("S003", "Math", 65)
    gb.add_score("S003", "Science", 70)
    gb.add_score("S003", "English", 68)
    gb.add_score("S003", "Python", 72)

    # Neha — needs help
    gb.add_score("S004", "Math", 45)
    gb.add_score("S004", "Science", 52)
    gb.add_score("S004", "English", 58)
    gb.add_score("S004", "Python", 40)

    # Amit — good in some subjects
    gb.add_score("S005", "Math", 90)
    gb.add_score("S005", "Science", 60)
    gb.add_score("S005", "English", 75)
    gb.add_score("S005", "Python", 85)

    # --- Test error handling ---
    print("\nTesting error handling:")
    gb.add_score("S001", "Math", 150)    # too high
    gb.add_score("S001", "Math", -10)    # too low
    gb.add_score("S999", "Math", 80)     # student doesn't exist

    # --- Show the report ---
    gb.show_report()

    # --- Subject statistics ---
    subject_stats(gb)

    # --- Test individual student ---
    print("\n\nIndividual student lookup:")
    astha = gb.get_student("S001")
    if astha:
        print(f"  {astha}")
        print(f"  Scores: {astha.scores}")
        print(f"  Best subject: {astha.highest_subject()}")

    # --- Save and Load ---
    print("\nSaving data:")
    gb.save()

    print("\nLoading data into a new gradebook:")
    gb2 = GradeBook()
    gb2.load()
    gb2.show_report()

    # --- Cleanup ---
    if os.path.exists("gradebook.json"):
        os.remove("gradebook.json")
        print("\nCleaned up gradebook.json")

    # --- TODO 4: Add your own code below! ---
    # Try these challenges:
    # a) Add a new student with scores and run show_report() again
    # b) Implement the failing_students() method (TODO 1 above)
    # c) Implement the export_csv() method (TODO 2 above)
    # d) Implement find_students_by_subject() (TODO 3 above)
    # e) Add a method to Student that returns the lowest subject

    print("\n" + "=" * 60)
    print("   Done! Check the TODOs in this file to practice more.")
    print("=" * 60)


# === TEST CASES ===
# These asserts verify the code above works correctly.
# If any assert fails, Python will tell you which one!

# Test 1: Student class — add_score, average, grade_letter, highest_subject
_st = Student("TestStudent", "T001")
assert _st.average() == 0.0, "Empty student average should be 0.0"
assert _st.highest_subject() is None, "Empty student highest_subject should be None"
assert _st.grade_letter() == "F", "Average 0 should give grade F"

_st.add_score("Math", 90)
_st.add_score("Science", 80)
assert _st.scores["Math"] == 90, "Math score should be 90"
assert _st.average() == 85.0, "Average of 90 and 80 should be 85.0"
assert _st.grade_letter() == "B", "Average 85 should give grade B"
assert _st.highest_subject() == "Math", "Highest subject should be Math"

# Test 2: Student raises errors for bad scores
_raised = False
try:
    _st.add_score("Bad", 150)
except ValueError:
    _raised = True
assert _raised, "Score > 100 should raise ValueError"

_raised = False
try:
    _st.add_score("Bad", -1)
except ValueError:
    _raised = True
assert _raised, "Negative score should raise ValueError"

_raised = False
try:
    _st.add_score("Bad", "ninety")
except TypeError:
    _raised = True
assert _raised, "Non-numeric score should raise TypeError"

# Test 3: Student grade boundaries
def _make_student_with_avg(avg):
    _s = Student("X", "X0")
    _s.add_score("Test", avg)
    return _s

assert _make_student_with_avg(95).grade_letter() == "A", "95 avg should be A"
assert _make_student_with_avg(85).grade_letter() == "B", "85 avg should be B"
assert _make_student_with_avg(75).grade_letter() == "C", "75 avg should be C"
assert _make_student_with_avg(65).grade_letter() == "D", "65 avg should be D"
assert _make_student_with_avg(45).grade_letter() == "F", "45 avg should be F"

# Test 4: Student to_dict and from_dict round-trip
_st2 = Student("Alice", "A001")
_st2.add_score("Math", 88)
_st2.add_score("English", 76)
_d = _st2.to_dict()
assert _d["name"] == "Alice", "to_dict name wrong"
assert _d["student_id"] == "A001", "to_dict student_id wrong"
assert _d["scores"]["Math"] == 88, "to_dict scores wrong"

_st3 = Student.from_dict(_d)
assert _st3.name == "Alice", "from_dict name wrong"
assert _st3.scores["Math"] == 88, "from_dict scores wrong"
assert _st3.average() == _st2.average(), "from_dict average should match original"

# Test 5: GradeBook — add_student, get_student, add_score, class_average
_gb = GradeBook(filename="_test_gradebook.json")
_s = _gb.add_student("Bob", "B001")
assert _s is not None, "add_student should return the Student object"
assert "B001" in _gb.students, "Student B001 should be in gradebook"

# Duplicate student_id returns None
_dup = _gb.add_student("Duplicate", "B001")
assert _dup is None, "Duplicate student_id should return None"

_gb.add_score("B001", "Math", 80)
_gb.add_score("B001", "Science", 90)
assert _gb.students["B001"].scores["Math"] == 80, "Score should be stored"
assert round(_gb.class_average(), 1) == 85.0, "Class average should be 85.0"

# Test 6: GradeBook — top_students ordering
_gb.add_student("Carol", "C001")
_gb.add_score("C001", "Math", 95)
_top = _gb.top_students(2)
assert len(_top) <= 2, "top_students(2) should return at most 2"
assert _top[0].name == "Carol", "Carol with 95 avg should be top student"

# Test 7: GradeBook — save and load round-trip
_gb.save()
_gb_loaded = GradeBook(filename="_test_gradebook.json")
_gb_loaded.load()
assert "B001" in _gb_loaded.students, "Loaded gradebook should have B001"
assert _gb_loaded.students["B001"].scores["Math"] == 80, "Loaded score should match"

# Cleanup test file
import os as _os
if _os.path.exists("_test_gradebook.json"):
    _os.remove("_test_gradebook.json")

# Test 8: subject_stats function runs without error (smoke test)
_gb2 = GradeBook()
_gb2.add_student("Dave", "D001")
_gb2.add_score("D001", "Math", 70)
subject_stats(_gb2)   # should not raise any exceptions

# Test 9: find_students_by_subject — placeholder returns a list
_result = find_students_by_subject(_gb2, "Math")
assert isinstance(_result, list), "find_students_by_subject should return a list"

print("\nAll tests passed!")

# This is the pattern from 06_modules.py!
# The code below only runs when you execute this file directly.
if __name__ == "__main__":
    main()
