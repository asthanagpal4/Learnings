# HOW TO RUN:
#   uv run python 04_algorithms/project_data_pipeline.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- MINI PROJECT: DATA PIPELINE ---
# This project ties together everything you've learned:
#   lists, dicts, file I/O, error handling, sorting, searching,
#   and (optionally) NumPy.
#
# You'll build a simple data pipeline — the same kind of thing ML engineers
# do every day before training a model:
#   1. Load data from a CSV file
#   2. Clean the data (handle missing values, fix formats)
#   3. Compute statistics (mean, median, min, max)
#   4. Filter and sort the data
#   5. Save the cleaned data
#
# This is REAL ML preprocessing, just on a small scale!

import csv
import os
import time

# Try to import numpy (optional — the project works without it too)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("NumPy is available! We'll use it for some computations.")
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not installed. No worries — we'll use plain Python!")

print()
print("=" * 60)
print("MINI PROJECT: DATA PIPELINE")
print("A taste of real ML data preprocessing!")
print("=" * 60)


# ============================================================
# STEP 1: CREATE A SAMPLE CSV FILE
# ============================================================
# In real ML, you'd download a dataset. Here we'll create one.
# This simulates a dataset of students with their scores and info.

print("\n--- STEP 1: Creating sample dataset ---\n")

# This is our raw data — notice it has messy parts (like real data!)
sample_data = """name,age,math_score,science_score,english_score,city
Astha,25,88,92,85,Delhi
Bob,22,76,,79,Mumbai
Cara,28,95,88,91,Delhi
Dev,,65,70,68,Bangalore
Eve,24,82,85,,Chennai
Frank,26,91,94,89,Mumbai
Grace,23,,78,82,Delhi
Hank,30,58,62,55,Chennai
Ivy,21,97,99,95,Bangalore
Jack,27,73,68,71,Mumbai
Kara,25,84,87,80,Delhi
Leo,29,45,50,48,
Mia,22,88,91,86,Chennai
Nate,26,79,82,75,Bangalore
Olivia,24,92,90,88,Delhi
Pete,31,60,,63,Mumbai
Quinn,23,85,83,87,Chennai
Rose,27,71,75,69,Bangalore
Sam,25,93,96,90,Delhi
Tina,22,67,72,missing,Mumbai"""

# Write the CSV file
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_data.csv")
with open(csv_path, "w") as f:
    f.write(sample_data)
print(f"Created: {csv_path}")
print(f"File size: {os.path.getsize(csv_path)} bytes")


# ============================================================
# STEP 2: LOAD THE DATA
# ============================================================

print("\n--- STEP 2: Loading data ---\n")

def load_csv(filepath):
    """Load a CSV file and return a list of dictionaries (one per row)."""
    data = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))  # Convert each row to a regular dict
    return data

raw_data = load_csv(csv_path)
print(f"Loaded {len(raw_data)} rows")
print(f"Columns: {list(raw_data[0].keys())}")
print(f"\nFirst 3 rows (raw):")
for row in raw_data[:3]:
    print(f"  {row}")


# ============================================================
# STEP 3: CLEAN THE DATA
# ============================================================
# Real data is ALWAYS messy. Common problems:
#   - Missing values (empty strings, "missing", "N/A")
#   - Wrong data types (everything from CSV is a string)
#   - Inconsistent formats
#
# This is THE most important step in ML — bad data = bad model!

print("\n--- STEP 3: Cleaning data ---\n")

def is_missing(value):
    """Check if a value is missing or invalid."""
    if value is None:
        return True
    value = str(value).strip().lower()
    return value in ("", "missing", "n/a", "na", "null", "none")

def safe_int(value, default=None):
    """Convert a value to int, returning default if it fails."""
    if is_missing(value):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value, default=None):
    """Convert a value to float, returning default if it fails."""
    if is_missing(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def clean_data(raw_rows):
    """Clean the raw data and return a list of cleaned dictionaries."""
    cleaned = []
    issues_found = 0

    for i, row in enumerate(raw_rows):
        clean_row = {}

        # Name: strip whitespace
        clean_row["name"] = row.get("name", "").strip()

        # Age: convert to int, track if missing
        age = safe_int(row.get("age"))
        if age is None:
            issues_found += 1
            print(f"  Warning: Missing age for {clean_row['name']} (row {i+1})")
        clean_row["age"] = age

        # Scores: convert to float, track if missing
        for subject in ["math_score", "science_score", "english_score"]:
            score = safe_float(row.get(subject))
            if score is None:
                issues_found += 1
                print(f"  Warning: Missing {subject} for {clean_row['name']} (row {i+1})")
            clean_row[subject] = score

        # City: strip whitespace, handle missing
        city = row.get("city", "").strip()
        if is_missing(city):
            issues_found += 1
            city = "Unknown"
            print(f"  Warning: Missing city for {clean_row['name']} (row {i+1})")
        clean_row["city"] = city

        cleaned.append(clean_row)

    print(f"\nCleaning complete! Found {issues_found} issues.")
    return cleaned

cleaned_data = clean_data(raw_data)
print(f"Cleaned {len(cleaned_data)} rows")


# ============================================================
# STEP 4: HANDLE MISSING VALUES
# ============================================================
# Common strategies for missing values:
#   1. Drop rows with missing data
#   2. Fill with a default (mean, median, 0)
# We'll use strategy 2: fill missing scores with the average.

print("\n--- STEP 4: Handling missing values ---\n")

def compute_column_mean(data, column):
    """Compute the mean of a column, ignoring None values."""
    values = [row[column] for row in data if row[column] is not None]
    if not values:
        return 0
    return sum(values) / len(values)

# Fill missing scores with the column average
for column in ["math_score", "science_score", "english_score"]:
    mean_val = compute_column_mean(cleaned_data, column)
    filled_count = 0
    for row in cleaned_data:
        if row[column] is None:
            row[column] = round(mean_val, 1)
            filled_count += 1
    print(f"  {column}: mean = {mean_val:.1f}, filled {filled_count} missing values")

# Fill missing ages with median
ages = sorted([row["age"] for row in cleaned_data if row["age"] is not None])
median_age = ages[len(ages) // 2]
for row in cleaned_data:
    if row["age"] is None:
        row["age"] = median_age
        print(f"  Filled missing age for {row['name']} with median: {median_age}")


# ============================================================
# STEP 5: COMPUTE STATISTICS
# ============================================================

print("\n--- STEP 5: Computing statistics ---\n")

def compute_stats(values):
    """Compute basic statistics for a list of numbers."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    stats = {
        "count": n,
        "mean": sum(values) / n,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "median": sorted_vals[n // 2] if n % 2 == 1
                  else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2,
    }

    # Standard deviation
    mean = stats["mean"]
    variance = sum((x - mean) ** 2 for x in values) / n
    stats["std_dev"] = variance ** 0.5

    return stats

# Compute stats for each score column
for column in ["math_score", "science_score", "english_score", "age"]:
    values = [row[column] for row in cleaned_data]
    stats = compute_stats(values)
    print(f"  {column}:")
    print(f"    Count: {stats['count']}, Mean: {stats['mean']:.1f}, "
          f"Median: {stats['median']:.1f}")
    print(f"    Min: {stats['min']}, Max: {stats['max']}, "
          f"Std Dev: {stats['std_dev']:.1f}")

# If NumPy is available, let's also do it the NumPy way
if NUMPY_AVAILABLE:
    print("\n  --- Same stats using NumPy (much simpler code!) ---")
    for column in ["math_score", "science_score", "english_score"]:
        values = np.array([row[column] for row in cleaned_data])
        print(f"  {column}: mean={np.mean(values):.1f}, "
              f"std={np.std(values):.1f}, "
              f"min={np.min(values):.0f}, max={np.max(values):.0f}")


# ============================================================
# STEP 6: ANALYZE AND FILTER DATA
# ============================================================

print("\n--- STEP 6: Analysis and filtering ---\n")

# Add a computed column: average score
for row in cleaned_data:
    row["avg_score"] = round(
        (row["math_score"] + row["science_score"] + row["english_score"]) / 3, 1
    )

# Sort by average score (highest first)
ranked = sorted(cleaned_data, key=lambda r: r["avg_score"], reverse=True)

print("Top 5 students by average score:")
print(f"  {'Name':<10} {'Math':>6} {'Science':>8} {'English':>8} {'Average':>8}")
print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
for row in ranked[:5]:
    print(f"  {row['name']:<10} {row['math_score']:>6.1f} {row['science_score']:>8.1f} "
          f"{row['english_score']:>8.1f} {row['avg_score']:>8.1f}")

print(f"\nBottom 3 students:")
for row in ranked[-3:]:
    print(f"  {row['name']:<10} Average: {row['avg_score']:.1f}")

# Count students by city
print("\nStudents per city:")
city_counts = {}
city_avg_scores = {}
for row in cleaned_data:
    city = row["city"]
    city_counts[city] = city_counts.get(city, 0) + 1
    if city not in city_avg_scores:
        city_avg_scores[city] = []
    city_avg_scores[city].append(row["avg_score"])

for city in sorted(city_counts.keys()):
    count = city_counts[city]
    avg = sum(city_avg_scores[city]) / len(city_avg_scores[city])
    print(f"  {city:<12} {count} students, avg score: {avg:.1f}")

# Filter: students who scored above 80 in ALL subjects
high_achievers = [
    row for row in cleaned_data
    if row["math_score"] >= 80 and row["science_score"] >= 80
    and row["english_score"] >= 80
]
print(f"\nHigh achievers (80+ in all subjects): {len(high_achievers)}")
for row in high_achievers:
    print(f"  {row['name']} — Math: {row['math_score']:.0f}, "
          f"Science: {row['science_score']:.0f}, "
          f"English: {row['english_score']:.0f}")


# ============================================================
# STEP 7: SAVE CLEANED DATA
# ============================================================

print("\n--- STEP 7: Saving cleaned data ---\n")

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "student_data_cleaned.csv")

columns = ["name", "age", "math_score", "science_score",
           "english_score", "city", "avg_score"]

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(cleaned_data)

print(f"Saved cleaned data to: {output_path}")
print(f"File size: {os.path.getsize(output_path)} bytes")


# ============================================================
# STEP 8: SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)
print(f"""
What we did (just like a real ML pipeline):
  1. Created a raw CSV dataset (simulating downloading data)
  2. Loaded it into Python (parsing)
  3. Cleaned messy values (missing data, wrong types)
  4. Filled missing values with column averages (imputation)
  5. Computed statistics (understanding your data)
  6. Filtered and sorted (feature engineering, data selection)
  7. Saved the clean version (ready for ML training!)

Files created:
  Raw data:     {csv_path}
  Cleaned data: {output_path}
""")


# === TEST CASES ===
# These asserts verify the pipeline functions work correctly.
# If any assert fails, Python will tell you which one!

# is_missing: truthy missing values
assert is_missing("") == True
assert is_missing("missing") == True
assert is_missing("Missing") == True    # case-insensitive
assert is_missing("N/A") == True
assert is_missing("na") == True
assert is_missing("null") == True
assert is_missing("none") == True
assert is_missing(None) == True
# is_missing: valid values should return False
assert is_missing("Astha") == False
assert is_missing("Delhi") == False
assert is_missing("0") == False

# safe_int: converts correctly
assert safe_int("25") == 25
assert safe_int("0") == 0
assert safe_int("") is None
assert safe_int("missing") is None
assert safe_int("abc") is None
assert safe_int(None) is None
assert safe_int("42", default=0) == 42
assert safe_int("", default=0) == 0

# safe_float: converts correctly
assert safe_float("3.14") == 3.14
assert safe_float("88") == 88.0
assert safe_float("") is None
assert safe_float("missing") is None
assert safe_float(None) is None

# compute_column_mean: ignores None values
assert compute_column_mean([{"score": 10}, {"score": 20}, {"score": None}], "score") == 15.0
assert compute_column_mean([{"score": None}], "score") == 0  # all missing -> 0
assert compute_column_mean([{"score": 100}], "score") == 100.0

# compute_stats: known values
stats = compute_stats([1, 2, 3, 4, 5])
assert stats["count"] == 5
assert abs(stats["mean"] - 3.0) < 0.0001
assert stats["min"] == 1
assert stats["max"] == 5
assert stats["median"] == 3

stats_even = compute_stats([1, 2, 3, 4])
assert abs(stats_even["median"] - 2.5) < 0.0001

# pipeline results: verify actual pipeline output on cleaned_data
assert len(cleaned_data) == 20        # all 20 rows kept
assert cleaned_data[0]["name"] == "Astha"

# No None values should remain after missing-value filling
for row in cleaned_data:
    assert row["math_score"] is not None
    assert row["science_score"] is not None
    assert row["english_score"] is not None
    assert row["age"] is not None

# avg_score is computed for all rows
for row in cleaned_data:
    assert "avg_score" in row
    expected_avg = round((row["math_score"] + row["science_score"] + row["english_score"]) / 3, 1)
    assert abs(row["avg_score"] - expected_avg) < 0.05

# high_achievers: every person in the list must score >= 80 in all subjects
for row in high_achievers:
    assert row["math_score"] >= 80
    assert row["science_score"] >= 80
    assert row["english_score"] >= 80

print("\nAll tests passed!")

# ============================================================
# YOUR TURN: TODOs TO EXTEND THIS PROJECT
# ============================================================

print("=" * 60)
print("TODOs FOR ASTHA — Extend this project!")
print("=" * 60)
print("""
Here are things you can add to practice. Try them one at a time!

TODO 1 (Easy): Add a new column "grade" based on avg_score:
   - A: 90+, B: 80-89, C: 70-79, D: 60-69, F: below 60
   - Print how many students got each grade.
   Hint: Use if/elif in a loop over cleaned_data.

TODO 2 (Easy): Find the subject with the highest average score.
   Hint: Compute the mean for each subject column, compare them.

TODO 3 (Medium): Add a search feature — let the user type a name
   and find that student's data.
   Hint: Use input() and linear search through cleaned_data.

TODO 4 (Medium): Detect outliers — find students whose score in any
   subject is more than 2 standard deviations from the mean.
   Hint: If abs(score - mean) > 2 * std_dev, it's an outlier.

TODO 5 (Harder): Read a REAL CSV file! Download any CSV dataset from:
   https://www.kaggle.com/datasets
   Modify load_csv() and clean_data() to work with it.

TODO 6 (ML Preview): If you have NumPy, try "normalizing" the scores
   to be between 0 and 1: normalized = (score - min) / (max - min)
   This is a common ML preprocessing step!
""")

# Write your TODO solutions below:
# -----------------------------------


print("Done! Great job completing this mini-project!")
