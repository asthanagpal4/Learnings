# HOW TO RUN:
#   uv run python 01_foundations/project_word_counter.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# --- MINI-PROJECT: WORD FREQUENCY COUNTER ---
# This project brings together: strings, lists, dictionaries, functions, loops.
#
# You will build a program that counts how often each word appears in text.
# This is the foundation of NLP (Natural Language Processing), the field
# behind chatbots, translation, and large language models like ChatGPT.
#
# The structure is provided. Some parts are complete, some have TODOs
# for you to fill in. The program will run even before you fill in the TODOs,
# but it will work better once you complete them!


# ============================================================
# STEP 1: The sample text to analyze
# ============================================================
# (You can change this to any text you want later!)

sample_text = """
Machine learning is a branch of artificial intelligence.
Machine learning uses data to learn patterns.
The more data you have, the better the learning.
Data is the fuel for machine learning.
Without good data, machine learning cannot learn well.
Learning from data is what makes machine learning powerful.
"""


# ============================================================
# STEP 2: Clean the text
# ============================================================

def clean_text(text):
    """
    Takes raw text and returns a cleaned-up version.
    - Convert to lowercase (so "Machine" and "machine" count as same word)
    - Remove punctuation (periods, commas, etc.)
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation by keeping only letters, numbers, and spaces
    cleaned = ""
    for char in text:
        if char.isalpha() or char.isspace():
            # TODO 1: Add this character to the 'cleaned' string
            # Hint: cleaned = cleaned + char
            cleaned = cleaned + char

    return cleaned


# ============================================================
# STEP 3: Split text into a list of words
# ============================================================

def get_words(text):
    """
    Takes cleaned text and returns a list of words.
    """
    # TODO 2: Use the .split() method to break the text into words
    # Hint: "hello world".split() gives ["hello", "world"]
    words = text.split()
    return words


# ============================================================
# STEP 4: Count word frequencies
# ============================================================

def count_words(word_list):
    """
    Takes a list of words and returns a dictionary
    where keys are words and values are how many times they appear.

    Example: ["hi", "bye", "hi"] -> {"hi": 2, "bye": 1}
    """
    counts = {}

    for word in word_list:
        # TODO 3: Fill in the counting logic
        # If the word is already in counts, add 1 to its value
        # If the word is NOT in counts, set its value to 1
        #
        # Hint: you can use either the if/else pattern:
        #   if word in counts:
        #       counts[word] += 1
        #   else:
        #       counts[word] = 1
        #
        # Or the shorter .get() pattern:
        #   counts[word] = counts.get(word, 0) + 1

        counts[word] = counts.get(word, 0) + 1

    return counts


# ============================================================
# STEP 5: Sort words by frequency
# ============================================================

def sort_by_frequency(word_counts):
    """
    Takes a word counts dictionary and returns a list of (word, count)
    tuples, sorted from most frequent to least frequent.
    """
    # Convert dict to list of tuples, then sort
    # sorted() with key parameter tells Python what to sort by
    # reverse=True means highest first
    sorted_words = sorted(word_counts.items(), key=lambda pair: pair[1], reverse=True)
    return sorted_words


# ============================================================
# STEP 6: Display the results
# ============================================================

def display_results(sorted_words, top_n=10):
    """
    Prints the top N most frequent words in a nice format.
    """
    print("\n" + "=" * 40)
    print("   WORD FREQUENCY RESULTS")
    print("=" * 40)

    # TODO 4: Loop through the sorted_words (up to top_n items)
    # and print each word with its count and a simple bar chart.
    #
    # Expected output format:
    #     1. learning     : 5 | *****
    #     2. machine      : 4 | ****
    #     3. data         : 4 | ****
    #     ...
    #
    # Hint: use enumerate(sorted_words[:top_n], start=1)
    # The bar is just "*" multiplied by the count: "*" * count

    for rank, (word, count) in enumerate(sorted_words[:top_n], start=1):
        bar = "*" * count
        print(f"  {rank:>3}. {word:<15}: {count} | {bar}")

    print("=" * 40)


# ============================================================
# STEP 7: Extra stats
# ============================================================

def show_stats(word_list, word_counts):
    """
    Shows some extra statistics about the text.
    """
    print("\n--- Text Statistics ---")
    print(f"Total words: {len(word_list)}")
    print(f"Unique words: {len(word_counts)}")

    # TODO 5 (CHALLENGE): Calculate and print the average word length
    # Hint: loop through word_list, sum up len(word) for each word,
    # then divide by the number of words
    #
    # Expected output: something like "Average word length: 4.3"

    total_length = 0
    for word in word_list:
        total_length += len(word)

    if len(word_list) > 0:
        avg_length = total_length / len(word_list)
        print(f"Average word length: {avg_length:.1f}")

    print()


# ============================================================
# MAIN PROGRAM: Put it all together!
# ============================================================

def main():
    """
    The main function that runs the whole program.
    """
    print("Word Frequency Counter")
    print("Analyzing text...\n")

    # Step 1: Show the original text
    print("--- Original Text ---")
    print(sample_text.strip())

    # Step 2: Clean the text
    cleaned = clean_text(sample_text)

    # Step 3: Get list of words
    words = get_words(cleaned)
    print(f"\n(Found {len(words)} words after cleaning)")

    # Step 4: Count word frequencies
    counts = count_words(words)

    # Step 5: Sort by frequency
    sorted_words = sort_by_frequency(counts)

    # Step 6: Display top words
    display_results(sorted_words, top_n=10)

    # Step 7: Show stats
    show_stats(words, counts)

    # Bonus: show all unique words (using a set!)
    unique = set(words)
    print("--- All unique words (alphabetical) ---")
    for word in sorted(unique):
        print(f"  {word}")


# This line makes the main() function run when you execute the file
if __name__ == "__main__":
    main()


# === TEST CASES ===
# These asserts verify the functions above work correctly.
# If any assert fails, Python will tell you which one!

# Test: clean_text removes punctuation and lowercases
assert clean_text("Hello, World!") == "hello world", "clean_text should remove punctuation and lowercase"
assert clean_text("Machine Learning.") == "machine learning", "clean_text should handle period"
assert "a" in clean_text("A, B, C!") and "," not in clean_text("A, B, C!"), "clean_text should remove commas and exclamation"

# Test: get_words splits into a list of words
assert get_words("hello world") == ["hello", "world"], "get_words should split on spaces"
assert get_words("machine learning is fun") == ["machine", "learning", "is", "fun"], "get_words is wrong"

# Test: count_words counts correctly
assert count_words(["hi", "bye", "hi"]) == {"hi": 2, "bye": 1}, "count_words is wrong"
assert count_words(["a", "b", "a", "c", "a"]) == {"a": 3, "b": 1, "c": 1}, "count_words with 'a' repeated is wrong"
assert count_words([]) == {}, "count_words on empty list should return {}"

# Test: sort_by_frequency returns sorted tuples, most frequent first
sorted_result = sort_by_frequency({"apple": 3, "banana": 1, "cherry": 2})
assert sorted_result[0] == ("apple", 3), "most frequent word should come first"
assert sorted_result[-1] == ("banana", 1), "least frequent word should come last"

# Test: full pipeline on sample_text
_cleaned = clean_text(sample_text)
_words = get_words(_cleaned)
_counts = count_words(_words)
assert _counts["learning"] == 7, "learning should appear 7 times in sample_text"
assert _counts["machine"] == 5, "machine should appear 5 times in sample_text"
assert _counts["data"] == 5, "data should appear 5 times in sample_text"
assert len(_words) > 0, "word list should not be empty"
assert len(_counts) > 0, "word counts should not be empty"

print("\nAll tests passed!")

# ============================================================
# EXTRA CHALLENGES (after you've completed the TODOs above)
# ============================================================
#
# Challenge A: Modify the program to ignore "stop words" -- very common
#   words like "the", "is", "a", "and", "to", "for", "of", "in".
#   Create a set of stop words and skip them during counting.
#   Hint: in count_words(), add "if word not in stop_words:"
#
# Challenge B: Try running this program on different text!
#   Change sample_text to a paragraph from a news article or a book.
#
# Challenge C: Add a function that finds words that appear exactly once.
#   These are called "hapax legomena" -- a real NLP concept!
#   Hint: loop through word_counts and collect words where count == 1
#
# Challenge D: Make the program read text from a file instead of a string.
#   Hint: with open("filename.txt", "r") as f: text = f.read()
