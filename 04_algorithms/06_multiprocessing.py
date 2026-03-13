# HOW TO RUN:
#   uv run python 04_algorithms/06_multiprocessing.py
# If all tests pass, you'll see "All tests passed!" at the end.
# If a test fails, Python will show which assert failed and on which line.

# =============================================================================
# 06_multiprocessing.py — Processes, Threads, and Doing Many Things at Once
# =============================================================================
#
# Run this file with:
#   uv run python 04_algorithms/06_multiprocessing.py
#
# What you will learn:
#   1. What is concurrency and why it matters
#   2. Threads vs Processes — what's the difference?
#   3. The GIL — why Python threads have a hidden limitation
#   4. threading module — good for waiting tasks (I/O-bound)
#   5. multiprocessing module — good for heavy math tasks (CPU-bound)
#   6. concurrent.futures — the easy, modern way to do both
#   7. Practical example: loading multiple data files in parallel
#   8. Decision guide — when to use what
# =============================================================================

import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# =============================================================================
# SECTION 1: What is Concurrency?
# =============================================================================
#
# Imagine you are cooking dinner:
#   - You put rice on the stove (it will cook for 20 minutes on its own)
#   - While rice is cooking, you chop vegetables
#   - While vegetables roast, you prepare a salad
#
# You are ONE person, but you are making progress on MULTIPLE dishes at the
# same time. That is concurrency — doing many things in overlapping time.
#
# Without concurrency (sequential):
#   Wait for rice -> Wait for vegetables -> Make salad  (total: 40 minutes)
#
# With concurrency:
#   Start rice, chop while waiting, make salad while roasting  (total: 20 min)
#
# In programming, concurrency matters when:
#   - You need to download 100 files (don't wait for each one to finish)
#   - You need to process huge amounts of data (split the work across cores)
#   - You need your program to stay responsive (don't freeze the screen)

print("=" * 60)
print("SECTION 1: Concurrency — Cooking Many Dishes")
print("=" * 60)

def cook_dish_sequential(dishes):
    """Cook dishes one at a time — slow."""
    start = time.time()
    for dish in dishes:
        time.sleep(0.3)  # simulate cooking time
    end = time.time()
    print(f"  Sequential: cooked {len(dishes)} dishes in {end - start:.2f} seconds")

dishes = ["Rice", "Dal", "Sabzi", "Roti", "Salad"]
cook_dish_sequential(dishes)


# =============================================================================
# SECTION 2: Threads vs Processes
# =============================================================================
#
# A PROCESS is a running program. When you open Python, that is one process.
# Each process has its own memory space — it doesn't share memory with others.
#
# A THREAD is a smaller unit inside a process. One process can have many
# threads. Threads share the same memory, so they can talk to each other
# easily — but that also causes problems (two threads editing the same
# variable at the same time = chaos).
#
# Simple analogy:
#   Process = a restaurant kitchen
#   Thread  = a chef inside that kitchen
#
#   Multiple kitchens (processes) = totally separate, no sharing
#   Multiple chefs (threads)      = same kitchen, share tools, can bump into
#                                   each other
#
#   Python has a rule for threads called the GIL (explained next), which
#   means threads can get in each other's way for Python code.

print("\n" + "=" * 60)
print("SECTION 2: Threads vs Processes")
print("=" * 60)
print("  Thread  = chef inside one kitchen (shared memory)")
print("  Process = separate kitchen entirely (separate memory)")
print("  Use threads for waiting tasks, processes for heavy computation.")

# -----------------------------------------------------------------------------
# FIRST-PRINCIPLES: Process Spawning Overhead
# -----------------------------------------------------------------------------
#
# Creating a new process is O(1) in algorithmic terms (you call fork/spawn
# once), but the CONSTANT is large:
#
#   - fork() on Linux copies the entire process address space. Modern kernels
#     use copy-on-write (COW) so physical memory is not duplicated immediately,
#     but the page tables themselves must be duplicated. For a Python process
#     using 200 MB of RAM, this can take several milliseconds.
#
#   - On Windows (and with Python's "spawn" start method), the situation is
#     worse: a brand-new Python interpreter is started, the module is
#     re-imported, and the target function + arguments are serialized with
#     pickle, sent over a pipe, and deserialized on the other side.
#
# Compare this to THREADS:
#   - Creating a thread is much cheaper (microseconds, not milliseconds).
#   - Threads share the same memory — no copying, no serialization.
#   - But threads are limited by the GIL for CPU-bound Python code.
#
# Rule of thumb:
#   - If each task takes < 1ms, the process-spawn overhead dominates.
#     Spawning 1000 processes for 1000 tiny tasks is SLOWER than sequential.
#   - If each task takes > 100ms, the spawn overhead is negligible.
#   - For tasks in between, benchmark to decide.
#
# This is why ProcessPoolExecutor REUSES processes (a pool) rather than
# spawning a new one for every single task.
# -----------------------------------------------------------------------------


# =============================================================================
# SECTION 3: The GIL (Global Interpreter Lock)
# =============================================================================
#
# Python has a rule called the GIL — Global Interpreter Lock.
#
# The GIL is like a single microphone in a meeting room.
# Only the person holding the microphone can speak (run Python code).
# Even if 4 people (threads) are in the room, only 1 speaks at a time.
#
# This means Python THREADS cannot truly run Python code in parallel.
# They take turns very quickly, which LOOKS like parallelism but isn't.
#
# HOWEVER — the GIL is released when a thread is just waiting:
#   - Waiting for a file to load from disk
#   - Waiting for a website to respond
#   - Waiting for a database query
#
# During that wait, another thread CAN run. So threads are still useful
# for tasks that involve a lot of waiting (called I/O-bound tasks).
#
# For tasks that need real CPU power (math, processing data), use
# multiprocessing — each process has its OWN GIL, so they truly run
# in parallel on different CPU cores.

print("\n" + "=" * 60)
print("SECTION 3: The GIL — Python's Hidden Traffic Rule")
print("=" * 60)
print("  GIL = only ONE thread runs Python code at a time")
print("  BUT: GIL is released during I/O waits (file reads, network)")
print("  So threads help for waiting tasks, NOT for heavy computation.")
print("  For real parallelism: use multiprocessing (separate GIL per process)")

# -----------------------------------------------------------------------------
# FIRST-PRINCIPLES: Memory Cost — Processes vs Threads vs Asyncio
# -----------------------------------------------------------------------------
#
# When you choose a concurrency model, memory is a hidden cost.
# Here is how the three main approaches compare:
#
#   1. multiprocessing (separate processes)
#      - Each process gets a FULL COPY of the Python interpreter and your data.
#      - If your main process uses 200 MB, each child adds ~200 MB (before COW
#        pages diverge). 4 worker processes = ~800 MB total.
#      - Data passed between processes must be serialized (pickled) and
#        deserialized — this copies the data AGAIN.
#      - Best when: you need true CPU parallelism AND the data per task is small.
#
#   2. threading (shared memory, but GIL)
#      - All threads share the SAME memory space. 4 threads add only ~8 MB
#        each (for their stack), so 4 threads = ~232 MB total.
#      - No serialization needed — threads can read the same Python objects.
#      - But the GIL means only one thread runs Python code at a time.
#      - Best when: tasks are I/O-bound (waiting for files, network, etc.)
#
#   3. asyncio (single thread, cooperative multitasking)
#      - Only ONE thread, ONE process. Near-zero extra memory.
#      - Tasks voluntarily yield control with "await" when they are waiting.
#      - Cannot use multiple CPU cores at all — purely for I/O-bound work.
#      - Best when: you have THOUSANDS of concurrent I/O tasks (web servers,
#        API calls) and memory efficiency matters.
#
# Summary table:
#
#   Approach        | Extra memory per task | True CPU parallelism? | GIL issue?
#   ----------------|-----------------------|-----------------------|-----------
#   multiprocessing | ~200 MB (full copy)   | YES                   | No (own GIL)
#   threading       | ~8 MB (stack only)    | NO (GIL)              | Yes
#   asyncio         | ~few KB (coroutine)   | NO (single thread)    | No
# -----------------------------------------------------------------------------


# =============================================================================
# SECTION 4: threading Module — Good for I/O-Bound Tasks
# =============================================================================
#
# A thread is created with threading.Thread()
# You give it a function (target) and arguments (args)
# .start() begins the thread
# .join() waits for it to finish before moving on
#
# Good use case: simulating file downloads or reading multiple files
# where most time is spent waiting, not computing.

print("\n" + "=" * 60)
print("SECTION 4: threading Module")
print("=" * 60)

def simulate_file_load(filename):
    """Pretend to load a file — most time is spent waiting (I/O)."""
    time.sleep(0.5)  # simulates disk read / network wait
    # Note: printing from threads is safe here but order may vary
    # print(f"  Loaded: {filename}")  # uncomment to see thread order

# --- Sequential loading ---
files = ["data_01.csv", "data_02.csv", "data_03.csv", "data_04.csv"]

start = time.time()
for f in files:
    simulate_file_load(f)
end = time.time()
print(f"  Sequential file loading: {end - start:.2f} seconds")

# --- Threaded loading ---
start = time.time()
threads = []
for f in files:
    t = threading.Thread(target=simulate_file_load, args=(f,))
    threads.append(t)
    t.start()   # all threads start almost simultaneously

for t in threads:
    t.join()    # wait for ALL threads to finish

end = time.time()
print(f"  Threaded file loading:   {end - start:.2f} seconds  <-- faster!")
print("  (Threads overlap their waiting time)")


# =============================================================================
# SECTION 5: multiprocessing Module — Good for CPU-Bound Tasks
# =============================================================================
#
# multiprocessing.Process works like threading.Thread but spawns a
# completely separate Python process — with its OWN memory and GIL.
#
# Good use case: heavy math, image processing, training ML models
#
# IMPORTANT: Always put multiprocessing code inside:
#   if __name__ == "__main__":
#
# This is because when Python spawns a new process, it re-imports this
# file. Without the guard, it would try to spawn processes infinitely.

def cpu_task(n):
    """A CPU-heavy task — count up to n doing math."""
    total = 0
    for i in range(n):
        total += i * i  # some math work
    return total

# We define cpu_task at module level (outside if __name__) so child
# processes can import it. The EXECUTION is inside the guard below.

# -----------------------------------------------------------------------------
# FIRST-PRINCIPLES: Amdahl's Law — The Fundamental Limit of Parallelism
# -----------------------------------------------------------------------------
#
# Before you throw 100 CPU cores at a problem, you need to understand
# Amdahl's Law: the speedup from parallelism is LIMITED by the serial
# (non-parallelizable) portion of your work.
#
# DERIVATION — step by step:
#
# Step 1: Define the total work
#   Let "s" = fraction of work that MUST be done sequentially (0 <= s <= 1)
#   Then (1 - s) = fraction that CAN be parallelized.
#   Total: s + (1 - s) = 1  (normalized to 1 unit of time on 1 processor)
#
# Step 2: With P processors, only the parallel part speeds up
#   Time with P processors = s + (1 - s) / P
#   (The serial part "s" takes the same time no matter how many cores you have)
#
# Step 3: Speedup = old time / new time
#   Speedup(P) = 1 / (s + (1 - s) / P)
#
# Step 4: The limit — what if you have INFINITE processors?
#   As P -> infinity, (1 - s) / P -> 0
#   Speedup(infinity) = 1 / s
#
#   So if 20% of your work is serial (s = 0.2), the MAXIMUM speedup you can
#   ever achieve is 1 / 0.2 = 5x, even with a million cores!
#
# This is why optimizing the serial bottleneck matters more than adding cores.
# In ML: data loading is often the serial bottleneck for training pipelines.
# -----------------------------------------------------------------------------

# --- Amdahl's Law: Python demonstration ---

def amdahls_law_speedup(serial_fraction, num_processors):
    """
    Calculate theoretical speedup using Amdahl's Law.

    Parameters:
        serial_fraction: fraction of work that cannot be parallelized (0 to 1)
        num_processors:  number of parallel processors

    Returns:
        Theoretical speedup factor
    """
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / num_processors)

print("\n" + "=" * 60)
print("FIRST-PRINCIPLES: Amdahl's Law — Theoretical Speedups")
print("=" * 60)

# Show how speedup changes with number of processors
# for a task that is 80% parallelizable (20% serial)
serial = 0.2
print(f"\n  For a task with {serial*100:.0f}% serial work:")
for p in [1, 2, 4, 8, 16, 64, 1000]:
    speedup = amdahls_law_speedup(serial, p)
    print(f"    {p:>4} processors -> {speedup:.2f}x speedup")

print(f"    infinite processors -> {1/serial:.2f}x speedup (theoretical max)")
print(f"\n  Notice: even 1000 processors barely improve over 64!")
print(f"  The serial 20% becomes the bottleneck.")


# =============================================================================
# SECTION 6: concurrent.futures — The Easy Modern Way
# =============================================================================
#
# concurrent.futures gives you a high-level, clean interface for both
# threads and processes. You don't need to manage .start() and .join()
# yourself.
#
# ThreadPoolExecutor  — manages a pool of threads for you
# ProcessPoolExecutor — manages a pool of processes for you
#
# The simplest way: use .map(function, list_of_inputs)
# It applies the function to each item, potentially in parallel.
#
# Think of it like Python's built-in map(), but tasks run simultaneously.

# --- ThreadPoolExecutor example ---
# (shown here, outside the __main__ guard, because threads are safe to
#  create anywhere — only multiprocessing needs the guard)

def simulate_api_call(url_id):
    """Simulate calling an API endpoint — lots of waiting."""
    time.sleep(0.4)
    return f"response_{url_id}"

print("\n" + "=" * 60)
print("SECTION 6: concurrent.futures")
print("=" * 60)

url_ids = [1, 2, 3, 4, 5]

# Sequential
start = time.time()
results = [simulate_api_call(uid) for uid in url_ids]
end = time.time()
print(f"  Sequential API calls: {end - start:.2f} seconds")

# ThreadPoolExecutor
start = time.time()
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(simulate_api_call, url_ids))
end = time.time()
print(f"  ThreadPoolExecutor:   {end - start:.2f} seconds  <-- faster!")
print(f"  Results: {results}")


# =============================================================================
# SECTION 7: Practical Example — Parallel Data Loading
# =============================================================================
#
# In machine learning, you often need to load many data files before
# training a model. If you have 20 CSV files, loading them one by one
# wastes time. Let's compare sequential vs parallel loading.
#
# We'll simulate this: each "file load" takes some time (disk/network I/O).
# We'll use both ThreadPoolExecutor and ProcessPoolExecutor and compare.

def load_csv_file(file_id):
    """
    Simulate loading one CSV file.
    In real ML work, you'd use: pandas.read_csv(filepath)
    Here we just sleep to simulate the I/O wait time.
    """
    time.sleep(0.3)  # simulate reading file from disk
    # Pretend we loaded 1000 rows
    rows = list(range(100))  # small fake dataset
    return {"file": f"data_{file_id:02d}.csv", "rows": len(rows)}


# =============================================================================
# SECTION 8: Decision Guide — When to Use What
# =============================================================================
#
#  Task type          | Best tool              | Why
#  -------------------|------------------------|-------------------------------
#  Download files     | ThreadPoolExecutor     | Waiting for network = I/O
#  Read many files    | ThreadPoolExecutor     | Waiting for disk = I/O
#  Call APIs          | ThreadPoolExecutor     | Waiting for response = I/O
#  Heavy math         | ProcessPoolExecutor    | Needs real CPU cores
#  Image processing   | ProcessPoolExecutor    | Needs real CPU cores
#  Train ML model     | ProcessPoolExecutor    | Needs real CPU cores
#  Simple scripts     | No concurrency needed  | Keep it simple!
#
#  Quick rule:
#    "Am I waiting for something external?" -> Threads
#    "Am I doing a lot of computation?"     -> Processes

print("\n" + "=" * 60)
print("SECTION 8: Decision Guide")
print("=" * 60)
print("  Waiting for files/network  ->  ThreadPoolExecutor")
print("  Heavy math / data crunch   ->  ProcessPoolExecutor")
print("  Simple, small scripts      ->  No concurrency needed")


# =============================================================================
# MAIN BLOCK — multiprocessing and ProcessPoolExecutor go here
# =============================================================================
#
# Everything that uses multiprocessing.Process or ProcessPoolExecutor
# MUST be inside this block. This prevents infinite spawning when
# child processes re-import this file.

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SECTION 5 (continued): multiprocessing.Process")
    print("=" * 60)

    # Sequential CPU work
    numbers = [5_000_000, 5_000_000, 5_000_000, 5_000_000]

    start = time.time()
    for n in numbers:
        cpu_task(n)
    end = time.time()
    print(f"  Sequential CPU tasks: {end - start:.2f} seconds")

    # Multiprocessing — each process runs on its own CPU core
    start = time.time()
    processes = []
    for n in numbers:
        p = multiprocessing.Process(target=cpu_task, args=(n,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end = time.time()
    print(f"  Multiprocessing CPU tasks: {end - start:.2f} seconds  <-- faster on multi-core!")
    print("  (Each process runs on a separate CPU core)")

    # -------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("SECTION 7: Practical Example — Parallel Data Loading")
    print("=" * 60)

    file_ids = list(range(1, 9))  # simulate 8 CSV files

    # Sequential loading
    start = time.time()
    results_seq = [load_csv_file(fid) for fid in file_ids]
    end = time.time()
    seq_time = end - start
    print(f"  Sequential loading ({len(file_ids)} files): {seq_time:.2f} seconds")

    # Parallel loading with ThreadPoolExecutor (good for I/O like disk reads)
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results_thread = list(executor.map(load_csv_file, file_ids))
    end = time.time()
    thread_time = end - start
    print(f"  ThreadPoolExecutor loading:              {thread_time:.2f} seconds")

    # Parallel loading with ProcessPoolExecutor
    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results_proc = list(executor.map(load_csv_file, file_ids))
    end = time.time()
    proc_time = end - start
    print(f"  ProcessPoolExecutor loading:             {proc_time:.2f} seconds")

    print(f"\n  Speedup with threads:   {seq_time / thread_time:.1f}x faster than sequential")
    print(f"  Speedup with processes: {seq_time / proc_time:.1f}x faster than sequential")
    print()
    print("  Note: For pure I/O simulation, threads are often just as fast")
    print("  as processes, and have less overhead (no new process to spawn).")
    print("  For real CSV files on disk, ThreadPoolExecutor is the better choice.")

    # Show the loaded data
    print("\n  Files loaded:")
    for r in results_thread:
        print(f"    {r['file']} -> {r['rows']} rows")

    # -------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
  Concurrency = doing multiple things in overlapping time
  Thread      = lightweight, shares memory, limited by GIL for CPU work
  Process     = heavyweight, separate memory, real parallelism for CPU work
  GIL         = Python rule that limits threads to one running at a time
                (but released during I/O waits, so threads still help there)

  threading.Thread         -> manual, low-level thread control
  multiprocessing.Process  -> manual, low-level process control
  ThreadPoolExecutor       -> easy thread pool, use with .map()
  ProcessPoolExecutor      -> easy process pool, use with .map()

  For ML work:
    Loading data files     -> ThreadPoolExecutor
    Preprocessing batches  -> ProcessPoolExecutor
    Training               -> GPU handles this (different story!)
    """)


# === TEST CASES ===
# These asserts verify the deterministic (non-timing) parts of the code above.
# If any assert fails, Python will tell you which one!

# amdahls_law_speedup: with 1 processor, speedup is always 1.0
assert abs(amdahls_law_speedup(0.2, 1) - 1.0) < 0.0001

# amdahls_law_speedup: known results for serial_fraction=0.2
assert abs(amdahls_law_speedup(0.2, 2) - (1.0 / (0.2 + 0.8 / 2))) < 0.0001
assert abs(amdahls_law_speedup(0.2, 4) - 2.5) < 0.0001
assert abs(amdahls_law_speedup(0.2, 8) - (1.0 / (0.2 + 0.8 / 8))) < 0.0001

# theoretical maximum speedup (infinite processors) approaches 1/serial_fraction
assert abs(amdahls_law_speedup(0.2, 1_000_000) - 5.0) < 0.01
assert abs(amdahls_law_speedup(0.5, 1_000_000) - 2.0) < 0.01
assert abs(amdahls_law_speedup(0.1, 1_000_000) - 10.0) < 0.01

# edge case: 0% serial work (fully parallel) -> speedup = num_processors
assert abs(amdahls_law_speedup(0.0, 4) - 4.0) < 0.0001

# cpu_task: deterministic output — sum of i*i for i in range(n)
# range(5) -> 0*0 + 1*1 + 2*2 + 3*3 + 4*4 = 0+1+4+9+16 = 30
assert cpu_task(5) == 30
assert cpu_task(0) == 0
assert cpu_task(1) == 0   # only i=0: 0*0=0

print("\nAll tests passed!")

# =============================================================================
# EXERCISES
# =============================================================================
#
# Try these on your own. Hints are provided — no solutions given!
#
# Exercise 1: Switch to ThreadPoolExecutor in the data loading example
# -----------------------------------------------------------------------
# In SECTION 7, the sequential loader uses a plain for loop.
# Modify the code so it uses ThreadPoolExecutor with max_workers=8
# (more workers than files — what happens? why?).
# Try different values of max_workers (1, 2, 4, 8) and observe the times.
#
# Hint: The pattern is:
#   with ThreadPoolExecutor(max_workers=N) as executor:
#       results = list(executor.map(your_function, your_list))
#
# -----------------------------------------------------------------------
# Exercise 2: Sum of squares with ProcessPoolExecutor
# -----------------------------------------------------------------------
# Write a function sum_of_squares(start, end) that returns the sum of
# squares for all numbers from start to end.
# Example: sum_of_squares(0, 5) = 0 + 1 + 4 + 9 + 16 = 30
#
# Now split the range 0 to 10_000_000 into 4 equal chunks:
#   [(0, 2500000), (2500000, 5000000), (5000000, 7500000), (7500000, 10000000)]
#
# Use ProcessPoolExecutor to compute each chunk in parallel, then add the
# results together. Compare with computing it sequentially in a single loop.
#
# Hint: executor.map() can work with a list of tuples if you use
#       executor.starmap() (look it up!), or you can use a wrapper function.
#
# -----------------------------------------------------------------------
# Exercise 3 (Derivation): Amdahl's Law — Work It Out By Hand
# -----------------------------------------------------------------------
# A data preprocessing pipeline is 80% parallelizable (the parallel part
# is reading and transforming files) and 20% serial (merging results into
# one dataset at the end).
#
# Part A: Derive the expected speedup on 4 cores.
#   - serial_fraction s = 0.2
#   - parallel_fraction (1 - s) = 0.8
#   - Time with 4 cores = s + (1 - s) / P = 0.2 + 0.8 / 4 = 0.2 + 0.2 = 0.4
#   - Speedup = 1 / 0.4 = 2.5x
#
# Part B: What if you had INFINITE cores?
#   - As P -> infinity, time = s + 0 = 0.2
#   - Speedup = 1 / 0.2 = 5.0x
#   - Even with unlimited hardware, you can never go faster than 5x!
#
# Part C: Verify your answers with code:
#   print(amdahls_law_speedup(0.2, 4))      # should print 2.5
#   print(amdahls_law_speedup(0.2, 10000))   # should print ~5.0
#
# Takeaway: before buying more GPUs, figure out what fraction of your
# pipeline is serial. If it is high, no amount of hardware will help.
# Optimize the serial part first!
#
# -----------------------------------------------------------------------
# Exercise 4 (Challenge): Threading vs Multiprocessing for CPU-bound work
# -----------------------------------------------------------------------
# Write a CPU-heavy function, for example: count_primes(limit) that
# counts how many prime numbers exist below `limit`.
#
# Run it 4 times on limit=500_000 using:
#   a) A plain for loop (sequential)
#   b) ThreadPoolExecutor with 4 workers
#   c) ProcessPoolExecutor with 4 workers
#
# Use time.time() to record how long each approach takes.
# Print the results and answer these questions in a comment:
#   - Why is threading NOT faster than sequential for this task?
#   - Why does multiprocessing help?
#   - What does this tell you about when to use each?
#
# Hint: A simple prime checker:
#   def is_prime(n):
#       if n < 2: return False
#       for i in range(2, int(n**0.5) + 1):
#           if n % i == 0: return False
#       return True
# =============================================================================
