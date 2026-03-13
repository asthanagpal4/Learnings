# Astha's Learning Progress — Data Structures + Python Curriculum

## About Astha
- Non-tech person, learning Linux for the first time
- Learning Python for a few days (comfortable with variables, print, if/else, basic loops)
- Long-term goal: transition to ML researcher (training and evaluating LLMs)

## Environment Setup (completed 2026-03-12)
- **OS**: Linux (Ubuntu-based, kernel 6.17)
- **Python**: 3.12.3 (system) managed via `uv`
- **uv** (v0.10.9): installed at `~/.local/bin/uv`
- **Project location**: `/home/astha/Documents/Puchku/`
- **Virtual environment**: `.venv/` created by `uv init --python 3.12`
- **Installed packages**: numpy (2.4.3), torch (PyTorch — for Section 8)
- **How to run files**: `uv run python <filename>` (ensures the .venv with numpy etc. is used)

## Curriculum Structure

All lesson files are self-contained, runnable `.py` files with:
- Concept explanations in comments
- Working code examples with printed output
- Exercises at the bottom (hints but no solutions) for Astha to complete
- Mini-projects at the end of each section

### Section 1: Foundations (`01_foundations/`) — 8 files
| File | Topic | Status |
|------|-------|--------|
| `01_lists.py` | Lists: creating, indexing, slicing, mutability, methods | Completed |
| `02_loops_and_lists.py` | Iterating, enumerate, list patterns, nested loops | Completed |
| `03_functions.py` | Defining functions, parameters, return values, scope | Completed |
| `04_strings.py` | String methods, formatting, immutability | Completed |
| `05_tuples.py` | Packing/unpacking, immutability, tuples vs lists | Not started |
| `06_dictionaries.py` | Key-value pairs, lookups, iteration, nesting | Not started |
| `07_sets.py` | Uniqueness, set operations, when to use sets | Not started |
| `project_word_counter.py` | **Mini-project**: Word frequency counter (NLP taste) | Not started |

### Section 2: Programming Concepts (`02_programming/`) — 7 files
| File | Topic | Status |
|------|-------|--------|
| `01_file_handling.py` | Reading/writing files (dataset loading prep) | Not started |
| `02_error_handling.py` | try/except, raising errors, defensive programming | Not started |
| `03_list_comprehensions.py` | List/dict/set comprehensions | Not started |
| `04_classes_basics.py` | Classes, `__init__`, methods, self, attributes | Not started |
| `05_classes_advanced.py` | Inheritance, `__str__`, `__len__`, special methods | Not started |
| `06_modules.py` | Organizing code, `__name__`, importing | Not started |
| `project_student_records.py` | **Mini-project**: Student grade tracker (mini-database) | Not started |

### Section 3: Core Data Structures (`03_data_structures/`) — 7 files
| File | Topic | Status |
|------|-------|--------|
| `01_stacks.py` | LIFO, implementing with list & class | Not started |
| `02_queues.py` | FIFO, `collections.deque` | Not started |
| `03_linked_list.py` | Nodes, pointers, building from scratch | Not started |
| `04_trees.py` | Binary tree, traversals (inorder, preorder, postorder) | Not started |
| `05_graphs_intro.py` | Adjacency list, BFS/DFS basics | Not started |
| `06_hash_tables.py` | How dicts work under the hood, collision handling | Not started |
| `project_maze_solver.py` | **Mini-project**: Maze solver using BFS/DFS | Not started |

### Section 4: Algorithms & ML Prep (`04_algorithms/`) — 7 files
| File | Topic | Status |
|------|-------|--------|
| `01_sorting.py` | Bubble sort, merge sort, `sorted()`, time complexity intro | Not started |
| `02_searching.py` | Linear search, binary search | Not started |
| `03_recursion.py` | Base case, recursive case, stack frames, tree recursion | Not started |
| `04_big_o.py` | Big-O notation with timed measurements | Not started |
| `05_numpy_intro.py` | Arrays vs lists, vectorized ops (bridge to ML) | Not started |
| `06_multiprocessing.py` | Processes vs threads, GIL, threading, multiprocessing, concurrent.futures | Not started |
| `project_data_pipeline.py` | **Mini-project**: CSV loading, cleaning, stats (ML preprocessing) | Not started |

### Section 5: Math for ML (`05_math_for_ml/`) — 6 files
| File | Topic | Status |
|------|-------|--------|
| `01_vectors_and_matrices.py` | Vectors, matrices, dot product, matrix multiply, transpose | Not started |
| `02_matrix_operations.py` | Element-wise ops, broadcasting, reshape, stacking, normalization, one-hot encoding | Not started |
| `03_functions_and_derivatives.py` | Math functions, derivatives, numerical differentiation, sigmoid/ReLU/tanh | Not started |
| `04_gradient_descent.py` | Loss functions, MSE, gradients, gradient descent algorithm, linear regression from scratch | Not started |
| `05_probability_basics.py` | Probability, distributions, softmax, cross-entropy loss | Not started |
| `project_linear_regression.py` | **Mini-project**: Complete linear regression pipeline with train/test split, evaluation, predictions | Not started |

### Section 6: Neural Networks (`06_neural_networks/`) — 5 files
| File | Topic | Status |
|------|-------|--------|
| `01_perceptron.py` | Single neuron, step activation, train AND/OR gates, XOR failure | Not started |
| `02_multilayer_network.py` | Multi-layer structure, sigmoid, weight matrices, forward pass, batch processing | Not started |
| `03_backpropagation.py` | Chain rule, sigmoid derivative, full backprop on XOR, loss decreasing | Not started |
| `04_training_loop.py` | Complete training recipe, batching, learning rate effects, circle classifier | Not started |
| `project_digit_classifier.py` | **Mini-project**: 3-layer MLP classifying synthetic 3x5 pixel digits 0-4, one-hot encoding, softmax, cross-entropy | Not started |

### Section 7: NLP & Tokenization (`07_nlp_tokenization/`) — 6 files
| File | Topic | Status |
|------|-------|--------|
| `01_text_processing.py` | Text as data, ASCII/UTF-8, `ord()`/`chr()`, text cleaning, vocabulary, word frequency with Counter | Not started |
| `02_character_level_model.py` | Character bigram model: count pairs, convert to probabilities, sample to generate text | Not started |
| `03_word_tokenization.py` | Word vs character tokenization, OOV problem, vocabulary size explosion, subword preview | Not started |
| `04_bpe_algorithm.py` | BPE step by step: count pairs, find best pair, merge, record rules, apply to new words | Not started |
| `05_bpe_tokenizer_complete.py` | Full BPETokenizer class: train(), encode(), decode(), get_vocab(), roundtrip demo | Not started |
| `project_tokenizer_analysis.py` | **Mini-project**: Compare BPE at 6 vocab sizes (10–200 merges), compression ratio table | Not started |

### Section 8: Transformers (`08_transformers/`) — 6 files
| File | Topic | Status |
|------|-------|--------|
| `01_pytorch_basics.py` | PyTorch tensors, autograd, nn.Module, training loop (linear regression) | Not started |
| `02_embeddings.py` | One-hot encoding, nn.Embedding, sinusoidal positional encoding, full pipeline | Not started |
| `03_attention_mechanism.py` | Q/K/V intuition, scaled dot-product attention, causal masking, MultiHeadAttention | Not started |
| `04_transformer_block.py` | LayerNorm, FFN, residual connections, TransformerBlock, full MiniTransformerLM | Not started |
| `05_training_pipeline.py` | CharDataset, DataLoader, training loop, text generation (greedy/temperature/top-k) | Not started |
| `project_pretrain_mini_llm.py` | **Capstone**: Pre-train ~2M param transformer LLM from scratch, AdamW, LR warmup, generation | Not started |

## How to Work Through the Curriculum
1. Run each file in order: `uv run python 01_foundations/01_lists.py`
2. Read the output, then open the file to study the code and comments
3. Complete the exercises at the bottom of each file
4. After finishing all lessons in a section, tackle the mini-project
5. Move to the next section

## Notes for Claude Code Instances
- When Astha asks for help with an exercise, read the specific file first to understand the context
- She's a beginner — keep explanations simple, avoid jargon
- Update the "Status" column in this file as she progresses (Not started → In progress → Completed)
- The files are designed to be self-contained — avoid modifying lesson files unless Astha asks
- If she needs a new package, install with `uv add <package>`
- All 28 original files verified to run without errors as of 2026-03-12
- Sections 5-8 (25 new files) added 2026-03-12 to build toward LLM pre-training
- First-principles thinking added to all 24 files in Sections 4-8 (2026-03-12):
  - Step-by-step mathematical derivations (sigmoid derivative, MSE gradient, attention scaling, etc.)
  - Big-O time and space complexity analysis for every algorithm
  - "Why does this work?" informal correctness arguments and proofs
  - Derivation exercises for deeper understanding (e.g., Amdahl's Law, Heaps' Law, chain rule)
- Section 5 (Math for ML): uses only numpy
- Section 6 (Neural Networks): uses only numpy
- Section 7 (NLP & Tokenization): uses only stdlib (re, collections)
- Section 8 (Transformers): requires PyTorch — install with `uv add torch`
- All files are self-contained and run on CPU
