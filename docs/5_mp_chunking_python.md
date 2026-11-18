# Mandatory Part: `src/chunking/python.py`

## 1. Overview & Architectural Role

[`src/chunking/python.py`](../src/chunking/python.py) implements the **AST-based Python Code Chunker** required by Chapter VI.1:
> *"A Python file and a Markdown page do not break apart the same way, so your program must implement two distinct chunking strategies: Python code chunking, Markdown / text chunking."*

Rather than slicing arbitrary lines that fracture class or function scopes, `PythonChunker` parses Python source code into syntactically meaningful units using the standard `ast` module.

---

## 2. Syntactic Unit Extraction

### 2.1 AST Traversal
`PythonChunker.chunk_file(file_path)` parses the file into an Abstract Syntax Tree via `ast.parse`:
1. **Module Docstrings & Imports:** If a module starts with a docstring or package-level declarations before classes/functions, they are captured as `python_module`.
2. **Classes (`ast.ClassDef`):** Captured as discrete `python_class` units.
3. **Functions (`ast.FunctionDef`, `ast.AsyncFunctionDef`):** Top-level standalone functions and async coroutines are captured as `python_func` units.
4. **Remaining Statements:** Any loose statements between blocks are grouped together cleanly.

### 2.2 Exact Character Span Mapping
Python's `ast` nodes provide line and column coordinates (`lineno`, `end_lineno`, `col_offset`, `end_col_offset`).
`PythonChunker` maps these coordinates to exact **0-based UTF-8 character offsets** (`first_character_index`, `last_character_index`) matching the file on disk:
- It precomputes a lookup table of line start offsets:
  ```python
  line_offsets = [0]
  for line in lines:
      line_offsets.append(line_offsets[-1] + len(line))
  ```
- The node span is computed using:
  ```python
  start_char = line_offsets[node.lineno - 1] + node.col_offset
  end_char = line_offsets[node.end_lineno - 1] + node.end_col_offset
  ```

---

## 3. Handling Oversized Python Code ($\le 2000$ Characters)

When a class or large function exceeds `max_chunk_size` (2000 characters):
1. **Class Decomposition:**
   - The class signature and docstring are extracted as an introductory chunk.
   - Individual methods within the class (`def __init__`, `def forward`, etc.) are segmented into separate method-level chunks.
2. **Sub-Function Block Splitting:**
   - If an individual function still exceeds 2000 characters, it is partitioned along logical statement boundaries (loops, conditional branches, or sequential statements) while preserving valid indentation.
3. **Hard Ceiling Guard:**
   - Finally, `BaseChunker.guard_chunks()` guarantees that 100% of generated chunks strictly respect the $\le 2000$ character ceiling.
