# Mandatory Part: `src/chunking/base.py`

## 1. Overview & Architectural Role

[`src/chunking/base.py`](../src/chunking/base.py) defines the **abstract base contract** for all document chunkers and implements the **file traversal engine** responsible for discovering files in `data/raw/vllm-0.10.1/`.

It also provides the **hard size guard** that guarantees no chunk ever exceeds `--max_chunk_size` (default: 2000 characters). This is critical because the subject states:
> *"the moulinette rejects any retrieved source longer than 2000 characters (its max_context_length), and a single over-long source makes your whole output invalid."*

---

## 2. File Traversal & Binary Filtering

### 2.1 `is_binary_file(file_path: Path) -> bool`
Reads the first 8,192 bytes of a file to check for null bytes (`b"\x00"`). If found, it marks the file as binary (such as compiled `.so`, images, or model weights) and skips it.

### 2.2 `discover_files(...)` / `FileTraversalEngine`
Recursively traverses `corpus_dir` and yields valid text files:
- **Allowed extensions:** `.py` and `.md`.
- **Ignored directories:** `.git`, `.github`, `.venv`, `venv`, `__pycache__`, `build`, `dist`, `.pytest_cache`, `.mypy_cache`.
- **Hidden files:** Files starting with `.` are ignored.
- **Binary filter:** Any file failing `is_binary_file` is rejected.

---

## 3. `BaseChunker` (Abstract Class)

All language chunkers (`PythonChunker` and `MarkdownChunker`) inherit from `BaseChunker`.

### 3.1 Core Methods
- `__init__(self, max_chunk_size: int = 2000)`: Sets the character ceiling. Raises `ValueError` if `max_chunk_size <= 0`.
- `chunk_file(self, file_path: Path) -> List[InternalChunk]`: Abstract method implemented by child classes.
- `read_file_safely(file_path, encoding="utf-8") -> str`: Reads files using UTF-8; if a decode error occurs, safely falls back to `errors="replace"` to prevent unhandled crashes.
- `estimate_token_count(text: str) -> int`: Fast token estimation using regex word and punctuation boundaries (`\w+|[^\w\s]`).

### 3.2 Hard Ceiling Guards
1. `split_chunk(self, chunk: InternalChunk) -> List[InternalChunk]`:
   - If a chunk exceeds `max_chunk_size`, it subdivides the content.
   - Searches backwards from `max_chunk_size` for natural split points:
     1. Paragraph boundary (`\n\n`)
     2. Line break (`\n`)
     3. Word boundary (` `)
     4. Hard cutoff at `max_chunk_size`
   - Computes exact relative 0-based character offsets (`first_character_index`, `last_character_index`) for each slice.
2. `clamp_chunk(self, chunk: InternalChunk) -> InternalChunk`:
   - Truncates oversized chunk content to `max_chunk_size` characters and adjusts `last_character_index`.
3. `guard_chunks(self, chunks: Sequence[InternalChunk], strategy="split", reindex=True)`:
   - Iterates through a list of chunks, guarantees every single chunk is $\le \text{max\_chunk\_size}$, and optionally re-indexes `chunk_id` sequentially from `0` to `N-1`.
