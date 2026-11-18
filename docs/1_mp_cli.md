# Mandatory Part: `src/cli.py` & `src/__main__.py`

## 1. Overview & Architectural Role

The Command-Line Interface (CLI) is the user- and grader-facing entrypoint of the RAG system.
According to the 42 subject specification:
- Every command is invoked as `uv run python -m src <command> [options]`.
- The CLI is powered by **Python Fire**.
- All file paths (`--dataset_path`, `--save_directory`, `--student_search_results_path`) must be configurable arguments and never hard-coded.
- Progress bars (`tqdm`) must be shown for long-running operations.
- Degenerate inputs (empty queries, nonsensical queries, $k \le 0$, missing files, malformed JSON) must be handled gracefully without unhandled tracebacks.

The CLI router is implemented in [`src/cli.py`](../src/cli.py), and the module execution entrypoint is in [`src/__main__.py`](../src/__main__.py).

---

## 2. File: `src/__main__.py`

### Role
When a user runs `uv run python -m src <command>`, Python searches for the package `src` and executes its `__main__.py` file as the top-level script (`__name__ == "__main__"`).

### Code Walkthrough
```python
from __future__ import annotations
import sys
from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
```
- It imports `main` from `src.cli` and calls `sys.exit(main())`.
- If `main()` returns an integer (e.g. `0` for success or `1` for error), `sys.exit()` sets the process exit code accordingly.
- It also re-exports all CLI symbols (`RAGCLI`, `index`, `search`, etc.) so external scripts can import from `src`.

---

## 3. File: `src/cli.py`

### 3.1 Constants and Defaults
```python
DEFAULT_SEARCH_K: int = 5
DEFAULT_DATASET_SEARCH_K: int = 10
DEFAULT_ANSWER_K: int = 5
DEFAULT_MAX_CHUNK_SIZE: int = 2000
```
These match the subject specification defaults for single search, batch search, answer generation, and maximum character chunk size.

### 3.2 Output Serialization & Fire Compatibility Wrappers
Python Fire prints command outputs using its internal `_PrintResult` function. By default, Fire checks `value_types.HasCustomStr(result)` and calls `str(result)`.
To guarantee that:
1. Output conforms 100% to Pydantic JSON schemas.
2. Fire does not open an interactive terminal pager (`less`) or display raw Python `repr(...)`.
3. Running commands from code returns real Pydantic model objects while running from CLI prints clean JSON.

`src/cli.py` defines 4 result wrapper classes that subclass the official Pydantic models:
1. `CLISearchResult(MinimalSearchResults)`:
   - Contains single-item fields (`question_id`, `question`, `retrieved_sources`) and dual batch compatibility fields (`search_results`, `k`).
   - Overrides `__str__` to return formatted JSON.
2. `CLISearchDatasetResult(StudentSearchResults)`:
   - Holds `search_results` and `k`.
   - Overrides `__str__` to return `"Saved student_search_results to <path>"`.
3. `CLIAnswerResult(MinimalAnswer)`:
   - Holds single-item fields + `answer`.
   - Overrides `__str__` to return formatted JSON.
4. `CLIAnswerDatasetResult(StudentSearchResultsAndAnswer)`:
   - Holds batch `search_results` with answers and `k`.
   - Overrides `__str__` to return `"Saved student_search_results_and_answer to <path>"`.

### 3.3 The Core Router: `RAGCLI`
The `RAGCLI` class encapsulates the 6 mandatory commands as methods:

#### `__init__(self, ...)`
- Configures default directories (`corpus_dir="data/raw/vllm-0.10.1"`, `output_dir="data/processed"`), models (`"Qwen/Qwen3-0.6B"`), and devices (`"cpu"`).
- All arguments are **keyword-only** (`*`), preventing Python Fire from accidentally treating subcommand names (`index`, `search`) as constructor arguments.

#### 1. `index(self, max_chunk_size=2000, corpus_dir=None, output_dir=None, ...)`
- **CLI:** `uv run python -m src index [--max_chunk_size 2000]`
- **Action:** Runs `CorpusIndexer(corpus_dir, output_dir, max_chunk_size).index()`.
- **Output:** Writes `chunks.json`, `bm25_index.pkl`, and `embeddings.npy` to `data/processed/`.
- **Stdout:** `Ingestion complete! Indices saved under data/processed`

#### 2. `search(self, query, k=5, output_dir=None, ...)`
- **CLI:** `uv run python -m src search <query> [--k 5]`
- **Action:** Lazily loads `BM25Index` from `data/processed/`, queries top-$k$ relevant chunks, clamps any oversized spans to `max_chunk_size`, and formats output.
- **Stdout:** JSON conforming to `MinimalSearchResults` and `StudentSearchResults`.

#### 3. `search_dataset(self, dataset_path, save_directory, k=10, ...)`
- **CLI:** `uv run python -m src search_dataset --dataset_path <p> --save_directory <d> [--k 10]`
- **Action:** Reads the `RagDataset` JSON (e.g. `dataset_docs_public.json`), iterates over all questions with a `tqdm` progress bar, retrieves top-$k$ sources for each question, and constructs a `StudentSearchResults` object.
- **Output File:** Writes pure `StudentSearchResults` JSON to `<save_directory>/<filename>`.
- **Stdout:** `Saved student_search_results to <save_directory>/<filename>`

#### 4. `answer(self, query, k=5, ...)`
- **CLI:** `uv run python -m src answer <query> [--k 5]`
- **Action:** Retrieves top-$k$ sources via BM25, loads `LLMInferenceEngine` (`Qwen/Qwen3-0.6B` on CPU), builds the grounded prompt, runs inference, and returns a `MinimalAnswer`.
- **Stdout:** Formatted JSON containing question, retrieved sources, and generated answer.

#### 5. `answer_dataset(self, student_search_results_path, save_directory, ...)`
- **CLI:** `uv run python -m src answer_dataset --student_search_results_path <p> --save_directory <d>`
- **Action:** Loads previously generated `StudentSearchResults` JSON, iterates through each question, generates grounded answers with `Qwen3-0.6B`, and writes a `StudentSearchResultsAndAnswer` JSON file.
- **Output File:** Writes `<save_directory>/<filename>`.
- **Stdout:** `Loaded N questions ... Processed N of N questions\nSaved student_search_results_and_answer to <path>`

#### 6. `evaluate(self, student_search_results_path, dataset_path, k=5, ...)`
- **CLI:** `uv run python -m src evaluate --student_search_results_path <p> --dataset_path <p> [--k 5]`
- **Action:** Loads student search results and ground truth `AnsweredQuestions`, computes character-span IoU $\ge 0.05$ matches, and prints a Recall@k table for verification.

---

## 4. Degenerate Input Handling

The subject requires that edge cases **never crash with an unhandled traceback**:
- **Empty Query (`""` or `"   "`):** Caught in `search` and `answer`. Returns empty sources list and fallback answer without calling the engine.
- **Negative / Zero $k$ (`k=0`, `k=-1`):** `_parse_k` clamps effective search $k$ to 0 and schema $k$ to $\ge 1$ to preserve valid Pydantic models.
- **Oversized Chunk Ceiling (`max_chunk_size > 2000` or `max_chunk_size <= 0`):** `_sanitize_max_chunk_size` clamps values into $[1, 2000]$.
- **Missing or Corrupt Dataset JSON:** Caught with `try-except (FileNotFoundError, json.JSONDecodeError, ValueError)`. Logs clear error messages to `sys.stderr` and returns clean empty models.
