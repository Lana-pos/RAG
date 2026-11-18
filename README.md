*This project has been created as part of the 42 curriculum by spostica.*

# RAG Against the Machine

## Description
A deterministic, privacy-preserving Retrieval-Augmented Generation (RAG) system engineered to index and query the vLLM codebase (`vllm-0.10.1`), retrieve character-precise evidence spans within a 2000-character ceiling, and generate grounded natural language answers using a local `Qwen/Qwen3-0.6B` model.

For full technical specifications, component interactions, and Architecture Decision Records (ADRs), see the [Technical Architecture Document](docs/ARCHITECTURE.md).

---

## Instructions

### Installation
Dependencies are managed deterministically with `uv`:
```bash
make install
# or directly:
uv sync
```

### Execution
Run the CLI using `uv run python -m src <command>`:
```bash
# Display CLI help
make run
# or
uv run python -m src --help

# Ingest & index the vLLM repository
uv run python -m src index --max_chunk_size 2000

# Search top-k sources for a query (e.g. k=10 or k=5)
uv run python -m src search "How to configure OpenAI server?" --k 10

# Batch search over a dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# Generate an answer for a single query (evaluator live demo with Qwen/Qwen3-0.6B)
uv run python -m src answer "How to configure OpenAI server?" --k 10

# Batch generate answers for a dataset
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions

# Evaluate recall@k against ground-truth
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

### Linting & Type Checking
```bash
make lint
# or strict mode:
make lint-strict
```

### Unit Tests
```bash
make test
# or directly:
uv run pytest
```

### Debugging & Cleanup
```bash
# Run in debug mode via pdb
make debug

# Clean temporary caches (__pycache__, .mypy_cache, .pytest_cache)
make clean
```

### Moulinette Import Compliance
Verify that no `moulinette` packages are imported anywhere in the project (mandatory scale check):
```bash
grep -rnE "^[[:space:]]*(import|from)[[:space:]]+moulinette" src/ tests/ --include="*.py"
```

---

## System Architecture
The system follows a clean, modular architecture decoupled across ingestion, indexing, retrieval, and generation:
- **Ingestion & Segmentation:** Dual chunking pipeline (`src/chunking/`) with an AST-based Python chunker and a heading-aware Markdown/text chunker.
- **Lexical Indexing (BM25):** Vectorized BM25Okapi inverted index (`src/retrieval/bm25.py`) with code-specific sub-word tokenization, file-path token boosting, and parent heading context preservation.
- **Dense Semantic Retrieval:** 384-dimensional dense vector embeddings generated with `all-MiniLM-L6-v2` (`src/retrieval/dense.py`).
- **Hybrid Fusion:** Reciprocal Rank Fusion (RRF, $k=60$) combining lexical and dense candidate ranks (`src/retrieval/hybrid.py`).
- **Context Augmentation & Inference:** Grounded prompt injection with CPU-optimized Hugging Face inference for `Qwen/Qwen3-0.6B` (`src/generation/`).
- **Data Exchange:** Strict Pydantic v2 schemas (`src/models.py`) validated against moulinette requirements.

For complete C4 system diagrams and flowcharts, refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Chunking Strategy
- **Python Code (`src/chunking/python.py`):** Uses Python's standard `ast` module to detect class, function, and async function nodes. Preserves syntactic integrity and maps line numbers to exact byte/character offsets (`first_character_index`, `last_character_index`).
- **Markdown & Text Documentation (`src/chunking/markdown.py`):** Splits text hierarchically on `#`, `##`, `###` headings while preserving code fences (` ``` `) and related paragraphs intact, strictly enforcing the $\le 2000$ character ceiling.

---

## Retrieval Method & Ranking Mechanism
1. **Lexical Retrieval:** Uses BM25Okapi ($k_1=1.5, b=0.4$). Queries and documents undergo code-aware tokenization (splitting `camelCase` and `snake_case` identifiers into sub-tokens while retaining full composite symbols), file-path token enrichment, parent heading context preservation, and conversational query stopword filtering.
2. **Semantic Retrieval:** `all-MiniLM-L6-v2` encodes questions and chunks into dense vectors for cosine similarity search.
3. **Hybrid RRF Ranking:** Lexical and semantic candidate ranks are fused using Reciprocal Rank Fusion:
   $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + \text{Rank}_m(d)}$$
   This guarantees that exact API identifiers and broader natural language concepts both retrieve high-relevance chunks.

---

## Performance Analysis
- **Indexing SLA:** Processes $1,927$ files ($28,385$ chunks) in $\sim 8\text{ seconds}$ on CPU ($\le 5\text{ minutes}$ SLA).
- **Retrieval Throughput:** Processes 200 batch questions in $< 0.2\text{ seconds}$ ($\le 90\text{ seconds}$ SLA).
- **Recall@k Results (Evaluated via 42 Moulinette at $\text{IoU} \ge 0.05$):**
  - **Documentation Dataset (`dataset_docs_public.json`):** Recall@1 = `66.0%`, Recall@3 = `79.0%`, **Recall@5 = `87.0%`** (exceeds $\ge 80\%$ target), Recall@10 = `89.0%`.
  - **Code Dataset (`dataset_code_public.json`):** Recall@1 = `56.6%`, Recall@3 = `73.7%`, **Recall@5 = `82.8%`** (exceeds $\ge 50\%$ target), Recall@10 = `89.9%`.
- **Effect of `--max_chunk_size` on Recall@5:**
  - `--max_chunk_size 2000` (default): Docs Recall@5 = `87.0%`, Code Recall@5 = `82.8%`.
  - `--max_chunk_size 1000`: Docs Recall@5 = `83.0%`, Code Recall@5 = `78.8%` (smaller chunks increase granularity for short spans but split larger classes/tables across multiple chunks).
  - `--max_chunk_size 500`: Docs Recall@5 = `76.0%`, Code Recall@5 = `71.7%` (fragmenting context reduces lexical co-occurrence per chunk).

---

## Design Decisions
Documented as formal Architecture Decision Records (ADRs) in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#17-architecture-decision-records-adrs):
- **ADR-001:** Pure Python/NumPy BM25Okapi for zero-daemon portability and sub-millisecond lookup.
- **ADR-002:** Dual AST and Markdown Header chunking rather than fixed-window character slicing.
- **ADR-003:** Reciprocal Rank Fusion (RRF) for scale-invariant hybrid ranking.
- **ADR-004:** Hugging Face PyTorch CPU inference (`Qwen/Qwen3-0.6B`) to prevent compilation issues during `uv sync`.
- **ADR-005:** File-path and parent-heading lexical enrichment with calibrated length normalization ($b=0.4$).

---

## Challenges Faced
- **Character Span Overflow:** Fixed by adding a pre-persistence hard clamp and logical block splitter in the base chunker preventing any snippet from exceeding 2000 characters.
- **Code Identifier Splitting:** Composite identifiers (e.g. `openai_compatible_server`, `AsyncEngineArgs`) caused vocabulary mismatch; solved by implementing a specialized sub-token code tokenizer (`src/retrieval/tokenizer.py`).
- **Conversational Query Noise in Mixed Code/Docs Corpus:** Common English question words (`how`, `where`, `what`, `can`, `you`) are rare in Python source files and thus receive artificially high IDF in a combined code+docs corpus; solved by filtering conversational stopwords and enriching chunks with file-path and section-title tokens.
- **Relative Path Matching:** Moulinette compares paths verbatim; solved by strictly normalizing paths relative to project root (`data/raw/vllm-0.10.1/...`) with forward slashes.

---

## Example Usage

### Full Pipeline & Checking Recall@5 (Docs `87.0%` / Code `82.8%`)
```bash
# 1. Build the corpus index (default max_chunk_size=2000)
uv run python -m src index --max_chunk_size 2000
# Output: Ingestion complete! Indices saved under data/processed/

# 2. Single-query search
uv run python -m src search "How to configure OpenAI server?" --k 5

# 3. Run batch search on Both Public Datasets (Docs & Code)
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 4A. Check Recall@5 with the official 42 Moulinette binary
# Docs dataset (Target >= 80% -> Achieves 87.0% Recall@5)
./moulinette/moulinette-ubuntu evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000

# Code dataset (Target >= 50% -> Achieves 82.8% Recall@5)
./moulinette/moulinette-ubuntu evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k 10 --max_context_length 2000

# 4B. Or check Recall@5 with the built-in CLI evaluator
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k 5

# 5. Single-query grounded answer generation (Qwen/Qwen3-0.6B)
uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5

# 6. Batch dataset answer generation
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### System Reliability & Edge Cases (`exam_edge_cases.sh`)
The system handles all degenerate and edge case inputs gracefully without Python tracebacks:
```bash
# 1. Empty query (returns empty sources with exit code 0)
uv run python -m src search "" --k 10

# 2. Gibberish query (returns 0 matching sources with exit code 0)
uv run python -m src search "asdfghjkl" --k 10

# 3. Non-positive k=0 (returns fallback answer with schema k=1, exit code 0)
uv run python -m src answer "What is vLLM?" --k 0

# 4. Nonexistent dataset path (reports clean error message, returns empty batch schema)
uv run python -m src search_dataset --dataset_path /nonexistent.json
```

---

## Bonus Features (Implemented & How to Run)

Two bonus features from Chapter IX are implemented and integrated into the CLI:

1. **Bonus 1 — Semantic Embeddings (`src/retrieval/dense.py`):** Builds a 384-dimensional normalized vector index (`data/processed/embeddings.npy`) on CPU using `sentence-transformers/all-MiniLM-L6-v2` alongside the lexical BM25 index (`data/processed/bm25_index.pkl`).
2. **Bonus 2 — Hybrid Retrieval (`src/retrieval/hybrid.py`):** Combines lexical BM25 rankings and dense semantic cosine-similarity rankings into a single ranked list using Reciprocal Rank Fusion ($\text{RRF}, k=60$).

### Commands to Run & Demonstrate the 2 Bonus Parts
```bash
# 0. Verify lightweight CPU semantic embedding generation (all-MiniLM-L6-v2):
uv run python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); emb = m.encode(['hello world']); print('Dense vector shape:', emb.shape)"

# 1. Build BOTH the BM25 lexical index and the Dense vector index (embeddings.npy)
uv run python -m src index --max_chunk_size 2000 --enable_dense True

# Verify the generated dense vector index file alongside BM25:
ls -lh data/processed/
# -> chunks.json, bm25_index.pkl, embeddings.npy

# 2. Run Hybrid Retrieval (BM25 + Dense Semantic RRF) on a single query
uv run python -m src search "How to configure OpenAI server?" --k 5 --enable_dense True

# Compare against Lexical-only (BM25 only, dense disabled):
uv run python -m src search "How to configure OpenAI server?" --k 5 --enable_dense False

# 3. Run Hybrid Retrieval over a full dataset and evaluate Recall@5
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions \
  --enable_dense True

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5
```

---

## Resources
- [vLLM Documentation](https://docs.vllm.ai/)
- [Okapi BM25 Specification](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Qwen/Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Sentence-Transformers](https://sbert.net/)

### AI Usage
AI was utilized to produce the comprehensive system architecture, formulate the ADRs, structure the Pydantic data schemas, and design the AST syntax-aware chunking pipeline.
